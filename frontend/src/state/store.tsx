import { createContext, useCallback, useContext, useEffect, useReducer, useRef } from "react";
import { api } from "../api/client";
import type { Conversation, Message, Usage } from "../api/types";

export interface AppState {
  conversations: Conversation[];
  selectedId: string | null;
  messages: Message[];
  pending: boolean;
  error: string | null;
  failedText: string | null;
  searchQuery: string;
  creatingConversation: boolean;
}

export const initialState: AppState = {
  conversations: [],
  selectedId: null,
  messages: [],
  pending: false,
  error: null,
  failedText: null,
  searchQuery: "",
  creatingConversation: false,
};

export type Action =
  | { type: "conversations_loaded"; conversations: Conversation[] }
  | { type: "selected"; id: string | null; messages: Message[] }
  | { type: "send_started"; userText: string }
  | { type: "send_succeeded"; message: Message; usage: Usage | null }
  | { type: "send_failed"; error: string; userText: string }
  | { type: "error_dismissed" }
  | { type: "search_changed"; query: string }
  | { type: "conversation_create_started" }
  | { type: "conversation_create_finished" };

export function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "conversations_loaded":
      return { ...state, conversations: action.conversations };
    case "selected":
      return {
        ...state,
        selectedId: action.id,
        messages: action.messages,
        error: null,
        // Switching conversations invalidates any pending retry: the
        // stashed text belonged to whatever conversation was selected when
        // it failed to send.
        failedText: null,
      };
    case "send_started": {
      const optimistic: Message = {
        id: `local-${Date.now()}`,
        conversation_id: state.selectedId ?? "",
        role: "user",
        content: action.userText,
        latency_ms: null,
        created_at: Date.now() / 1000,
      };
      return {
        ...state,
        pending: true,
        error: null,
        failedText: null,
        messages: [...state.messages, optimistic],
      };
    }
    case "send_succeeded":
      return {
        ...state,
        pending: false,
        messages: [...state.messages, { ...action.message, usage: action.usage }],
      };
    case "send_failed":
      // A failed send never made it into history server-side, so the
      // optimistic user message it appended is orphaned — drop it, and
      // stash its text so the error toast's retry button can re-send it
      // through the normal send flow.
      return {
        ...state,
        pending: false,
        error: action.error,
        failedText: action.userText,
        messages: state.messages.slice(0, -1),
      };
    case "error_dismissed":
      return { ...state, error: null };
    case "search_changed":
      return { ...state, searchQuery: action.query };
    case "conversation_create_started":
      return { ...state, creatingConversation: true };
    case "conversation_create_finished":
      return { ...state, creatingConversation: false };
    default:
      return state;
  }
}

const AppContext = createContext<ReturnType<typeof buildValue> | null>(null);

/**
 * Several actions (create, rename, delete, send, search) all end with a
 * conversation-list refresh, and nothing stops two of those from being in
 * flight at once — e.g. renaming one conversation right after creating
 * another. Without a guard, whichever GET happens to resolve last "wins"
 * and overwrites state, even if it was the one that fired first and is now
 * stale. `refreshSeqRef` is a simple sequence number: each refresh grabs
 * the next value, and only applies its result if it is still the most
 * recent refresh in flight by the time its response comes back.
 *
 * `selectSeqRef` guards `select()` the same way: clicking one conversation
 * row and then quickly clicking another fires two `listMessages` requests,
 * and nothing about network timing guarantees they resolve in the order
 * they were sent. Without this guard, a slow response for the *first*
 * click could resolve after the second click's `selected` dispatch and
 * silently replace the newly-selected conversation's messages with the
 * previous one's.
 */
export function buildValue(
  state: AppState,
  dispatch: React.Dispatch<Action>,
  refreshSeqRef: { current: number },
  selectSeqRef: { current: number },
) {
  const refresh = async (query = state.searchQuery) => {
    const requestId = ++refreshSeqRef.current;
    const conversations = await api.listConversations(query || undefined);
    if (requestId !== refreshSeqRef.current) return; // a newer refresh has since started
    dispatch({ type: "conversations_loaded", conversations });
  };
  const select = async (id: string | null) => {
    const requestId = ++selectSeqRef.current;
    const messages = id ? await api.listMessages(id) : [];
    if (requestId !== selectSeqRef.current) return; // a newer select has since started
    dispatch({ type: "selected", id, messages });
  };
  const actions = {
    refresh,
    select,
    newConversation: async () => {
      // Guards against a burst of clicks firing several concurrent creates
      // before the first one lands.
      if (state.creatingConversation) return;
      dispatch({ type: "conversation_create_started" });
      try {
        const conversation = await api.createConversation();
        await refresh();
        await select(conversation.id);
      } finally {
        dispatch({ type: "conversation_create_finished" });
      }
    },
    rename: async (id: string, title: string) => {
      await api.renameConversation(id, title);
      await refresh();
    },
    remove: async (id: string) => {
      await api.deleteConversation(id);
      await refresh();
      if (state.selectedId === id) await select(null);
    },
    send: async (text: string) => {
      if (!state.selectedId || state.pending || !text.trim()) return;
      dispatch({ type: "send_started", userText: text });
      try {
        const resp = await api.sendChat({ conversation_id: state.selectedId, message: text });
        dispatch({ type: "send_succeeded", message: resp.message, usage: resp.usage });
        // Titles/order may have changed, but the reply is already on screen:
        // refresh the sidebar in the background instead of holding up send.
        void refresh().catch(() => {});
      } catch (err) {
        dispatch({ type: "send_failed", error: (err as Error).message, userText: text });
      }
    },
    retryLast: async () => {
      // Re-sends the text stashed by the last failed send. Routing this
      // through the normal `send` flow means `send_started` clears both
      // `error` and `failedText` as a side effect of starting the new
      // attempt, and a second failure re-stashes correctly.
      const text = state.failedText;
      if (!text) return;
      await actions.send(text);
    },
    setPinned: async (id: string, pinned: boolean) => {
      await api.setPinned(id, pinned);
      await refresh();
    },
    dismissError: () => dispatch({ type: "error_dismissed" }),
    search: async (query: string) => {
      dispatch({ type: "search_changed", query });
      await refresh(query);
    },
  };
  return { state, actions };
}

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const refreshSeqRef = useRef(0);
  const selectSeqRef = useRef(0);
  const value = buildValue(state, dispatch, refreshSeqRef, selectSeqRef);
  const refreshOnce = useCallback(value.actions.refresh, []);
  useEffect(() => {
    void refreshOnce();
  }, [refreshOnce]);
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp outside provider");
  return ctx;
}
