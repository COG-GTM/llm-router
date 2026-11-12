import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { Message } from "../api/types";
import { buildValue, initialState } from "./store";

vi.mock("../api/client", () => ({
  api: {
    listConversations: vi.fn(),
    listMessages: vi.fn(),
    sendChat: vi.fn(),
  },
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

function message(overrides: Partial<Message> = {}): Message {
  return {
    id: "m1",
    conversation_id: "c1",
    role: "user",
    content: "hi",
    latency_ms: null,
    created_at: 0,
    ...overrides,
  };
}

describe("buildValue: select() race guard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("ignores a slow listMessages response that resolves under a newer selection", async () => {
    const slow = deferred<Message[]>();
    const fast: Message[] = [message({ id: "m2", conversation_id: "c2" })];
    vi.mocked(api.listMessages)
      .mockImplementationOnce(() => slow.promise) // select("c1")
      .mockImplementationOnce(async () => fast); // select("c2")

    const dispatch = vi.fn();
    const refreshSeqRef = { current: 0 };
    const selectSeqRef = { current: 0 };
    const { actions } = buildValue(initialState, dispatch, refreshSeqRef, selectSeqRef);

    const firstSelect = actions.select("c1"); // stays pending
    const secondSelect = actions.select("c2"); // resolves before the first
    await secondSelect;

    // The newer selection has already dispatched by the time the older,
    // slower request finally resolves.
    slow.resolve([message({ id: "stale", conversation_id: "c1" })]);
    await firstSelect;

    const selectedDispatches = dispatch.mock.calls
      .map(([action]) => action)
      .filter((action) => action.type === "selected");
    // Only the most-recently-initiated select's result was ever applied —
    // the stale "c1" response is discarded rather than clobbering "c2".
    expect(selectedDispatches).toHaveLength(1);
    expect(selectedDispatches[0]).toMatchObject({ id: "c2", messages: fast });
  });
});

describe("buildValue: retryLast()", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("does nothing when there is no stashed failedText", async () => {
    const dispatch = vi.fn();
    const refreshSeqRef = { current: 0 };
    const selectSeqRef = { current: 0 };
    const { actions } = buildValue(
      { ...initialState, failedText: null },
      dispatch,
      refreshSeqRef,
      selectSeqRef,
    );

    await actions.retryLast();
    expect(dispatch).not.toHaveBeenCalled();
  });

  it("re-sends the stashed text through the normal send flow", async () => {
    vi.mocked(api.sendChat).mockResolvedValue({
      message: message({ id: "m2", role: "assistant", content: "reply" }),
      latency_ms: 5,
      model: null,
      usage: null,
    });
    vi.mocked(api.listConversations).mockResolvedValue([]);

    const dispatch = vi.fn();
    const refreshSeqRef = { current: 0 };
    const selectSeqRef = { current: 0 };
    const state = {
      ...initialState,
      selectedId: "c1",
      failedText: "what is a hash map?",
      error: "network error",
    };
    const { actions } = buildValue(state, dispatch, refreshSeqRef, selectSeqRef);

    await actions.retryLast();

    expect(api.sendChat).toHaveBeenCalledWith(
      expect.objectContaining({ conversation_id: "c1", message: "what is a hash map?" }),
    );
    // The re-send goes through the normal send flow, which starts by
    // dispatching send_started (clearing failedText/error as a side effect).
    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({ type: "send_started", userText: "what is a hash map?" }),
    );
  });
});

describe("buildValue: send()", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("resolves once the reply lands, without waiting on the sidebar refresh", async () => {
    const reply = message({ id: "m2", role: "assistant", content: "reply" });
    vi.mocked(api.sendChat).mockResolvedValue({
      message: reply,
      latency_ms: 5,
      model: null,
      usage: null,
    });
    const slowRefresh = deferred<never[]>();
    vi.mocked(api.listConversations).mockImplementation(() => slowRefresh.promise);

    const dispatch = vi.fn();
    const { actions } = buildValue(
      { ...initialState, selectedId: "c1" },
      dispatch,
      { current: 0 },
      { current: 0 },
    );

    await actions.send("hello");

    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({ type: "send_succeeded", message: reply }),
    );
    expect(api.listConversations).toHaveBeenCalledTimes(1);
    expect(dispatch).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: "conversations_loaded" }),
    );

    slowRefresh.resolve([]);
    await vi.waitFor(() =>
      expect(dispatch).toHaveBeenCalledWith(
        expect.objectContaining({ type: "conversations_loaded", conversations: [] }),
      ),
    );
  });
});
