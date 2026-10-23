// Chat-path variant of bench/k6.js: exercises POST /v1/chat (store reads,
// history flattening, SQLite writes) instead of /v1/generate.
// setup() creates CONVERSATIONS conversations and seeds SEED_TURNS turns in
// each; the load phase then keeps appending turns, so history keeps growing.
import http from "k6/http";
import { check } from "k6";
import { SharedArray } from "k6/data";

const BASE = __ENV.BASE_URL || "http://localhost:8000";
const CONVERSATIONS = parseInt(__ENV.CONVERSATIONS || "20", 10);
const SEED_TURNS = parseInt(__ENV.SEED_TURNS || "10", 10);
const JSON_HEADERS = { headers: { "Content-Type": "application/json" } };

const workload = new SharedArray("workload", () =>
  JSON.parse(open("./workload.json"))
);

export const options = {
  setupTimeout: "180s",
  scenarios: {
    steady: {
      executor: "constant-arrival-rate",
      rate: 30,
      timeUnit: "1s",
      duration: "60s",
      preAllocatedVUs: 250,
    },
  },
  thresholds: {
    // only the load phase counts; setup/seeding requests are tagged out
    "http_req_failed{phase:load}": ["rate<0.01"],
    "http_req_duration{phase:load}": [],
    checks: ["rate>0.99"],
  },
  summaryTrendStats: ["avg", "p(50)", "p(95)", "p(99)", "max"],
};

function chatBody(cid, i) {
  const item = workload[i % workload.length];
  return JSON.stringify({ conversation_id: cid, message: item.prompt, max_tokens: 64 });
}

export function setup() {
  const seedParams = { ...JSON_HEADERS, tags: { phase: "setup" } };
  const ids = [];
  for (let c = 0; c < CONVERSATIONS; c++) {
    const res = http.post(`${BASE}/v1/conversations`, JSON.stringify({}), seedParams);
    if (res.status !== 201) throw new Error(`create conversation failed: ${res.status}`);
    ids.push(res.json().id);
  }
  for (let t = 0; t < SEED_TURNS; t++) {
    const batch = ids.map((cid, c) => ["POST", `${BASE}/v1/chat`, chatBody(cid, t + c), seedParams]);
    const responses = http.batch(batch);
    responses.forEach((r) => {
      if (r.status !== 200) throw new Error(`seed turn failed: ${r.status}`);
    });
  }
  return { ids };
}

export default function (data) {
  const cid = data.ids[Math.floor(Math.random() * data.ids.length)];
  const res = http.post(`${BASE}/v1/chat`, chatBody(cid, __ITER + __VU), {
    ...JSON_HEADERS,
    tags: { phase: "load" },
  });
  let body = {};
  try {
    body = res.json();
  } catch (e) {
    body = {};
  }
  check(res, {
    "status 200": (r) => r.status === 200,
    "assistant message non-empty": () =>
      !!body.message && typeof body.message.content === "string" && body.message.content.length > 0,
    "latency_ms reported": () => typeof body.latency_ms === "number",
  });
}

export function handleSummary(data) {
  const t = data.metrics["http_req_duration{phase:load}"].values;
  const err = data.metrics["http_req_failed{phase:load}"].values.rate;
  const chk = data.metrics.checks ? data.metrics.checks.values.rate : 1;
  const fmt = (label, value) => `  ${label.padEnd(12)} ${value}\n`;
  let out = `\n=== chat bench summary (60s @ 30 rps, ${CONVERSATIONS} convs, ${SEED_TURNS} seed turns) ===\n`;
  out += fmt("p50", `${t["p(50)"].toFixed(0)} ms`);
  out += fmt("p95", `${t["p(95)"].toFixed(0)} ms`);
  out += fmt("p99", `${t["p(99)"].toFixed(0)} ms`);
  out += fmt("error rate", `${(err * 100).toFixed(2)} %`);
  out += fmt("checks", `${(chk * 100).toFixed(2)} %`);
  return { stdout: out };
}
