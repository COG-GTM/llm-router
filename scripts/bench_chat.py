"""Micro-benchmark for POST /v1/chat (not part of `make bench`).

Starts the app under uvicorn in a subprocess with a fake upstream (fixed delay,
no network) and a throwaway on-disk SQLite file, fires N concurrent chat turns
spread over a few conversations, and reports latency percentiles and
throughput. This isolates the chat route's own work (store reads/writes) from
upstream noise.

    uv run python scripts/bench_chat.py --requests 2000 --concurrency 100
"""
import argparse
import asyncio
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time

import httpx

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

_SERVER = """
import asyncio, logging
import uvicorn
from app import main

logging.disable(logging.INFO)  # per-request logs would dominate the timings

class FakePool:
    async def forward(self, payload):
        await asyncio.sleep({delay_s})
        return 200, {{"completion": "mock answer " * 8, "model": "mock-large",
                      "usage": {{"prompt_tokens": 5, "completion_tokens": 16}}}}

    async def aclose(self):
        pass

main.UpstreamPool = FakePool
uvicorn.run(main.app, port={port}, log_level="warning", access_log=False)
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pct(sorted_ms: list[float], p: float) -> float:
    return sorted_ms[min(len(sorted_ms) - 1, round(p / 100 * (len(sorted_ms) - 1)))]


async def run(args, base_url: str) -> None:
    limits = httpx.Limits(max_connections=args.concurrency)
    async with httpx.AsyncClient(base_url=base_url, limits=limits, timeout=30) as client:
        for _ in range(100):
            try:
                await client.get("/v1/conversations")
                break
            except httpx.TransportError:
                await asyncio.sleep(0.1)
        cids = [(await client.post("/v1/conversations", json={})).json()["id"]
                for _ in range(args.conversations)]
        sem = asyncio.Semaphore(args.concurrency)
        latencies: list[float] = []
        errors = 0

        async def one(i: int) -> None:
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                resp = await client.post("/v1/chat", json={
                    "conversation_id": cids[i % len(cids)], "message": f"question {i}"})
                latencies.append((time.perf_counter() - t0) * 1000)
                if resp.status_code != 200 or not resp.json()["message"]["id"]:
                    errors += 1

        t_start = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(args.requests)))
        elapsed = time.perf_counter() - t_start

        stored = 0
        for cid in cids:
            stored += len((await client.get(f"/v1/conversations/{cid}/messages")).json())

    lat = sorted(latencies)
    print(f"requests={args.requests} concurrency={args.concurrency} "
          f"conversations={args.conversations} upstream_ms={args.upstream_ms}")
    print(f"p50={pct(lat, 50):.1f}ms p95={pct(lat, 95):.1f}ms p99={pct(lat, 99):.1f}ms "
          f"mean={statistics.mean(lat):.1f}ms max={lat[-1]:.1f}ms")
    print(f"throughput={args.requests / elapsed:.1f} req/s errors={errors} "
          f"persisted_messages={stored}/{2 * args.requests}")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--conversations", type=int, default=20)
    parser.add_argument("--upstream-ms", type=float, default=135)
    parser.add_argument("--db-dir", default=None,
                        help="directory for the throwaway SQLite file (default: a temp dir)")
    args = parser.parse_args()
    port = free_port()
    code = _SERVER.format(delay_s=args.upstream_ms / 1000, port=port)
    with tempfile.TemporaryDirectory(dir=args.db_dir) as tmp:
        env = {**os.environ, "APP_DB_PATH": os.path.join(tmp, "bench.db")}
        server = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, env=env)
        try:
            asyncio.run(run(args, f"http://127.0.0.1:{port}"))
        finally:
            server.terminate()
            server.wait()


if __name__ == "__main__":
    main_cli()
