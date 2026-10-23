"""Helpers for scripts/bench_runs.sh and per-task results files.

collect <harness> <script> <workdir> <runs> <out.json>
    Parse each run's handleSummary stdout (+ k6 --summary-export checks) and
    write {"harness", "script", "runs": [...], "median": {...}}.
compare <baseline.json> <after.json> <out.json> --task N --branch B --sha S
        [--baseline-sha S] [--pytest pass|fail] [--e2e pass|fail] [--notes TEXT]
    Combine two collect outputs into a bench/results/task-<N>.json file.
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

_LINE = re.compile(r"^\s*(p50|p95|p99|error rate|checks)\s+([0-9.]+)\s*(ms|%)", re.M)
_KEYS = {"p50": "p50_ms", "p95": "p95_ms", "p99": "p99_ms",
         "error rate": "error_rate_pct", "checks": "checks_pct"}


def _parse_run(workdir: Path, i: int) -> dict:
    stdout = (workdir / f"stdout{i}.txt").read_text()
    run = {_KEYS[m.group(1)]: float(m.group(2)) for m in _LINE.finditer(stdout)}
    run["raw_stdout"] = stdout
    summary = workdir / f"summary{i}.json"
    if summary.exists():
        data = json.loads(summary.read_text())
        checks = {}

        def walk(group):
            for name, c in (group.get("checks") or {}).items():
                checks[name] = {"passes": c.get("passes"), "fails": c.get("fails")}
            for sub in (group.get("groups") or {}).values():
                walk(sub)

        walk(data.get("root_group", {}))
        run["checks_detail"] = checks
    return run


def collect(args) -> None:
    workdir = Path(args.workdir)
    runs = [_parse_run(workdir, i) for i in range(1, int(args.runs) + 1)]
    complete = [r for r in runs if all(k in r for k in _KEYS.values())]
    if not complete:
        sys.exit("no parsable k6 runs")
    median = {k: statistics.median(r[k] for r in complete) for k in _KEYS.values()}
    out = {"harness": args.harness, "script": args.script, "runs": runs, "median": median}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")


def compare(args) -> None:
    base = json.loads(Path(args.baseline).read_text())
    after = json.loads(Path(args.after).read_text())
    if base["harness"] != after["harness"]:
        sys.exit("baseline and after used different harnesses")
    delta = {}
    for k in ("p50_ms", "p95_ms", "p99_ms"):
        b, a = base["median"][k], after["median"][k]
        delta[k] = {"baseline": b, "after": a, "abs_ms": round(b - a, 1),
                    "pct_improvement": round((b - a) / b * 100, 1) if b else None}
    for k in ("error_rate_pct", "checks_pct"):
        delta[k] = {"baseline": base["median"][k], "after": after["median"][k]}
    out = {
        "task": args.task,
        "branch": args.branch,
        "commit_sha": args.sha,
        "baseline_sha": args.baseline_sha,
        "harness": base["harness"],
        "script": base["script"],
        "runs_per_side": len(base["runs"]),
        "summary": delta,
        "pytest": args.pytest,
        "e2e": args.e2e,
        "notes": args.notes,
        "baseline": base,
        "after": after,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    for name in ("harness", "script", "workdir", "runs", "out"):
        c.add_argument(name)
    c.set_defaults(fn=collect)
    m = sub.add_parser("compare")
    m.add_argument("baseline")
    m.add_argument("after")
    m.add_argument("out")
    m.add_argument("--task", type=int, required=True)
    m.add_argument("--branch", required=True)
    m.add_argument("--sha", required=True)
    m.add_argument("--baseline-sha")
    m.add_argument("--pytest", choices=["pass", "fail"])
    m.add_argument("--e2e", choices=["pass", "fail"])
    m.add_argument("--notes", default="")
    m.set_defaults(fn=compare)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
