"""
Latency benchmark script — satisfies requirement #4: P50/P70/P100 numbers
across a real query set, not a single best-case run, broken down per stage
so the honest bottleneck is visible (see docs/PROGRESS.md notes on the
200ms target).

Usage:
    python -m benchmarks.run_latency_benchmark --queries benchmarks/test_queries.txt

Wire this up to your fully-constructed RagOrchestrator (real embedder, real
Sarvam/Groq clients, real indexed corpus) before running — this script is
deliberately agnostic to how the orchestrator was built so it works whether
you're testing with text-only queries (STT skipped) or real audio.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from app.harness.orchestrator import RagOrchestrator
from app.harness.schemas import QueryRequest


def percentile(values: list[float], pct: float) -> float:
    """pct in [0, 100]. P100 is just max() but implemented consistently here
    so all three percentiles go through one code path."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if pct >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def run_benchmark(orchestrator: RagOrchestrator, queries: list[str]) -> dict:
    total_times: list[float] = []
    stage_times: dict[str, list[float]] = {}
    refusals = 0

    for q in queries:
        response = orchestrator.handle_query(QueryRequest(text_query=q))
        total_times.append(response.total_duration_ms)
        if not response.answered:
            refusals += 1
        for stage_timing in response.stage_timings:
            stage_times.setdefault(stage_timing.stage, []).append(stage_timing.duration_ms)

    report = {
        "num_queries": len(queries),
        "refusals": refusals,
        "total_latency_ms": {
            "p50": round(percentile(total_times, 50), 2),
            "p70": round(percentile(total_times, 70), 2),
            "p100": round(percentile(total_times, 100), 2),
            "mean": round(statistics.mean(total_times), 2) if total_times else 0,
        },
        "per_stage_latency_ms": {},
    }

    for stage, times in stage_times.items():
        report["per_stage_latency_ms"][stage] = {
            "p50": round(percentile(times, 50), 2),
            "p70": round(percentile(times, 70), 2),
            "p100": round(percentile(times, 100), 2),
            "mean": round(statistics.mean(times), 2) if times else 0,
        }

    return report


def main():
    parser = argparse.ArgumentParser(description="Run latency benchmark against the RAG pipeline")
    parser.add_argument("--queries", type=str, required=True,
                         help="Path to a text file, one query per line")
    parser.add_argument("--output", type=str, default="benchmarks/latency_report.json",
                         help="Where to write the JSON report")
    args = parser.parse_args()

    queries = [line.strip() for line in Path(args.queries).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(queries) < 20:
        print(f"WARNING: only {len(queries)} test queries — the task explicitly asks for "
              f"'a reasonable number of test queries, not a single best-case run'. "
              f"Aim for 50+ for a credible P50/P70/P100 spread.")

    # NOTE: this is where YOU wire in your real, fully-constructed orchestrator
    # (real embedder + indexed corpus + Groq client + guardrails). Left as an
    # explicit error rather than a silent mock so this can never accidentally
    # produce fake numbers for submission.
    raise NotImplementedError(
        "Construct your real RagOrchestrator here (see backend/app/harness/orchestrator.py "
        "and the wiring pattern tested in PROGRESS.md step 6) and pass it to run_benchmark(). "
        "This is deliberately not stubbed out for you — benchmark numbers must come from your "
        "actual pipeline, not a placeholder."
    )

    report = run_benchmark(orchestrator, queries)  # noqa: F821
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
