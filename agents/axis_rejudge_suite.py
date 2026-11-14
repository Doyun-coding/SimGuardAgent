#!/usr/bin/env python3
"""Run the saved-result BehaviorJudge comparison over the six paper datasets."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REJUDGE = ROOT / "agents" / "axis_rejudge.py"

DATASETS = {
    "jailbreak_attack": ROOT / "results/attacks-bench/jailbreak/GA_Jailbreak__simguard_request_result__qwen35_9b__attack.json",
    "jailbreak_benign": ROOT / "results/attacks-bench/jailbreak/GA_Jailbreak__simguard_request_result__qwen35_9b__benign.json",
    "rag_attack": ROOT / "results/attacks-bench/rag/SafeRAG_nctd_poisoned__agentdojo__simguard__qwen35_9b__attack_contextfixed.json",
    "rag_benign": ROOT / "results/attacks-bench/rag/SafeRAG_nctd__agentdojo__simguard__qwen35_9b__benign_contextfixed.json",
    "memory_attack": ROOT / "attacks-bench/memory/LLM-agent-memory-poisoning/.artifacts/agentdojo/simguard_stable_attack/logs",
    "memory_benign": ROOT / "attacks-bench/memory/LLM-agent-memory-poisoning/.artifacts/agentdojo/simguard_stable_benign/logs",
}


def _slug(model: str) -> str:
    value = model.removeprefix("openrouter/")
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge saved Jailbreak/RAG/Memory runs")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument(
        "--dataset", choices=["all", *DATASETS], default="all",
        help="Run all six datasets or one named dataset.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "behavior_judge")
    args = parser.parse_args()

    selected = DATASETS.items() if args.dataset == "all" else [(args.dataset, DATASETS[args.dataset])]
    failed: list[str] = []
    for name, source in selected:
        output = args.output_dir / f"{name}__judge_{_slug(args.judge_model)}.json"
        print(f"\n=== {name} -> {output} ===", flush=True)
        completed = subprocess.run(
            [
                sys.executable, str(REJUDGE),
                "--input", str(source),
                "--output", str(output),
                "--judge-model", args.judge_model,
                "--reasoning-effort", args.reasoning_effort,
                "--workers", str(args.workers),
                "--max-attempts", str(args.max_attempts),
            ],
            cwd=ROOT,
            check=False,
        )
        if completed.returncode:
            failed.append(name)
    if failed:
        raise SystemExit(f"Failed datasets: {', '.join(failed)}")


if __name__ == "__main__":
    main()
