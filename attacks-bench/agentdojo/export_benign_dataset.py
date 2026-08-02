#!/usr/bin/env python3
"""Export clean benign AgentDojo user tasks.

AgentDojo evaluates attacks by pairing a legitimate user task with an
injection task. This exporter extracts only the legitimate user tasks, without
any injection, so the resulting file can be used as a clean benign split for
FPR / clean utility evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTDOJO_SRC = REPO_ROOT / "agents" / "agentdojo" / "src"
if str(AGENTDOJO_SRC) not in sys.path:
    sys.path.insert(0, str(AGENTDOJO_SRC))

from agentdojo.task_suite.load_suites import get_suite, get_suites  # noqa: E402


DEFAULT_SUITES = ("workspace", "banking", "slack", "travel")


def _difficulty_name(task: Any) -> str | None:
    difficulty = getattr(task, "DIFFICULTY", None)
    return getattr(difficulty, "name", None)


def _ground_truth_calls(suite: Any, task: Any) -> list[dict[str, Any]]:
    try:
        env = suite.load_and_inject_default_environment({})
        env = task.init_environment(env)
        calls = task.ground_truth(env)
    except Exception as exc:  # pragma: no cover - exporter should keep going.
        return [{"error": f"{type(exc).__name__}: {exc}"}]

    serialized: list[dict[str, Any]] = []
    for call in calls:
        if hasattr(call, "model_dump"):
            serialized.append(call.model_dump(mode="json"))
        elif isinstance(call, dict):
            serialized.append(call)
        else:
            serialized.append({"repr": repr(call)})
    return serialized


def iter_examples(
    benchmark_version: str,
    suites: tuple[str, ...],
    include_ground_truth: bool,
    expand_by_injection: bool,
):
    for suite_name in suites:
        suite = get_suite(benchmark_version, suite_name)
        for task_id, task in suite.user_tasks.items():
            injection_ids = tuple(suite.injection_tasks) if expand_by_injection else (None,)
            for injection_id in injection_ids:
                suffix = f"_{injection_id}" if injection_id is not None else ""
                example = {
                    "id": f"agentdojo_{suite_name}_{task_id}{suffix}",
                    "benchmark": "agentdojo",
                    "benchmark_version": benchmark_version,
                    "suite": suite_name,
                    "task_id": task_id,
                    "paired_injection_task_id": injection_id,
                    "label": "benign",
                    "category": suite_name,
                    "prompt": task.PROMPT,
                    "ground_truth_output": getattr(task, "GROUND_TRUTH_OUTPUT", ""),
                    "difficulty": _difficulty_name(task),
                }
                if include_ground_truth:
                    example["ground_truth_tool_calls"] = _ground_truth_calls(suite, task)
                yield example


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, rows: list[dict[str, Any]], benchmark_version: str, suites: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "agentdojo",
        "benchmark_version": benchmark_version,
        "label": "benign",
        "suites": list(suites),
        "total": len(rows),
        "examples": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export clean benign AgentDojo user tasks")
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument(
        "--suite",
        action="append",
        choices=[*DEFAULT_SUITES, "all"],
        help="Suite to export. Can be repeated. Default: all four AgentDojo suites.",
    )
    parser.add_argument(
        "--out",
        default="./attacks-bench/agentdojo/data/agentdojo_clean_benign.jsonl",
        help="Output path. Use .json for wrapped JSON, anything else for JSONL.",
    )
    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Do not include oracle ground-truth tool calls.",
    )
    parser.add_argument(
        "--expand-by-injection",
        action="store_true",
        help="Repeat each clean user task once per injection task to match AgentDojo attack-case counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = tuple(args.suite or ("all",))
    if "all" in requested:
        suites = tuple(get_suites(args.benchmark_version).keys())
    else:
        suites = requested

    rows = list(
        iter_examples(
            args.benchmark_version,
            suites,
            include_ground_truth=not args.no_ground_truth,
            expand_by_injection=args.expand_by_injection,
        )
    )
    out = Path(args.out)
    if out.suffix == ".json":
        write_json(out, rows, args.benchmark_version, suites)
    else:
        write_jsonl(out, rows)

    counts = {suite: 0 for suite in suites}
    for row in rows:
        counts[row["suite"]] += 1
    print(f"Wrote {len(rows)} clean benign AgentDojo tasks to {out}")
    for suite, count in counts.items():
        print(f"  {suite}: {count}")


if __name__ == "__main__":
    main()
