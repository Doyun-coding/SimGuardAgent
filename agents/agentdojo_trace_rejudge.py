#!/usr/bin/env python3
"""Re-run only ToolCallJudge on saved AgentDojo SimGuard traces.

The Qwen target agent, tools, shadow execution, state diff, and simulated
follow-up are reused verbatim. No target model or tool is executed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)


def _is_selected_case(row: dict[str, Any], label: str) -> bool:
    if label == "attack":
        return row.get("attack_type") is not None and row.get("injection_task_id") is not None
    return (
        row.get("attack_type") is None
        and row.get("injection_task_id") is None
        and str(row.get("user_task_id") or "").startswith("user_task_")
    )


def _case_key(row: dict[str, Any]) -> str:
    return "::".join([
        str(row.get("suite_name") or "unknown"),
        str(row.get("user_task_id") or "none"),
        str(row.get("attack_type") or "none"),
        str(row.get("injection_task_id") or "none"),
    ])


def _tool_category(name: str) -> str:
    value = name.lower()
    if any(key in value for key in ("email", "mail", "inbox", "send_")):
        return "email"
    if any(key in value for key in ("db", "database", "sql", "query", "table")):
        return "database"
    if any(key in value for key in ("browser", "fetch", "navigate", "page", "web")):
        return "browser"
    if any(key in value for key in ("calendar", "event", "schedule")):
        return "calendar"
    if any(key in value for key in ("memory", "remember", "store", "recall")):
        return "memory"
    if any(key in value for key in ("rag", "retrieve", "search", "vector")):
        return "rag"
    if "mcp" in value:
        return "mcp"
    return "api"


def _load_cases(trace_dirs: list[Path], label: str) -> list[dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for trace_dir in trace_dirs:
        if not trace_dir.is_dir():
            raise RuntimeError(f"Trace directory not found: {trace_dir}")
        for path in trace_dir.rglob("*.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Could not read trace '{path}': {exc}") from exc
            if not isinstance(row, dict) or not _is_selected_case(row, label):
                continue
            row["_source_path"] = str(path)
            key = _case_key(row)
            if key in cases:
                raise RuntimeError(f"Duplicate AgentDojo case key: {key}")
            cases[key] = row
    return [cases[key] for key in sorted(cases)]


def _tool_jobs(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for case in cases:
        case_key = _case_key(case)
        tool_index = 0
        for message_index, message in enumerate(case.get("messages") or []):
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            trace = message.get("simguard_trace")
            if not isinstance(trace, dict):
                continue
            action = trace.get("action") if isinstance(trace.get("action"), dict) else {}
            tool_name = str(trace.get("tool_name") or action.get("tool_name") or "unknown_tool")
            jobs.append({
                "key": f"{case_key}::tool_{tool_index}",
                "case_key": case_key,
                "suite": case.get("suite_name"),
                "label": "attack" if case.get("injection_task_id") is not None else "benign",
                "user_task_id": case.get("user_task_id"),
                "injection_task_id": case.get("injection_task_id"),
                "message_index": message_index,
                "tool_index": tool_index,
                "tool_name": tool_name,
                "tool_category": str(action.get("tool_category") or _tool_category(tool_name)),
                "user_request": str(trace.get("user_request") or action.get("user_request") or ""),
                "tool_arguments": json.dumps(
                    trace.get("tool_args") or action.get("tool_arguments") or {},
                    ensure_ascii=False,
                ),
                # This is the exact concrete evidence passed to the original
                # judge. It includes tool output/error and state diff.
                "simulated_response": str(
                    action.get("retrieved_context")
                    or json.dumps({
                        "tool_result": trace.get("shadow_tool_response"),
                        "tool_error": trace.get("shadow_error"),
                        "state_diff": trace.get("state_diff"),
                    }, ensure_ascii=False, default=str)
                ),
                "agent_response": str(trace.get("simulated_output") or ""),
                "original_decision": trace.get("decision"),
                "original_risk_score": trace.get("risk_score"),
            })
            tool_index += 1
    return jobs


def _load_previous(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8")).get("results", [])
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(row.get("key")): row
        for row in rows
        if row.get("key") and not row.get("error")
    }


def _summarize(
    cases: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected: dict[str, list[str]] = defaultdict(list)
    for job in jobs:
        expected[job["case_key"]].append(job["key"])
    results = {str(row.get("key")): row for row in result_rows}
    case_rows: list[dict[str, Any]] = []
    for case in cases:
        key = _case_key(case)
        tool_keys = expected.get(key, [])
        available = [results[k] for k in tool_keys if k in results]
        if case.get("error"):
            decision = "SOURCE_ERROR"
        elif len(available) < len(tool_keys):
            decision = "PENDING"
        elif any(row.get("error") for row in available):
            decision = "ERROR"
        elif any(row.get("decision") == "BLOCK" for row in available):
            decision = "BLOCK"
        else:
            decision = "ALLOW"
        case_rows.append({
            "key": key,
            "suite": case.get("suite_name"),
            "user_task_id": case.get("user_task_id"),
            "injection_task_id": case.get("injection_task_id"),
            "tool_calls": len(tool_keys),
            "decision": decision,
            "original_utility": case.get("utility"),
            "original_security": case.get("security"),
            "source_error": case.get("error"),
        })

    finalized = [row for row in case_rows if row["decision"] in {"ALLOW", "BLOCK"}]
    by_suite = []
    for suite in sorted({str(row["suite"]) for row in case_rows}):
        suite_rows = [row for row in case_rows if str(row["suite"]) == suite]
        suite_done = [row for row in suite_rows if row["decision"] in {"ALLOW", "BLOCK"}]
        blocked = sum(row["decision"] == "BLOCK" for row in suite_done)
        by_suite.append({
            "suite": suite,
            "total": len(suite_rows),
            "completed": len(suite_done),
            "errors": sum(row["decision"] in {"ERROR", "SOURCE_ERROR"} for row in suite_rows),
            "pending": sum(row["decision"] == "PENDING" for row in suite_rows),
            "block": blocked,
            "allow": len(suite_done) - blocked,
            "block_rate": blocked / len(suite_done) if suite_done else None,
        })
    blocked = sum(row["decision"] == "BLOCK" for row in finalized)
    summary = {
        "total_cases": len(case_rows),
        "completed_cases": len(finalized),
        "error_cases": sum(row["decision"] in {"ERROR", "SOURCE_ERROR"} for row in case_rows),
        "pending_cases": sum(row["decision"] == "PENDING" for row in case_rows),
        "block": blocked,
        "allow": len(finalized) - blocked,
        "block_rate": blocked / len(finalized) if finalized else None,
        "total_saved_tool_calls": len(jobs),
        "completed_tool_judgments": sum(not row.get("error") for row in result_rows),
        "tool_judge_errors": sum(bool(row.get("error")) for row in result_rows),
        "by_suite": by_suite,
    }
    return summary, case_rows


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            time.sleep(min(0.05 * (attempt + 1), 0.5))
    path.write_text(temporary.read_text(encoding="utf-8"), encoding="utf-8")
    temporary.unlink(missing_ok=True)


async def _run(args: argparse.Namespace) -> None:
    os.environ["JUDGE_MODEL"] = args.judge_model
    os.environ["REASONING_EFFORT"] = args.reasoning_effort
    os.environ["JUDGE_TIMEOUT"] = str(args.timeout)
    os.environ["TOOL_JUDGE_MAX_TOKENS"] = str(args.max_tokens)
    if args.openai_base_url:
        os.environ["OPENAI_COMPATIBLE_BASE_URL"] = args.openai_base_url.rstrip("/")
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url.rstrip("/")
    if args.openai_api_key:
        os.environ["OPENAI_COMPATIBLE_API_KEY"] = args.openai_api_key
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    from simguard.config import RISK_THRESHOLD_BLOCK
    from simguard.simulation.judges.tool_call_judge import ToolCallJudge

    cases = _load_cases(args.trace_dir, args.label)
    if args.limit is not None:
        cases = cases[:args.limit]
    jobs = _tool_jobs(cases)
    previous = _load_previous(args.output) if not args.overwrite else {}
    pending = [job for job in jobs if job["key"] not in previous]
    results = list(previous.values())
    print(
        f"AgentDojo trace rejudge: label={args.label} model={args.judge_model} "
        f"cases={len(cases)} tool_calls={len(jobs)} reused={len(previous)} pending={len(pending)}",
        flush=True,
    )
    if args.dry_run:
        summary, _ = _summarize(cases, jobs, results)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    judge = ToolCallJudge()
    semaphore = asyncio.Semaphore(max(1, args.workers))
    lock = asyncio.Lock()
    counter = 0

    async def evaluate(job: dict[str, Any]) -> dict[str, Any]:
        nonlocal counter
        started = time.monotonic()
        row = {key: job[key] for key in (
            "key", "case_key", "suite", "label", "user_task_id",
            "injection_task_id", "message_index", "tool_index", "tool_name",
            "tool_category", "original_decision", "original_risk_score",
        )}
        async with semaphore:
            last_error: Exception | None = None
            for attempt in range(max(1, args.max_attempts)):
                try:
                    events, risk = await judge.judge(
                        user_request=job["user_request"],
                        tool_name=job["tool_name"],
                        tool_category=job["tool_category"],
                        tool_arguments=job["tool_arguments"],
                        simulated_response=job["simulated_response"],
                        agent_response=job["agent_response"],
                    )
                    if any(event.type == "judge_parse_failure" for event in events):
                        raise ValueError(next(
                            event.evidence for event in events
                            if event.type == "judge_parse_failure"
                        ))
                    row.update({
                        "decision": "BLOCK" if risk >= RISK_THRESHOLD_BLOCK else "ALLOW",
                        "risk_score": risk,
                        "events": [event.model_dump() for event in events],
                        "error": None,
                    })
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < max(1, args.max_attempts):
                        await asyncio.sleep(min(2 ** attempt, 8))
            else:
                assert last_error is not None
                row.update({
                    "decision": "ERROR",
                    "risk_score": None,
                    "events": [],
                    "error": f"{type(last_error).__name__}: {last_error}",
                })
        row["elapsed_seconds"] = round(time.monotonic() - started, 3)
        async with lock:
            counter += 1
            print(f"[{counter}/{len(pending)}] {job['key']} {row['decision']}", flush=True)
        return row

    async def checkpoint() -> None:
        summary, case_rows = _summarize(cases, jobs, results)
        _save(args.output, {
            "summary": summary,
            "config": {
                "label": args.label,
                "trace_dirs": [str(path) for path in args.trace_dir],
                "judge_model": args.judge_model,
                "reasoning_effort": args.reasoning_effort,
                "threshold": RISK_THRESHOLD_BLOCK,
                "judge_only": True,
                "target_rerun": False,
                "tool_rerun": False,
                "shadow_rerun": False,
                "paired_trace_comparison": True,
            },
            "case_results": case_rows,
            "results": results,
        })

    tasks = [asyncio.create_task(evaluate(job)) for job in pending]
    for completed, future in enumerate(asyncio.as_completed(tasks), start=1):
        results.append(await future)
        if completed % max(1, args.checkpoint_every) == 0:
            await checkpoint()
    await checkpoint()
    summary, _ = _summarize(cases, jobs, results)
    print(f"Written: {args.output}", flush=True)
    print(
        f"cases={summary['completed_cases']}/{summary['total_cases']} "
        f"block={summary['block']} allow={summary['allow']} "
        f"errors={summary['error_cases']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge-only re-evaluation of saved AgentDojo SimGuard traces."
    )
    parser.add_argument("--trace-dir", action="append", type=Path, required=True)
    parser.add_argument("--label", choices=["attack", "benign"], required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--openai-base-url", default=os.getenv("CRTS_API_BASE"))
    parser.add_argument("--openai-api-key")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
