#!/usr/bin/env python3
"""Re-run ToolCallJudge from saved BIRD or MCPTox shadow evidence.

The target model and shadow executor are not called. Reusing the same Qwen
tool call and execution evidence makes judge-model comparisons paired.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
# When this file is launched as ``python .\agents\toolcall_rejudge.py``,
# Python adds only the ``agents`` directory to sys.path.  Add the repository
# root explicitly so the sibling ``simguard`` package is importable.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def _case_key(case: dict[str, Any], dataset: str) -> str:
    if dataset == "mcptox":
        return f"{case.get('server_name', '')}::{case.get('id', '')}"
    return str(case.get("id", ""))


def _case_label(case: dict[str, Any], dataset: str) -> str:
    value = case.get("label")
    if value:
        return str(value)
    # The saved MCPTox Qwen attack run predates the explicit label field; all
    # 1,348 rows in that artifact come from malicious_instance records.
    return "attack" if dataset == "mcptox" else "unknown"


def _load(paths: list[Path], dataset: str) -> tuple[list[dict[str, Any]], int]:
    """Merge saved results by case key; later retry files take precedence."""
    merged: dict[str, dict[str, Any]] = {}
    replacements = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases = payload.get("cases")
        if not isinstance(cases, list):
            raise RuntimeError(f"Saved {dataset} result has no cases list: {path}")
        for case in cases:
            if not isinstance(case, dict):
                continue
            key = _case_key(case, dataset)
            if not key:
                continue
            replacements += int(key in merged)
            merged[key] = case
    return list(merged.values()), replacements


def _state_diff_without_labels(state_diff: Any) -> Any:
    """저장된 이전 결과의 state_diff에서 데이터셋 공격 정보를 제거한다."""
    if not isinstance(state_diff, str):
        return state_diff
    marker = state_diff.find('"poisoned_tool_description_present"')
    if marker != -1:
        return state_diff[:marker].rstrip(", ") + "}"
    if state_diff.startswith('{"shadow_metadata"'):
        try:
            return json.loads(state_diff)["state_diff"]
        except (json.JSONDecodeError, KeyError):
            return ""
    return state_diff


def _judge_input(case: dict[str, Any], dataset: str) -> dict[str, str]:
    if dataset == "bird":
        evidence = {
            "generated_sql": case.get("query") or case.get("attack_sql"),
            "shadow_execution_result": case.get("result"),
            "shadow_error": case.get("shadow_error"),
            "state_diff": _state_diff_without_labels(case.get("state_diff")),
            "db_id": case.get("db_id"),
        }
        return {
            "user_request": str(case.get("user_prompt") or ""),
            "tool_name": "execute_sql",
            "tool_category": "database",
            "tool_arguments": json.dumps(
                {"query": case.get("query") or case.get("attack_sql") or ""},
                ensure_ascii=False,
            ),
            "simulated_response": json.dumps(evidence, ensure_ascii=False, default=str),
            "agent_response": "",
        }
    evidence = {
        "shadow_result": case.get("shadow_result"),
        "state_diff": _state_diff_without_labels(case.get("state_diff")),
        "parse_error": case.get("parse_error"),
    }
    return {
        "user_request": str(case.get("query") or ""),
        "tool_name": str(case.get("tool_name") or "no_tool_call"),
        "tool_category": "mcp",
        "tool_arguments": json.dumps(case.get("tool_arguments") or {}, ensure_ascii=False),
        "simulated_response": json.dumps(evidence, ensure_ascii=False, default=str),
        "agent_response": str(case.get("raw_response") or ""),
    }


def _load_previous(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8")).get("results", [])
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(row.get("key")): row for row in rows if not row.get("error")}


def _save(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["results"]
    completed = [row for row in rows if not row.get("error")]
    payload.update({
        "total": len(rows),
        "completed": len(completed),
        "errors": len(rows) - len(completed),
        "blocked": sum(row.get("decision") == "BLOCK" for row in completed),
    })
    payload["block_rate"] = (
        payload["blocked"] / payload["completed"] if payload["completed"] else None
    )
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
    if args.openai_base_url:
        base_url = args.openai_base_url.rstrip("/")
        os.environ["OPENAI_COMPATIBLE_BASE_URL"] = base_url
        os.environ["OPENAI_BASE_URL"] = base_url
    if args.openai_api_key:
        os.environ["OPENAI_COMPATIBLE_API_KEY"] = args.openai_api_key
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    # Import only after the selected model is in the environment.
    from simguard.config import RISK_THRESHOLD_BLOCK
    from simguard.simulation.judges.tool_call_judge import ToolCallJudge

    cases, replacements = _load(args.input, args.dataset)
    if args.label != "all":
        cases = [case for case in cases if _case_label(case, args.dataset) == args.label]
    if args.exclude_category:
        excluded = set(args.exclude_category)
        cases = [case for case in cases if case.get("category") not in excluded]
    if args.limit is not None:
        cases = cases[:args.limit]
    previous = _load_previous(args.output)
    results = list(previous.values())
    pending = [case for case in cases if _case_key(case, args.dataset) not in previous]
    payload: dict[str, Any] = {
        "dataset": args.dataset,
        "label": args.label,
        "sources": [str(path) for path in args.input],
        "source_merge": "later files overwrite the same case key",
        "source_replacements": replacements,
        "judge_model": args.judge_model,
        "reasoning_effort": args.reasoning_effort,
        "threshold": RISK_THRESHOLD_BLOCK,
        "judge_only": True,
        "target_rerun": False,
        "shadow_rerun": False,
        "results": results,
    }
    print(
        f"ToolCallJudge-only dataset={args.dataset} model={args.judge_model} "
        f"total={len(cases)} merged_replacements={replacements} "
        f"reused={len(previous)} pending={len(pending)}",
        flush=True,
    )
    judge = ToolCallJudge()
    semaphore = asyncio.Semaphore(max(1, args.workers))
    lock = asyncio.Lock()
    counter = 0

    async def evaluate(case: dict[str, Any]) -> dict[str, Any]:
        nonlocal counter
        started = time.monotonic()
        key = _case_key(case, args.dataset)
        row: dict[str, Any] = {
            "key": key,
            "id": case.get("id"),
            "label": _case_label(case, args.dataset),
            "category": case.get("category") or case.get("security_risk"),
        }
        async with semaphore:
            last_error: Exception | None = None
            for attempt in range(max(1, args.max_attempts)):
                try:
                    events, risk = await judge.judge(**_judge_input(case, args.dataset))
                    if any(event.type == "judge_parse_failure" for event in events):
                        raise ValueError(events[0].evidence)
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
                    "decision": "ERROR", "risk_score": 1.0,
                    "events": [],
                    "error": f"{type(last_error).__name__}: {last_error}",
                })
        row["elapsed_seconds"] = round(time.monotonic() - started, 3)
        async with lock:
            counter += 1
            print(f"[{counter}/{len(pending)}] {key} {row['decision']}", flush=True)
        return row

    tasks = [asyncio.create_task(evaluate(case)) for case in pending]
    for future in asyncio.as_completed(tasks):
        results.append(await future)
        payload["results"] = results
        _save(args.output, payload)
    _save(args.output, payload)
    print(
        f"Written: {args.output}\ncompleted={payload['completed']}/{payload['total']} "
        f"blocked={payload['blocked']} errors={payload['errors']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rejudge saved BIRD/MCPTox tool calls")
    parser.add_argument("--dataset", required=True, choices=["bird", "mcptox"])
    parser.add_argument(
        "--input", type=Path, action="append", required=True,
        help="Saved result JSON. Repeat in chronological order; later files overwrite duplicate case IDs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--openai-base-url", default=os.getenv("CRTS_API_BASE"))
    parser.add_argument("--openai-api-key")
    parser.add_argument("--label", choices=["attack", "benign", "all"], default="attack")
    parser.add_argument("--exclude-category", action="append", default=[])
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
