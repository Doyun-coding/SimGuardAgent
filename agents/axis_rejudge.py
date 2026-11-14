#!/usr/bin/env python3
"""Offline BehaviorJudge model comparison for saved Jailbreak, RAG, and Memory runs.

The target agent is never called.  Prompt inputs are reconstructed from the
benchmark datasets, while saved responses / simulated outputs are reused.  For
Inspect ``.eval`` memory logs, the original document and proposed saved-memory
tool arguments are read directly from the ZIP archive.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def _expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_dir():
            matches = sorted(candidate.glob("*.eval"))
        else:
            matches = [Path(item) for item in sorted(glob.glob(pattern))]
            if not matches and candidate.is_file():
                matches = [candidate]
        paths.extend(matches)
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique:
        raise SystemExit("No input files matched.")
    return unique


def _save(path: Path, payload: dict[str, Any]) -> None:
    results = payload["results"]
    completed = sum(not item.get("error") for item in results)
    blocked = sum(item.get("decision") == "BLOCK" for item in results if not item.get("error"))
    payload["total"] = len(results)
    payload["completed"] = completed
    payload["blocked"] = blocked
    payload["errors"] = len(results) - completed
    payload["block_rate"] = round(blocked / completed * 100, 2) if completed else 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Windows Search, antivirus, and editor previews can briefly open the JSON
    # without delete-sharing, which makes an otherwise valid os.replace fail
    # with WinError 5. Retry the atomic promotion before falling back to a
    # direct update; the complete .tmp remains recoverable if both fail.
    last_error: PermissionError | None = None
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(0.05 * (attempt + 1), 0.5))
    try:
        path.write_text(temporary.read_text(encoding="utf-8"), encoding="utf-8")
        temporary.unlink(missing_ok=True)
    except PermissionError:
        assert last_error is not None
        raise RuntimeError(
            f"Result file remained locked after save retries: {path}. "
            f"The newer recovery copy is preserved at {temporary}."
        ) from last_error


def _load_completed_for_resume(output: Path) -> dict[str, dict[str, Any]]:
    """Recover completed rows from the newest valid JSON or recovery file."""
    candidates: list[dict[str, Any]] = []
    for candidate in (output, output.with_suffix(output.suffix + ".tmp")):
        if not candidate.exists():
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and isinstance(value.get("results"), list):
            candidates.append(value)
    if not candidates:
        return {}
    newest = max(candidates, key=lambda value: len(value["results"]))
    return {
        str(item.get("key") or item.get("id")): item
        for item in newest["results"]
        if not item.get("error")
    }


def _load_prompt_items(path: Path) -> tuple[str, list[dict[str, Any]]]:
    from run_benchmark import _load_ga_jailbreak, _load_saferag

    source = json.loads(path.read_text(encoding="utf-8"))
    records = source.get("results", [])
    if not records:
        return "prompt", []
    label = str(records[0].get("label") or ("benign" if "benign" in path.name else "attack"))
    benchmark = str(source.get("benchmark") or "").lower()
    if benchmark == "ga_jailbreak" or "jailbreak" in path.name.lower():
        entries = _load_ga_jailbreak(None, label=label)
        kind = "jailbreak"
    elif benchmark == "saferag" or "saferag" in path.name.lower():
        entries = _load_saferag(None, label=label, mix="poisoned")
        kind = "rag"
    else:
        raise RuntimeError(f"Cannot infer prompt benchmark for {path}")
    entry_by_id = {str(entry["id"]): entry for entry in entries}
    items: list[dict[str, Any]] = []
    for record in records:
        item_id = str(record.get("id"))
        entry = entry_by_id.get(item_id)
        if entry is None:
            items.append({"key": item_id, "id": item_id, "load_error": "benchmark entry not found"})
            continue
        items.append({
            "key": item_id,
            "id": item_id,
            "kind": kind,
            "label": label,
            "category": record.get("category", entry.get("category", "")),
            "user_query": entry["task"],
            "contexts": "\n\n".join(entry.get("contexts", [])),
            "ground_truth": "\n\n".join(entry.get("clean_contexts", [])),
            "final_answer": (
                record.get("final_simulated_output")
                or record.get("response")
                or record.get("raw_response")
                or ""
            ),
            "shadow_answer": record.get("shadow_simulated_output") or "",
        })
    return kind, items


def _memory_candidates(sample: dict[str, Any]) -> list[str]:
    scorer = (sample.get("scores") or {}).get("tool_call_scorer") or {}
    metadata = scorer.get("metadata") or {}
    values = metadata.get("saved_memories") or []
    if isinstance(values, str):
        values = [values]
    if not values and scorer.get("answer"):
        values = [scorer["answer"]]
    return [str(value).strip() for value in values if str(value).strip()]


def _load_memory_items(paths: list[Path]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            names = sorted(name for name in archive.namelist() if name.startswith("samples/") and name.endswith(".json"))
            for name in names:
                sample = json.loads(archive.read(name))
                sample_id = str(sample.get("id") or Path(name).stem)
                items.append({
                    "key": f"{path.name}::{sample_id}",
                    "id": sample_id,
                    "kind": "memory",
                    "label": "benign" if "benign" in str(path).lower() else "attack",
                    "source_eval": str(path),
                    "user_query": "The assistant is about to write this into the user's persistent memory.",
                    "contexts": str((sample.get("metadata") or {}).get("doc_text") or "")[:12000],
                    "memory_candidates": _memory_candidates(sample),
                })
    return items


async def _run(args: argparse.Namespace) -> None:
    inputs = _expand_inputs(args.input)
    output = Path(args.output).resolve()
    os.environ["JUDGE_MODEL"] = args.judge_model
    os.environ["REASONING_EFFORT"] = args.reasoning_effort
    os.environ["JUDGE_MAX_ATTEMPTS"] = str(args.max_attempts)

    # Import after configuring the environment: simguard.config resolves the
    # judge model at module-import time.
    from simguard.config import RISK_THRESHOLD_BLOCK
    from simguard.simulation.judges.behavior_judge import BehaviorJudge

    if all(path.suffix.lower() == ".eval" for path in inputs):
        kind = "memory"
        items = _load_memory_items(inputs)
    elif len(inputs) == 1 and inputs[0].suffix.lower() == ".json":
        kind, items = _load_prompt_items(inputs[0])
    else:
        raise SystemExit("Use one prompt JSON or one/more Memory .eval files per run.")

    previous = _load_completed_for_resume(output)
    results = list(previous.values())
    pending = [item for item in items if item["key"] not in previous]
    payload: dict[str, Any] = {
        "judge_model": args.judge_model,
        "reasoning_effort": args.reasoning_effort,
        "kind": kind,
        "sources": [str(path) for path in inputs],
        "threshold": RISK_THRESHOLD_BLOCK,
        "results": results,
    }
    print(
        f"BehaviorJudge-only: kind={kind} model={args.judge_model} "
        f"total={len(items)} reused={len(previous)} pending={len(pending)}",
        flush=True,
    )
    judge = BehaviorJudge()
    semaphore = asyncio.Semaphore(max(1, args.workers))
    lock = asyncio.Lock()
    counter = 0

    async def evaluate(item: dict[str, Any]) -> dict[str, Any]:
        nonlocal counter
        started = time.monotonic()
        result: dict[str, Any] = {
            "key": item["key"],
            "id": item["id"],
            "label": item.get("label"),
            "category": item.get("category", ""),
            "source_eval": item.get("source_eval"),
        }
        async with semaphore:
            try:
                if item.get("load_error"):
                    raise RuntimeError(item["load_error"])
                candidates: list[tuple[str, str]]
                if kind == "memory":
                    candidates = [
                        (f"memory_{index + 1}", value)
                        for index, value in enumerate(item.get("memory_candidates", []))
                    ]
                    if not candidates:
                        result.update({
                            "decision": "ALLOW", "risk_score": 0.0,
                            "judge_source": "no_memory_candidate", "axis_detail": {},
                            "events": [], "error": None,
                        })
                        return result
                else:
                    candidates = [("final_answer", item.get("final_answer", ""))]
                    if item.get("shadow_answer"):
                        candidates.append(("shadow_answer", item["shadow_answer"]))

                judgments: list[dict[str, Any]] = []
                for source, answer in candidates:
                    events, risk, detail = await judge.judge(
                        user_query=item["user_query"],
                        contexts=item.get("contexts", ""),
                        answer=answer,
                        ground_truth=item.get("ground_truth") or None,
                    )
                    judgments.append({
                        "source": source,
                        "answer": answer,
                        "risk_score": risk,
                        "decision": "BLOCK" if risk >= RISK_THRESHOLD_BLOCK else "ALLOW",
                        "axis_detail": detail,
                        "events": [event.model_dump() for event in events],
                    })
                selected = max(judgments, key=lambda value: value["risk_score"])
                result.update({
                    "decision": selected["decision"],
                    "risk_score": selected["risk_score"],
                    "judge_source": selected["source"],
                    "axis_detail": selected["axis_detail"],
                    "events": selected["events"],
                    "judgments": judgments,
                    "error": None,
                })
            except Exception as exc:
                result.update({
                    "decision": "ERROR", "risk_score": 1.0,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            finally:
                result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        async with lock:
            counter += 1
            status = result.get("decision", "ERROR")
            print(f"[{counter}/{len(pending)}] {item['id']} {status}", flush=True)
        return result

    tasks = [asyncio.create_task(evaluate(item)) for item in pending]
    for future in asyncio.as_completed(tasks):
        result = await future
        async with lock:
            results.append(result)
            payload["results"] = results
            _save(output, payload)
    _save(output, payload)
    print(
        f"Written: {output}\ncompleted={payload['completed']}/{payload['total']} "
        f"blocked={payload['blocked']} errors={payload['errors']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rejudge saved runs with BehaviorJudge only")
    parser.add_argument("--input", action="append", required=True, help="JSON, .eval, directory, or glob; repeatable")
    parser.add_argument("--output", required=True, help="Model-specific output JSON")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
