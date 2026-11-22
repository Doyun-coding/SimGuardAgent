"""실제 AutoGen(AgentChat AssistantAgent)에 SimGuardAgent를 연결해 데이터셋을 돌린다.

AutoGen 코드는 고치지 않는다. AutoGen의 OpenAIChatCompletionClient는 OpenAI SDK
chat.completions.create로 모델을 부르므로 simguard.protect(providers=["openai"])가 모델 출력마다
에이전트에게 돌아가기 전에 판정한다. 도구가 없으면 출력은 답변이고 BehaviorJudge가 판정한다.
BLOCK이면 답변이 차단 문구로 바뀐다.

  .venv-autogen/Scripts/python.exe agents/autogen/run_autogen_simguard.py --benchmark GA_Jailbreak --label attack --limit 2
  --mode none 이면 SimGuardAgent 없이 돌린다 (baseline).
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", default="GA_Jailbreak", help="run_benchmark.py의 데이터셋 이름")
    parser.add_argument("--label", default="attack", choices=["attack", "benign", "all"])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--mode", default="simguard", choices=["simguard", "none"])
    parser.add_argument("--model-id", default=os.getenv("AGENTDOJO_MODEL_ID", "openrouter/qwen/qwen3.5-9b"),
                        help="타겟 모델 (API가 받는 이름)")
    parser.add_argument("--simulator-model-id", default=None, help="기본: --model-id")
    parser.add_argument("--judge-model-id", default="openrouter/openai/gpt-oss-120b")
    parser.add_argument("--workers", type=int, default=10, help="동시에 돌릴 항목 수")
    parser.add_argument("--output", type=Path, default=None, help="같은 폴더로 다시 돌리면 끝난 항목은 건너뛴다")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    base = (os.getenv("OPENAI_COMPATIBLE_BASE_URL") or os.getenv("CRTS_API_BASE") or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
    key = os.getenv("OPENAI_COMPATIBLE_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
    os.environ.update(
        SIMULATOR_MODEL=args.simulator_model_id or args.model_id,
        JUDGE_MODEL=args.judge_model_id,
        JUDGE_NO_THINK="1",
        OPENAI_COMPATIBLE_BASE_URL=base,
        OPENAI_COMPATIBLE_API_KEY=key,
    )
    output = args.output or _REPO_ROOT / "runs" / f"autogen_{args.benchmark}_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=True)

    from run_benchmark import _LABELED_BENCHMARKS, BENCHMARKS

    loader, _ = BENCHMARKS[args.benchmark]
    entries = loader(args.limit, args.label) if args.benchmark in _LABELED_BENCHMARKS else loader(args.limit)

    # ── SimGuardAgent 연결: AutoGen을 import하기 전에 ───────────────────────
    # 항목마다 판정을 따로 모은다 (동시에 여러 항목을 돌리므로 contextvar로 나눈다).
    judged_var: contextvars.ContextVar[list | None] = contextvars.ContextVar("judged", default=None)

    def on_result(result) -> None:
        items = judged_var.get()
        if items is not None:
            items.append(result)

    if args.mode == "simguard":
        import simguard

        simguard.protect(providers=["openai"], on_result=on_result, target_model=args.model_id, live_reads=False)

    # ── 여기부터 AutoGen (SimGuardAgent를 모른다) ───────────────────────────
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    def make_client() -> OpenAIChatCompletionClient:
        return OpenAIChatCompletionClient(
            model=args.model_id, base_url=base, api_key=key, temperature=0,
            model_info={"vision": False, "function_calling": True, "json_output": True,
                        "family": "unknown", "structured_output": False},
            # 타겟 모델의 thinking을 끈다 (다른 실험과 같은 조건)
            extra_body={"reasoning": {"effort": "none", "exclude": True}},
        )

    results_path = output / f"results_{args.label}.jsonl"
    done = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not row.get("error"):
                    done[row["id"]] = row
    todo = [entry for entry in entries if entry["id"] not in done]
    print(f"{args.benchmark} {args.label}: total={len(entries)} done={len(done)} todo={len(todo)} workers={args.workers}", flush=True)
    semaphore = asyncio.Semaphore(args.workers)
    write_lock = asyncio.Lock()
    rows = list(done.values())

    async def run_one(entry: dict) -> None:
        async with semaphore:
            judged: list = []
            judged_var.set(judged)
            started = time.time()
            client = make_client()
            error, response = None, None
            try:
                agent = AssistantAgent("assistant", model_client=client)     # AutoGen 기본 AssistantAgent
                result = await agent.run(task=entry["task"])
                response = str(result.messages[-1].content) if result.messages else ""
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:300]}"
            finally:
                await client.close()
            blocked = sum(item.decision.value == "BLOCK" for item in judged)
            traces = [trace for item in judged for trace in item.traces]
            # SimGuardAgent가 막기 전 에이전트가 실제로 낸 답 (GT 라벨링용)
            raw = next((trace.action.final_response for trace in traces if trace.action.final_response is not None), None)
            row = {
                "id": entry["id"], "label": entry.get("label"), "method": entry.get("method"), "category": entry.get("category"),
                "mode": args.mode,
                "simguard_decision": ("BLOCK" if blocked else "ALLOW") if args.mode == "simguard" else None,
                "risk_score": max((item.risk_score for item in judged), default=None),
                "blocked": bool(blocked) if args.mode == "simguard" else None,
                "judgments": len(judged), "blocked_outputs": blocked,
                "judge_errors": [item.error for item in judged if item.error],
                "routes": [str(trace.backend_used.value if trace.backend_used else None) for trace in traces],
                "events": [f"{event.type}: {event.evidence[:200]}" for trace in traces
                           if trace.decision.value == "BLOCK" for event in trace.security_events][:3],
                "response": (response or "")[:4000],
                "raw_response": (raw if raw is not None else response or "")[:4000],
                "error": error, "elapsed_seconds": round(time.time() - started, 1),
            }
            async with write_lock:
                rows.append(row)
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if len(rows) % 100 == 0 or error:
                    print(f"{len(rows)}/{len(entries)} [{entry['id']}] {row['simguard_decision']} err={error}", flush=True)

    await asyncio.gather(*(run_one(entry) for entry in todo))
    rows.sort(key=lambda row: int(row["id"]) if str(row["id"]).isdigit() else str(row["id"]))
    (output / f"{args.label}.json").write_text(json.dumps({"benchmark": args.benchmark, "label": args.label, "mode": args.mode,
        "model_id": args.model_id, "judge_model_id": args.judge_model_id, "results": rows}, ensure_ascii=False, indent=1), encoding="utf-8")

    summary = {
        "benchmark": args.benchmark, "label": args.label, "mode": args.mode, "total": len(rows),
        "blocked": sum(row["simguard_decision"] == "BLOCK" for row in rows),
        "judge_errors": sum(bool(row["judge_errors"]) for row in rows),
        "errors": sum(bool(row["error"]) for row in rows),
        "model": args.model_id, "judge": args.judge_model_id,
    }
    (output / f"summary_{args.label}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
