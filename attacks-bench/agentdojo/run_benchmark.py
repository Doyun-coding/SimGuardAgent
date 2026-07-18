"""
AgentDojo + SimGuard 벤치마크 러너

Usage:
    python run_benchmark.py --suite banking --attack tool_knowledge --defense simguard
    python run_benchmark.py --suite banking --defense baseline
    python run_benchmark.py --suite banking --defense simguard --workers 4

defense 옵션:
    simguard   - SimGuardPreLLMChecker + SimGuardToolsExecutor
    baseline   - ToolsExecutor (방어 없음)

--workers:
    user task 단위 병렬 실행. 각 스레드가 독립 파이프라인 인스턴스 사용.
    권장: simguard 3~5, baseline 5~10
"""
from __future__ import annotations

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.logging import RichHandler

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
_BENCH_DIR = Path(__file__).parent
_REPO_ROOT  = _BENCH_DIR.parents[1]
_AGENTDOJO  = _REPO_ROOT / "agents" / "agentdojo" / "src"

for p in [str(_REPO_ROOT), str(_AGENTDOJO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

load_dotenv()

# ── AgentDojo imports ─────────────────────────────────────────────────────────
import openai
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor, tool_result_to_str
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import (
    SuiteResults,
    benchmark_suite_without_injections,
    run_task_with_injection_tasks,
)
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite

# ── SimGuard executors ────────────────────────────────────────────────────────
from simguard_executor import SimGuardPreLLMChecker, SimGuardToolsExecutor

BENCHMARK_VERSION = "v1.2.2"


# ─────────────────────────────────────────────────────────────────────────────
# 파이프라인 팩토리 — 스레드마다 독립 인스턴스 생성
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(defense: str, model_id: str, base_url: str, api_key: str) -> AgentPipeline:
    """스레드마다 호출해 독립적인 파이프라인 인스턴스를 생성한다."""
    client     = openai.OpenAI(api_key=api_key, base_url=base_url)
    llm        = OpenAILLM(client, model_id)
    system_msg = SystemMessage(load_system_message(None))
    init_query = InitQuery()

    if defense == "simguard":
        pre_checker = SimGuardPreLLMChecker()
        executor    = SimGuardToolsExecutor(tool_output_formatter=tool_result_to_str)
        tools_loop  = ToolsExecutionLoop([executor, llm])
        pipeline    = AgentPipeline([system_msg, init_query, pre_checker, llm, tools_loop])
        pipeline.name = f"{model_id}-simguard"

    else:  # baseline
        executor   = ToolsExecutor(tool_result_to_str)
        tools_loop = ToolsExecutionLoop([executor, llm])
        pipeline   = AgentPipeline([system_msg, init_query, llm, tools_loop])
        pipeline.name = f"{model_id}-baseline"

    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# 병렬 벤치마크 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_with_injections_parallel(
    suite_obj,
    attack_name: str,
    defense: str,
    model_id: str,
    base_url: str,
    api_key: str,
    user_tasks: tuple[str, ...],
    injection_tasks: tuple[str, ...],
    logdir: Path,
    force_rerun: bool,
    workers: int,
) -> SuiteResults:
    """user task 단위로 병렬 실행. 각 스레드가 독립 파이프라인을 생성한다."""

    # 실행할 user task 목록 결정
    if user_tasks:
        ut_ids = list(user_tasks)
    else:
        ut_ids = list(suite_obj.user_tasks.keys())

    it_ids = list(injection_tasks) if injection_tasks else None

    utility_results:   dict[tuple[str, str], bool] = {}
    security_results:  dict[tuple[str, str], bool] = {}
    completed = [0]
    total     = len(ut_ids)

    def run_one(ut_id: str):
        """단일 user task를 모든 injection task에 대해 실행."""
        # 스레드마다 새 파이프라인 — SimGuardToolsExecutor 상태 격리
        pipeline   = build_pipeline(defense, model_id, base_url, api_key)
        attacker   = load_attack(attack_name, suite_obj, pipeline)
        user_task  = suite_obj.get_user_task_by_id(ut_id)

        with OutputLogger(str(logdir)):
            u_res, s_res = run_task_with_injection_tasks(
                suite=suite_obj,
                agent_pipeline=pipeline,
                user_task=user_task,
                attack=attacker,
                logdir=logdir,
                force_rerun=force_rerun,
                injection_tasks=it_ids,
                benchmark_version=BENCHMARK_VERSION,
            )
        return u_res, s_res

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, ut_id): ut_id for ut_id in ut_ids}
        for fut in as_completed(futures):
            ut_id = futures[fut]
            try:
                u_res, s_res = fut.result()
                utility_results.update(u_res)
                security_results.update(s_res)
            except Exception as e:
                logging.error(f"[{ut_id}] failed: {e}")
            finally:
                completed[0] += 1
                blocked = sum(1 for v in security_results.values() if v)
                print(f"  [{completed[0]:>3}/{total}] {ut_id} done"
                      f"  |  security so far: {blocked}/{len(security_results)}")

    return SuiteResults(
        utility_results=utility_results,
        security_results=security_results,
        injection_tasks_utility_results={},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 결과 저장 / CLI
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: SuiteResults, outpath: Path, suite: str, defense: str, attack: str | None):
    out = {"suite": suite, "defense": defense, "attack": attack, "utility": {}, "security": {}}

    for (ut, it), v in results["utility_results"].items():
        out["utility"][f"{ut}/{it}"] = v
    for (ut, it), v in results["security_results"].items():
        out["security"][f"{ut}/{it}"] = v

    u_vals = list(results["utility_results"].values())
    s_vals = list(results["security_results"].values())
    out["avg_utility"]  = round(sum(u_vals) / len(u_vals) * 100, 2) if u_vals else 0.0
    out["avg_security"] = round(sum(s_vals) / len(s_vals) * 100, 2) if s_vals else None

    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


@click.command()
@click.option("--suite", "-s", default="banking",
              type=click.Choice(["banking", "workspace", "slack", "travel"]))
@click.option("--attack", "-a", default=None, type=str,
              help="공격 이름 (e.g. tool_knowledge). 없으면 utility만 측정.")
@click.option("--defense", "-d", default="simguard",
              type=click.Choice(["simguard", "baseline"]))
@click.option("--workers", "-w", default=1, type=int,
              help="병렬 실행 수 (user task 단위). simguard 권장 3~5, baseline 권장 5~10.")
@click.option("--user-task", "-ut", "user_tasks", multiple=True)
@click.option("--injection-task", "-it", "injection_tasks", multiple=True)
@click.option("--model-id", default=None)
@click.option("--logdir", default="./runs", type=Path)
@click.option("--outdir", default="./results", type=Path)
@click.option("--force-rerun", "-f", is_flag=True)
def main(
    suite, attack, defense, workers,
    user_tasks, injection_tasks,
    model_id, logdir, outdir, force_rerun,
):
    model_id = model_id or os.environ.get("OPENAI_MODEL_ID", "gpt-4o-mini")
    base_url  = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key   = os.environ.get("OPENAI_API_KEY", "")

    print(f"[bench] suite={suite}  defense={defense}  attack={attack}"
          f"  model={model_id}  workers={workers}")

    suite_obj = get_suite(BENCHMARK_VERSION, suite)
    logdir    = logdir / f"{suite}_{defense}" / (attack or "no_attack")
    logdir.mkdir(parents=True, exist_ok=True)

    if attack is None:
        # utility 측정 — 순차 실행 (workers 무관)
        pipeline = build_pipeline(defense, model_id, base_url, api_key)
        with OutputLogger(str(logdir)):
            results = benchmark_suite_without_injections(
                pipeline, suite_obj,
                user_tasks=list(user_tasks) or None,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=BENCHMARK_VERSION,
            )
    else:
        results = run_with_injections_parallel(
            suite_obj=suite_obj,
            attack_name=attack,
            defense=defense,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            user_tasks=user_tasks,
            injection_tasks=injection_tasks,
            logdir=logdir,
            force_rerun=force_rerun,
            workers=workers,
        )

    outpath = outdir / f"{suite}_{defense}_{attack or 'no_attack'}.json"
    out     = save_results(results, outpath, suite, defense, attack)

    print(f"\n[결과] {suite} / {defense} / {attack or 'no attack'}")
    print(f"  Utility  : {out['avg_utility']:.1f}%")
    if out["avg_security"] is not None:
        print(f"  Security : {out['avg_security']:.1f}%")
    print(f"  저장됨   : {outpath}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[RichHandler(show_path=False, markup=True)],
    )
    main()
