"""
SimGuardAgent Jailbreak Benchmark Runner

사용법:
  # 서버 먼저 실행 (별도 터미널)
  cd agents/gpt-researcher && python backend/run_server.py

  # 원본 에이전트 (SimGuard 없음)
  python attacks-bench/jailbreak/run_benchmark.py --agent gpt-researcher --mode baseline

  # 에이전트 + SimGuardAgent
  python attacks-bench/jailbreak/run_benchmark.py --agent gpt-researcher --mode simguard

  # GA_Jailbreak_Benchmark (jailbreak only)
  python attacks-bench/jailbreak/run_benchmark.py --agent gpt-researcher --mode baseline --data ga-jailbreak

  # 소규모 테스트 (10개만)
  python attacks-bench/jailbreak/run_benchmark.py --agent gpt-researcher --mode simguard --limit 10

  # 결과 분석
  python attacks-bench/jailbreak/run_benchmark.py --analyze

새 에이전트 추가:
  attacks-bench/jailbreak/agents/<name>.py 에 AgentAdapter 구현 후
  아래 AGENT_REGISTRY에 등록
"""

import argparse
import asyncio
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import httpx

# ── 에이전트 어댑터 등록 ──────────────────────────────
from agents.gpt_researcher import GPTResearcherAdapter
from agents.smolagents_agent import SmolagentsAdapter
from agents.langgraph_agent import LanggraphAdapter

AGENT_REGISTRY: dict[str, type] = {
    "gpt-researcher": GPTResearcherAdapter,
    "smolagents":     SmolagentsAdapter,
    "langgraph":      LanggraphAdapter,
}
# ─────────────────────────────────────────────────────

DATA_DIR      = Path(__file__).parent / "data"
RESULTS_DIR   = Path(__file__).parent.parent.parent / "results" / "attacks-bench" / "jailbreak"
IPI_AGENT_DIR = Path(__file__).parent.parent / "ipi"
DELAY_BETWEEN = 2  # 요청 간 대기 (초)

_INJECAGENT_DIR   = DATA_DIR.parent.parent / "ipi" / "data" / "InjecAgent"
_SAFEDIAL_FILE    = DATA_DIR.parent.parent / "multi-turn" / "data" / "SafeDialBench" / "datasets_en.jsonl"
_INJECAGENT_FILE_MAP = {
    "injecagent":          ["dh-base", "dh-enhanced", "ds-base", "ds-enhanced"],
    "injecagent-base":     ["dh-base", "ds-base"],
    "injecagent-enhanced": ["dh-enhanced", "ds-enhanced"],
}
_INJECAGENT_PATHS = {
    "dh-base":     _INJECAGENT_DIR / "test_cases_dh_base.json",
    "dh-enhanced": _INJECAGENT_DIR / "test_cases_dh_enhanced.json",
    "ds-base":     _INJECAGENT_DIR / "test_cases_ds_base.json",
    "ds-enhanced": _INJECAGENT_DIR / "test_cases_ds_enhanced.json",
}


# ── 데이터 로딩 ───────────────────────────────────────

def load_data(data_type: str) -> list[dict]:
    # InjecAgent — data_type에 따라 파일 선택
    if data_type in _INJECAGENT_FILE_MAP:
        tags = _INJECAGENT_FILE_MAP[data_type]
        rows = []
        for tag in tags:
            raw = json.loads(_INJECAGENT_PATHS[tag].read_text(encoding="utf-8"))
            for i, r in enumerate(raw):
                rows.append({
                    "Index":             f"{tag}_{i:04d}",
                    "Goal":              r["User Instruction"],
                    "Category":          r.get("Attack Type", "Unknown"),
                    "_label":            "injected",
                    "_injected_content": r["Tool Response"],
                    "_inject_task":      r.get("Attacker Instruction", ""),
                    "_attack_style":     tag,
                })
        return rows

    # SafeDialBench (multi-turn jailbreak)
    if data_type in ("safedial", "safedial-plain"):
        rows = []
        for line in _SAFEDIAL_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            # safedial       → 마지막 multi-turn 질문 (우회 공격)
            # safedial-plain → plain_query (직접 공격)
            goal = e["plain_query"] if data_type == "safedial-plain" else e["history"][-1]["user"]
            rows.append({
                "Index":    str(e["id"]),
                "Goal":     goal,
                "Category": e.get("task", "unknown"),
                "_label":   "attack",
                "_method":  e.get("method", ""),
            })
        return rows

    # GA_Jailbreak_Benchmark
    if data_type in ("ga-jailbreak", "ga-jailbreak-benign"):
        path = DATA_DIR / "GA_Jailbreak_Benchmark" / "ga_jailbreak.csv"
        target_label = "1" if data_type == "ga-jailbreak" else "0"
        label_name   = "jailbreak" if data_type == "ga-jailbreak" else "benign"
        rows = []
        with open(path, encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if row["label"] != target_label:
                    continue
                rows.append({
                    "Index":     str(i),
                    "Goal":      row["text"],
                    "Category":  row["policy"],
                    "_label":    label_name,
                    "_strategy": row.get("strategy_id", ""),
                })
        return rows

    # HarmBench
    files = {
        "harmful": DATA_DIR / "harmful-behaviors.csv",
        "benign":  DATA_DIR / "benign-behaviors.csv",
    }
    rows = []
    with open(files[data_type], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["_label"] = data_type
            rows.append(row)
    return rows


# ── 결과 파일 경로 ─────────────────────────────────────

def result_path(agent: str, mode: str, data_type: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{agent}__{mode}__{data_type}.json"


def load_done_indices(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(r["index"]) for r in data.get("results", []) if not r.get("error")}


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        all_results = json.load(f).get("results", [])
    # 중복 제거: 같은 index면 성공 결과(non-error) 우선 유지
    seen: dict[str, dict] = {}
    for r in all_results:
        idx = str(r["index"])
        if idx not in seen or (seen[idx].get("error") and not r.get("error")):
            seen[idx] = r
    return list(seen.values())


def save_results(path: Path, agent: str, mode: str, data_type: str, results: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "agent":     agent,
            "mode":      mode,
            "data_type": data_type,
            "total":     len(results),
            "results":   results,
        }, f, indent=2, ensure_ascii=False)


# ── 벤치마크 실행 ──────────────────────────────────────

async def run_benchmark(agent_name: str, mode: str, data_type: str, limit: int | None, workers: int = 1):
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"알 수 없는 에이전트: '{agent_name}'. 가능: {list(AGENT_REGISTRY)}")

    # injecagent 데이터는 IPI 어댑터 사용
    # (jailbreak/agents 와 ipi/agents 가 같은 패키지명 충돌 → importlib으로 직접 로드)
    is_ipi = (data_type in _INJECAGENT_FILE_MAP)
    if is_ipi:
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "smolagents_ipi",
            IPI_AGENT_DIR / "agents" / "smolagents_ipi.py",
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        ipi_adapter = _mod.SmolagentsIPIAdapter()
    else:
        ipi_adapter = None

    adapter = AGENT_REGISTRY[agent_name]()
    rows    = load_data(data_type)
    if limit:
        rows = rows[:limit]

    out_path     = result_path(agent_name, mode, data_type)
    done_indices = load_done_indices(out_path)
    results      = load_results(out_path)
    todo         = [r for r in rows if str(r["Index"]) not in done_indices]

    total = len(rows)
    done  = len(done_indices)

    print(f"\n{'='*60}")
    print(f"에이전트 : {agent_name}")
    print(f"모드     : {mode.upper()}")
    print(f"데이터   : {data_type}  (전체={total}, 완료={done}, 남음={len(todo)})")
    print(f"workers  : {workers}")
    print(f"저장     : {out_path}")
    print(f"{'='*60}\n")

    sem   = asyncio.Semaphore(workers)
    lock  = asyncio.Lock()
    counter = {"n": done}

    async def process(row: dict, client: httpx.AsyncClient):
        async with sem:
            task  = row["Goal"]
            label = row["_label"]
            idx   = row["Index"]

            async with lock:
                counter["n"] += 1
                n = counter["n"]
            print(f"[{n}/{total}] [{label}] {task[:70]}...")

            if is_ipi:
                ipi_case = {
                    "user_task":        task,
                    "injected_content": row.get("_injected_content", ""),
                    "inject_task":      row.get("_inject_task", ""),
                }
                resp = await ipi_adapter.query(ipi_case, mode)
            else:
                resp = await adapter.query(task, mode, client=client)

            decision = resp.get("simguard_decision")
            risk     = resp.get("risk_score")
            report   = resp.get("report") or ""
            blocked  = (ipi_adapter if is_ipi else adapter).is_blocked(resp) if mode != "baseline" else None

            record = {
                "index":             idx,
                "category":          row["Category"],
                "label":             label,
                "goal":              task,
                "agent":             agent_name,
                "mode":              mode,
                "attack_style":      row.get("_attack_style", ""),
                "simguard_decision": decision,
                "risk_score":        risk,
                "blocked":           blocked,
                "report":            report,
                "guard_verdict":     resp.get("guard_verdict"),
                "error":             resp.get("error"),
            }

            async with lock:
                results.append(record)
                save_results(out_path, agent_name, mode, data_type, results)

            if mode != "baseline":
                print(f"  [{idx}] → decision={decision}  risk={risk}  blocked={blocked}")
            else:
                print(f"  [{idx}] → report={'있음' if report else '없음'}  error={resp.get('error')}")

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[process(row, client) for row in todo])

    print(f"\n완료 → {out_path}")


# ── 결과 분석 ──────────────────────────────────────────

def analyze():
    if not RESULTS_DIR.exists():
        print("결과 파일 없음.")
        return

    files = list(RESULTS_DIR.glob("*.json"))
    if not files:
        print("결과 파일 없음.")
        return

    agents = sorted({f.name.split("__")[0] for f in files})

    for agent in agents:
        print(f"\n{'='*60}")
        print(f"에이전트: {agent}")
        print(f"{'='*60}")

        for mode in ["baseline", "simguard"]:
            agent_files = [f for f in files if f.name.startswith(f"{agent}__{mode}__")]
            if not agent_files:
                continue

            rows = []
            for f in agent_files:
                rows.extend(load_results(f))

            harmful = [r for r in rows if r["label"] == "harmful"]
            benign  = [r for r in rows if r["label"] == "benign"]

            print(f"\n  [ {mode.upper()} ]  총={len(rows)} (harmful={len(harmful)}, benign={len(benign)})")

            if mode == "simguard":
                if harmful:
                    tp  = sum(1 for r in harmful if r["blocked"] is True)
                    tpr = tp / len(harmful) * 100
                    print(f"  TPR (harmful → BLOCK) : {tp}/{len(harmful)} = {tpr:.1f}%")
                if benign:
                    fp_ = sum(1 for r in benign if r["blocked"] is True)
                    fpr = fp_ / len(benign) * 100
                    print(f"  FPR (benign  → BLOCK) : {fp_}/{len(benign)} = {fpr:.1f}%")

                if harmful:
                    cat = defaultdict(lambda: [0, 0])
                    for r in harmful:
                        cat[r["category"]][1] += 1
                        if r["blocked"] is True:
                            cat[r["category"]][0] += 1
                    print("  카테고리별 탐지율 (harmful):")
                    for c, (b, t) in sorted(cat.items()):
                        bar = "█" * int(b / t * 20)
                        print(f"    {c:<35} {bar:<20} {b}/{t} ({b/t*100:.0f}%)")

                scores = [r["risk_score"] for r in rows if r.get("risk_score") is not None]
                if scores:
                    print(f"  평균 risk_score: {sum(scores)/len(scores):.3f}")
            else:
                errors = sum(1 for r in rows if r.get("error"))
                print(f"  오류: {errors}건")


# ── 진입점 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SimGuardAgent Jailbreak Benchmark")
    parser.add_argument("--agent",   default="gpt-researcher",
                        choices=list(AGENT_REGISTRY),
                        help="테스트할 에이전트")
    parser.add_argument("--mode",    default="simguard",
                        choices=["baseline", "simguard", "llamaguard", "gptsafeguard"],
                        help="baseline / simguard / llamaguard / gptsafeguard")
    parser.add_argument("--data",    default="harmful",
                        choices=["harmful", "benign", "ga-jailbreak", "ga-jailbreak-benign",
                                 "injecagent", "injecagent-base", "injecagent-enhanced",
                                 "safedial", "safedial-plain"],
                        help="injecagent=전체 / injecagent-base=자연스러운주입 / injecagent-enhanced=명시적주입")
    parser.add_argument("--limit",   type=int, default=None,
                        help="테스트할 최대 개수 (디버깅용)")
    parser.add_argument("--workers", type=int, default=1,
                        help="동시 요청 수 (기본값: 1)")
    parser.add_argument("--analyze", action="store_true",
                        help="저장된 결과 분석")
    args = parser.parse_args()

    if args.analyze:
        analyze()
    else:
        asyncio.run(run_benchmark(args.agent, args.mode, args.data, args.limit, args.workers))


if __name__ == "__main__":
    main()
