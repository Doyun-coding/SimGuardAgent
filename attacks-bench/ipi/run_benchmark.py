"""
SimGuardAgent IPI Benchmark Runner

데이터셋:
  base     — test_cases_dh_base.json     (510 케이스, 공격)
  enhanced — test_cases_dh_enhanced.json  (510 케이스, 강화 공격)
  benign   — user_cases.jsonl             ( 17 케이스, 정상 → False Positive 측정)

사용법:
  python attacks-bench/ipi/run_benchmark.py --dataset base     --mode baseline
  python attacks-bench/ipi/run_benchmark.py --dataset base     --mode simguard
  python attacks-bench/ipi/run_benchmark.py --dataset enhanced --mode baseline
  python attacks-bench/ipi/run_benchmark.py --dataset enhanced --mode simguard
  python attacks-bench/ipi/run_benchmark.py --dataset benign   --mode baseline
  python attacks-bench/ipi/run_benchmark.py --dataset benign   --mode simguard
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_IPI_ROOT = Path(__file__).parent
if str(_IPI_ROOT) not in sys.path:
    sys.path.insert(0, str(_IPI_ROOT))

from agents.smolagents_ipi import SmolagentsIPIAdapter

INJECAGENT_DIR = _IPI_ROOT / "data" / "InjecAgent"
RESULTS_DIR    = _REPO_ROOT / "results" / "attacks-bench" / "ipi" / "injecagent"
DELAY          = 1

DATASET_FILES = {
    "base":     INJECAGENT_DIR / "test_cases_dh_base.json",
    "enhanced": INJECAGENT_DIR / "test_cases_dh_enhanced.json",
    # dh/ds × base/enhanced 4개를 합친 것 (2108건). setting·variant 필드로 사후 분해 가능
    "all":      INJECAGENT_DIR / "test_cases_all.json",
    "benign":   INJECAGENT_DIR / "user_cases.jsonl",
}

BENIGN_FILLER = "No additional content available."


def load_dataset(dataset: str) -> list[dict]:
    """원본 InjecAgent 포맷 → 내부 case 포맷으로 변환."""
    path = DATASET_FILES[dataset]

    if dataset == "benign":
        raw = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        cases = []
        for i, r in enumerate(raw):
            tool_resp = r["Tool Response Template"].replace("<Attacker Instruction>", BENIGN_FILLER)
            cases.append({
                "id":               f"benign_{i:04d}",
                "label":            "benign",
                "category":         "benign",
                "user_task":        r["User Instruction"],
                "injected_content": tool_resp,
                "inject_task":      "",
            })
        return cases
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        prefix = {"base": "dh-base", "enhanced": "dh-enhanced"}.get(dataset, "all")
        cases = []
        for i, r in enumerate(raw):
            # 합친 파일(all)은 레코드마다 고유 id·setting·variant를 이미 갖고 있다
            case_id = r["id"] if "id" in r else f"{prefix}_{i:04d}"
            cases.append({
                "id":               case_id,
                "label":            "injected",
                "category":         r.get("Attack Type", ""),
                "user_task":        r["User Instruction"],
                "injected_content": r["Tool Response"],
                "inject_task":      r.get("Attacker Instruction", ""),
                "setting":          r.get("setting", ""),   # dh / ds
                "variant":          r.get("variant", ""),   # base / enhanced
            })
        return cases


def result_path(dataset: str, mode: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = {"baseline": "baseline", "simguard": "simguard", "llamaguard": "llamaguard", "gptsafeguard": "gptsafeguard"}[mode]
    return RESULTS_DIR / f"smolagents__{tag}__injecagent-{dataset}.json"


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(r["id"]) for r in json.loads(path.read_text(encoding="utf-8")).get("results", []) if not r.get("error")}


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    all_results = json.loads(path.read_text(encoding="utf-8")).get("results", [])
    seen: dict[str, dict] = {}
    for r in all_results:
        idx = str(r["id"])
        if idx not in seen or (seen[idx].get("error") and not r.get("error")):
            seen[idx] = r
    return list(seen.values())


def save_results(path: Path, dataset: str, mode: str, results: list[dict]):
    n       = len(results)
    n_blk   = sum(1 for r in results if r.get("blocked") is True)
    path.write_text(
        json.dumps({
            "agent":      "smolagents",
            "mode":       mode,
            "dataset":    dataset,
            "total":      n,
            "block_rate": round(n_blk / n, 4) if n else 0,
            "results":    results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


async def run_benchmark(dataset: str, mode: str, limit: int | None, workers: int):
    adapter = SmolagentsIPIAdapter()

    cases = load_dataset(dataset)
    if limit:
        cases = cases[:limit]

    out_path  = result_path(dataset, mode)
    done_ids  = load_done(out_path)
    results   = load_results(out_path)
    todo      = [c for c in cases if c["id"] not in done_ids]
    total     = len(cases)
    done      = len(done_ids)

    print(f"\n{'='*60}")
    print(f"  데이터셋 : {dataset}")
    print(f"  모드     : {mode.upper()}")
    print(f"  케이스   : 전체={total}  완료={done}  남음={len(todo)}")
    print(f"  workers  : {workers}")
    print(f"  저장     : {out_path}")
    print(f"{'='*60}\n")

    sem     = asyncio.Semaphore(workers)
    lock    = asyncio.Lock()
    counter = {"n": done}

    async def process(case: dict):
        async with sem:
            async with lock:
                counter["n"] += 1
                n = counter["n"]

            print(f"[{n}/{total}] [{case['label']}:{case['category']}] {case['user_task'][:65]}...")

            resp     = await adapter.query(case, mode)
            decision = resp.get("simguard_decision")
            risk     = resp.get("risk_score")
            blocked  = adapter.is_blocked(resp) if mode != "baseline" else None

            record = {
                "id":               case["id"],
                "category":         case["category"],
                "label":            case["label"],
                "user_task":        case["user_task"],
                "injected_content": case["injected_content"][:200],
                "inject_task":      case.get("inject_task", ""),
                "mode":             mode,
                "simguard_decision":decision,
                "risk_score":       risk,
                "blocked":          blocked,
                "ipi_events":       resp.get("ipi_events", []),
                "report":           resp.get("report") or "",
                "error":            resp.get("error"),
            }

            async with lock:
                results.append(record)
                save_results(out_path, dataset, mode, results)

            if mode != "baseline":
                risk_str = f"{risk:.3f}" if risk is not None else "N/A"
                lg_str   = f"  lg_verdict={resp.get('lg_verdict')}" if mode == "llamaguard" else ""
                print(f"  [{case['id']}] decision={decision}  risk={risk_str}  blocked={blocked}{lg_str}")
            else:
                print(f"  [{case['id']}] report_len={len(record['report'])}  error={resp.get('error')}")

            await asyncio.sleep(DELAY)

    await asyncio.gather(*[process(c) for c in todo])

    # 최종 요약
    all_results = load_results(out_path)
    n     = len(all_results)
    n_blk = sum(1 for r in all_results if r.get("blocked") is True)
    print(f"\n{'='*55}")
    print(f"  완료 케이스 : {n}")
    if mode != "baseline":
        print(f"  차단 (BLOCK): {n_blk}/{n} ({n_blk/n*100:.1f}%)")
    print(f"{'='*55}")
    print(f"\n완료 → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="SimGuardAgent IPI Benchmark")
    parser.add_argument("--dataset", default="base",
                        choices=["base", "enhanced", "all", "benign"],
                        help="base/enhanced=dh만, all=dh+ds 2108건, benign=정상")
    parser.add_argument("--mode",    default="simguard", choices=["baseline", "simguard", "llamaguard", "gptsafeguard"])
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.dataset, args.mode, args.limit, args.workers))


if __name__ == "__main__":
    main()
