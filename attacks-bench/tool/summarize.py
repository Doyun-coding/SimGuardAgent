"""
AgentDojo 벤치마크 결과 분석

사용법:
  python attacks-bench/tool/summarize.py
  python attacks-bench/tool/summarize.py --suite github
  python attacks-bench/tool/summarize.py --suite github banking
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
RESULTS_DIR = _REPO_ROOT / "results" / "attacks-bench" / "tool" / "agentdojo"


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("results", [])


def stats(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    n_util    = sum(1 for r in results if r.get("utility")  is True)
    n_sec     = sum(1 for r in results if r.get("security") is True)
    n_blk     = sum(1 for r in results if r.get("blocked")  is True)
    n_timeout = sum(1 for r in results if r.get("error") == "timeout")
    n_error   = sum(1 for r in results if r.get("error") and r.get("error") != "timeout")
    return {
        "total":        n,
        "utility":      n_util,
        "utility_rate": n_util / n,
        "attack_ok":    n_sec,
        "asr":          n_sec / n,
        "blocked":      n_blk,
        "block_rate":   n_blk / n,
        "timeout":      n_timeout,
        "error":        n_error,
    }


def print_suite(suite: str):
    b_path = RESULTS_DIR / "baseline" / f"{suite}.json"
    s_path = RESULTS_DIR / "simguard" / f"{suite}.json"
    b = load(b_path)
    s = load(s_path)

    if not b and not s:
        print(f"  [{suite}] 결과 파일 없음")
        return

    bs = stats(b)
    ss = stats(s)

    print(f"\n{'━'*60}")
    print(f"  Suite: {suite.upper()}")
    print(f"{'━'*60}")
    print(f"  {'항목':<28} {'Baseline':>10}  {'SimGuard':>10}")
    print(f"  {'-'*50}")

    def row(label, b_val, s_val, fmt=".1%"):
        bv = f"{b_val:{fmt}}" if b_val is not None else "N/A"
        sv = f"{s_val:{fmt}}" if s_val is not None else "N/A"
        print(f"  {label:<28} {bv:>10}  {sv:>10}")

    def row_n(label, b_val, b_n, s_val, s_n):
        bv = f"{b_val}/{b_n}" if b_val is not None else "N/A"
        sv = f"{s_val}/{s_n}" if s_val is not None else "N/A"
        print(f"  {label:<28} {bv:>10}  {sv:>10}")

    if bs:
        row_n("총 케이스",       bs["total"], "",    ss.get("total"), "")
        row_n("utility (완료)",  bs["utility"], bs["total"], ss.get("utility"), ss.get("total"))
        row("utility rate",      bs["utility_rate"],  ss.get("utility_rate"))
        row_n("공격 성공",       bs["attack_ok"], bs["total"], ss.get("attack_ok"), ss.get("total"))
        row("공격 성공률 (ASR)", bs["asr"],           ss.get("asr"))
        if ss:
            row_n("SimGuard 차단", None, None, ss["blocked"], ss["total"])
            row("차단율",         None,                ss.get("block_rate"))
        row_n("timeout",         bs["timeout"], bs["total"], ss.get("timeout"), ss.get("total"))
        row_n("기타 에러",       bs["error"],   bs["total"], ss.get("error"),   ss.get("total"))

    # 방어 효과 계산
    if bs and ss and bs["asr"] > 0:
        reduction = (bs["asr"] - ss.get("asr", 0)) / bs["asr"]
        print(f"\n  {'공격 성공률 감소율':<28} {reduction:>10.1%}")
    print()


def print_injection_breakdown(suite: str):
    """injection task별 공격 성공률"""
    b_path = RESULTS_DIR / "baseline" / f"{suite}.json"
    s_path = RESULTS_DIR / "simguard" / f"{suite}.json"
    b = load(b_path)
    s = load(s_path)
    if not b:
        return

    # injection task별 집계
    inj_ids = sorted({r["inj_task_id"] for r in b})

    print(f"\n  [injection task별 공격 성공률 — {suite}]")
    print(f"  {'inj_task_id':<22} {'baseline ASR':>14}  {'simguard ASR':>14}  {'차단됨':>8}")
    print(f"  {'-'*62}")

    for inj_id in inj_ids:
        b_sub = [r for r in b if r["inj_task_id"] == inj_id]
        s_sub = [r for r in s if r["inj_task_id"] == inj_id]
        b_asr = sum(1 for r in b_sub if r.get("security") is True) / len(b_sub) if b_sub else 0
        s_asr = sum(1 for r in s_sub if r.get("security") is True) / len(s_sub) if s_sub else 0
        s_blk = sum(1 for r in s_sub if r.get("blocked") is True)
        print(f"  {inj_id:<22} {b_asr:>13.1%}  {s_asr:>13.1%}  {s_blk:>6}/{len(s_sub) if s_sub else '-'}")


def main():
    parser = argparse.ArgumentParser(description="AgentDojo 벤치마크 결과 분석")
    parser.add_argument("--suite", nargs="*",
                        help="분석할 스위트 (기본: 결과 파일 있는 것 전부)")
    parser.add_argument("--breakdown", action="store_true",
                        help="injection task별 세부 분석 출력")
    args = parser.parse_args()

    # 분석할 스위트 결정
    if args.suite:
        suites = args.suite
    else:
        # 결과 파일이 있는 스위트 자동 탐색
        found = set()
        for p in RESULTS_DIR.rglob("*.json"):
            found.add(p.stem)
        suites = sorted(found)

    if not suites:
        print("결과 파일이 없습니다.")
        return

    print(f"\n{'━'*60}")
    print(f"  AgentDojo 벤치마크 결과 분석")
    print(f"  대상: {', '.join(suites)}")
    print(f"{'━'*60}")

    for suite in suites:
        print_suite(suite)
        if args.breakdown:
            print_injection_breakdown(suite)

    # 전체 합산 요약
    all_b, all_s = [], []
    for suite in suites:
        all_b += load(RESULTS_DIR / "baseline" / f"{suite}.json")
        all_s += load(RESULTS_DIR / "simguard" / f"{suite}.json")

    if len(suites) > 1 and all_b:
        bs, ss = stats(all_b), stats(all_s)
        print(f"{'━'*60}")
        print(f"  전체 합산 ({', '.join(suites)})")
        print(f"{'━'*60}")
        print(f"  {'항목':<28} {'Baseline':>10}  {'SimGuard':>10}")
        print(f"  {'-'*50}")
        print(f"  {'총 케이스':<28} {bs['total']:>10}  {ss.get('total',0):>10}")
        print(f"  {'utility rate':<28} {bs['utility_rate']:>10.1%}  {ss.get('utility_rate',0):>10.1%}")
        print(f"  {'공격 성공률 (ASR)':<28} {bs['asr']:>10.1%}  {ss.get('asr',0):>10.1%}")
        if ss:
            print(f"  {'차단율':<28} {'N/A':>10}  {ss.get('block_rate',0):>10.1%}")
        if bs["asr"] > 0 and ss:
            reduction = (bs["asr"] - ss["asr"]) / bs["asr"]
            print(f"  {'공격 성공률 감소율':<28} {reduction:>10.1%}")
        print()


if __name__ == "__main__":
    main()
