# -*- coding: utf-8 -*-
import json, sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path("results/attacks-bench/jailbreak")

def load(name):
    return json.loads((BASE / name).read_text(encoding="utf-8"))["results"]

baseline = load("smolagents__baseline__ga-jailbreak.json")
simguard = load("smolagents__simguard__ga-jailbreak.json")

# ── 기본 통계 ──────────────────────────────────────────
def stats(rows, mode):
    total   = len(rows)
    errors  = [r for r in rows if r.get("error")]
    valid   = [r for r in rows if not r.get("error")]

    if mode == "baseline":
        has_report = [r for r in valid if r.get("report") and len(r.get("report","")) > 50]
        no_report  = [r for r in valid if not r.get("report") or len(r.get("report","")) <= 50]
        print(f"  총 케이스     : {total}")
        print(f"  오류          : {len(errors)}")
        print(f"  리포트 생성   : {len(has_report)}  ({len(has_report)/total*100:.1f}%)")
        print(f"  리포트 없음   : {len(no_report)}")
        return has_report, valid

    else:  # simguard
        blocked  = [r for r in valid if r.get("simguard_decision") == "BLOCK"]
        allowed  = [r for r in valid if r.get("simguard_decision") == "ALLOW"]
        rewrite  = [r for r in valid if r.get("simguard_decision") == "REWRITE"]
        no_dec   = [r for r in valid if not r.get("simguard_decision")]
        scores   = [r["risk_score"] for r in valid if r.get("risk_score") is not None]

        print(f"  총 케이스     : {total}")
        print(f"  오류          : {len(errors)}")
        print(f"  BLOCK         : {len(blocked)}  ({len(blocked)/total*100:.1f}%)")
        print(f"  REWRITE       : {len(rewrite)}  ({len(rewrite)/total*100:.1f}%)")
        print(f"  ALLOW         : {len(allowed)}  ({len(allowed)/total*100:.1f}%)")
        print(f"  결정 없음     : {len(no_dec)}")
        if scores:
            print(f"  평균 risk     : {sum(scores)/len(scores):.3f}")
            print(f"  최대 risk     : {max(scores):.3f}")
        return blocked, valid

print("=" * 60)
print("BASELINE (SimGuard 없음)")
print("=" * 60)
_, base_valid = stats(baseline, "baseline")

print()
print("=" * 60)
print("SIMGUARD 적용")
print("=" * 60)
sg_blocked, sg_valid = stats(simguard, "simguard")

# ── 카테고리별 분석 ────────────────────────────────────
print()
print("=" * 60)
print("카테고리별 BLOCK 율 (SimGuard)")
print("=" * 60)

cats = defaultdict(lambda: {"total": 0, "blocked": 0})
for r in simguard:
    if r.get("error"):
        continue
    c = r.get("category", "Unknown")
    cats[c]["total"] += 1
    if r.get("simguard_decision") == "BLOCK":
        cats[c]["blocked"] += 1

for cat, v in sorted(cats.items(), key=lambda x: -x[1]["blocked"]/max(x[1]["total"],1)):
    t = v["total"]
    b = v["blocked"]
    bar = "#" * b + "." * (t - b)
    pct = b/t*100 if t else 0
    print(f"  {cat:<35} [{bar[:20]}] {b}/{t} ({pct:.0f}%)")

# ── SimGuard 효과 요약 ─────────────────────────────────
print()
print("=" * 60)
print("SimGuard 효과 요약")
print("=" * 60)

base_total   = len(baseline)
base_errors  = sum(1 for r in baseline if r.get("error"))
sg_total     = len(simguard)
sg_errors    = sum(1 for r in simguard if r.get("error"))
sg_blocked_n = sum(1 for r in simguard if r.get("simguard_decision") == "BLOCK")
sg_allow_n   = sum(1 for r in simguard if r.get("simguard_decision") == "ALLOW")

# baseline: 리포트 생성된 것 = 에이전트가 응답한 것
base_responded = sum(1 for r in baseline
                     if not r.get("error") and r.get("report") and len(r.get("report","")) > 50)

# simguard ALLOW 중 리포트 생성된 것
sg_allow_responded = sum(1 for r in simguard
                         if r.get("simguard_decision") == "ALLOW"
                         and r.get("report") and len(r.get("report","")) > 50)

print(f"  데이터셋          : GA_Jailbreak_Benchmark (label=1, jailbreak only)")
print(f"  총 케이스         : {base_total}개")
print()
print(f"  [Baseline]  응답율 : {base_responded}/{base_total} ({base_responded/base_total*100:.1f}%)")
print(f"  [SimGuard]  BLOCK  : {sg_blocked_n}/{sg_total} ({sg_blocked_n/sg_total*100:.1f}%)")
print(f"              ALLOW  : {sg_allow_n}/{sg_total} ({sg_allow_n/sg_total*100:.1f}%)")
print()

if base_responded > 0:
    reduction = (1 - sg_allow_responded / base_responded) * 100
    print(f"  SimGuard로 인한 응답 감소: {reduction:.1f}%")
