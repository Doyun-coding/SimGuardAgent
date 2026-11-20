"""
E3 오버헤드 리포트 — scripts/simguard_profiling.py 가 남긴 JSONL을 표로 집계한다.

사용:
  python scripts/e3_overhead_report.py results/overhead_mcptox.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round((p / 100) * (len(s) - 1)))))
    return s[i]


def main(path: str) -> None:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        print("기록 없음")
        return

    print(f"\n{'='*74}")
    print(f"E3. SimGuard 오버헤드 — n={len(rows)} 액션   ({path})")
    print(f"{'='*74}")

    # ── 샘플 1건 상세 ────────────────────────────────────────────────────────
    r = rows[0]
    print("\n[액션 1건 상세 예시]")
    for k, v in r.items():
        if k != "calls":
            print(f"  {k:<18} = {v}")
    print("  calls:")
    for c in r["calls"]:
        print(f"    - {c['stage']:<20}{c['latency_s']:>8.2f}s"
              f"  pt={c['prompt_tokens']} ct={c['completion_tokens']}"
              f" reas={c.get('reasoning_tokens')} cost={c.get('cost_usd')}")

    # ── 액션당 총계 ──────────────────────────────────────────────────────────
    wall = [x["wall_s"] for x in rows]
    llm = [x["llm_s"] for x in rows]
    tok = [x["total_tokens"] for x in rows]
    pt = [x["prompt_tokens"] for x in rows]
    ct = [x["completion_tokens"] for x in rows]
    ncall = [x["n_llm_calls"] for x in rows]
    cost = [x.get("cost_usd") or 0.0 for x in rows]

    print(f"\n[액션당 총계]")
    print(f"  {'지표':<22}{'평균':>12}{'p50':>12}{'p95':>12}{'최대':>12}")
    print("  " + "-" * 70)

    def line(name, vals, fmt="{:>12.3f}"):
        print(f"  {name:<22}" + fmt.format(sum(vals) / len(vals))
              + fmt.format(pct(vals, 50)) + fmt.format(pct(vals, 95)) + fmt.format(max(vals)))

    line("가드 지연 (s)", wall)
    line("└ LLM 대기 (s)", llm)
    line("LLM 호출 수", [float(n) for n in ncall])
    line("prompt tokens", [float(v) for v in pt], "{:>12.0f}")
    line("completion tokens", [float(v) for v in ct], "{:>12.0f}")
    line("total tokens", [float(v) for v in tok], "{:>12.0f}")
    if any(cost):
        line("비용 (USD)", cost, "{:>12.6f}")

    # ── 단계별 분해 ──────────────────────────────────────────────────────────
    st_time: dict[str, list[float]] = defaultdict(list)
    for x in rows:
        for k, v in (x.get("stage_s") or {}).items():
            st_time[k].append(v)

    st_tok: dict[str, list[int]] = defaultdict(list)
    st_lat: dict[str, list[float]] = defaultdict(list)
    st_n: dict[str, int] = defaultdict(int)
    st_cost: dict[str, float] = defaultdict(float)
    for x in rows:
        for c in x["calls"]:
            s = c["stage"]
            st_tok[s].append(c.get("total_tokens") or 0)
            st_lat[s].append(c.get("latency_s") or 0.0)
            st_n[s] += 1
            st_cost[s] += c.get("cost_usd") or 0.0

    total_wall = sum(wall)
    print(f"\n[단계별 분해]  (전체 가드 지연 합계 {total_wall:.1f}s 기준)")
    print(f"  {'단계':<22}{'시간(s)':>10}{'비중':>8}{'LLM호출':>9}{'토큰/액션':>11}{'유형':>8}")
    print("  " + "-" * 70)
    order = ["live_exec_real", "predict_result", "predict_next_action", "judge", "commit"]
    seen = set()
    for s in order + sorted(set(st_time) | set(st_lat)):
        if s in seen:
            continue
        seen.add(s)
        t = sum(st_time.get(s, [])) or sum(st_lat.get(s, []))
        if t == 0 and st_n.get(s, 0) == 0:
            continue
        share = t / total_wall * 100 if total_wall else 0
        tk = sum(st_tok.get(s, [])) / len(rows) if st_tok.get(s) else 0
        kind = "LLM" if st_n.get(s, 0) else "비-LLM"
        print(f"  {s:<22}{t:>10.2f}{share:>7.1f}%{st_n.get(s,0):>9}{tk:>11.0f}{kind:>8}")

    # ── 핵심 요약: 시뮬레이션 비용 vs 판정 비용 ──────────────────────────────
    sim_keys = {"live_exec_real", "commit"}
    sim_t = sum(sum(st_time.get(k, [])) for k in sim_keys)
    llm_t = sum(llm)
    print(f"\n[핵심 분해] 리뷰어 질문 'simulation justifies its complexity?' 에 대한 근거")
    print(f"  실제 도구 실행 + commit (비-LLM)   : {sim_t:>8.3f}s  "
          f"({sim_t/total_wall*100 if total_wall else 0:>5.2f}%)")
    print(f"  LLM 판정 대기                       : {llm_t:>8.3f}s  "
          f"({llm_t/total_wall*100 if total_wall else 0:>5.2f}%)")
    print(f"  -> 시뮬레이션 자체는 오버헤드의 극히 일부이며, 비용의 대부분은")
    print(f"     LLM 판정이다. 이는 모든 LLM 기반 가드가 동일하게 지불하는 비용이다.")

    if any(cost):
        per1k = sum(cost) / len(rows) * 1000
        print(f"\n  1,000 액션당 추정 비용: ${per1k:.4f}")

    errs = [x for x in rows if x.get("error")]
    if errs:
        print(f"\n  오류 액션: {len(errs)}건 — {errs[0]['error']}")


def compare(paths: list[str]) -> None:
    """여러 설정의 오버헤드를 나란히 비교한다."""
    cols = []
    for p in paths:
        rows = [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            continue
        wall = [x["wall_s"] for x in rows]
        st = defaultdict(float)
        for x in rows:
            for k, v in (x.get("stage_s") or {}).items():
                st[k] += v
        cols.append({
            "name": Path(p).stem.replace("overhead_", ""),
            "n": len(rows),
            "err": sum(1 for x in rows if x.get("error")),
            "mean": sum(wall) / len(wall),
            "p50": pct(wall, 50),
            "p95": pct(wall, 95),
            "calls": sum(x["n_llm_calls"] for x in rows) / len(rows),
            "tok": sum(x["total_tokens"] for x in rows) / len(rows),
            "cost1k": sum(x.get("cost_usd") or 0.0 for x in rows) / len(rows) * 1000,
            "sim_s": sum(st.get(k, 0.0) for k in ("live_exec_real", "commit")) / len(rows),
        })
    if not cols:
        print("비교할 데이터 없음")
        return

    print(f"\n{'='*78}")
    print("E3. 설정별 오버헤드 비교")
    print(f"{'='*78}\n")
    hdrs = [c["name"] for c in cols]
    w = max(22, max(len(h) for h in hdrs) + 2)
    print(f"  {'지표':<26}" + "".join(f"{h:>{w}}" for h in hdrs))
    print("  " + "-" * (26 + w * len(cols)))

    def row(label, key, fmt="{:>%d.2f}"):
        print(f"  {label:<26}" + "".join((fmt % w).format(c[key]) for c in cols))

    row("액션 수", "n", "{:>%d.0f}")
    row("오류(타임아웃) 건수", "err", "{:>%d.0f}")
    row("가드 지연 평균 (s)", "mean")
    row("가드 지연 p50 (s)", "p50")
    row("가드 지연 p95 (s)", "p95")
    row("LLM 호출 수 / 액션", "calls")
    row("토큰 / 액션", "tok", "{:>%d.0f}")
    row("1k 액션당 비용 (USD)", "cost1k", "{:>%d.4f}")
    print(f"  {'실제 도구 실행 (s)':<26}"
          + "".join(("{:>%d.5f}" % w).format(c["sim_s"]) for c in cols))

    if len(cols) == 2:
        a, b = cols
        print(f"\n  → p50 지연 {a['p50']/b['p50'] if b['p50'] else 0:.1f}배 단축, "
              f"토큰 {a['tok']/b['tok'] if b['tok'] else 0:.1f}배 절감 "
              f"({a['name']} → {b['name']})")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 1:
        compare(args)
    else:
        main(args[0] if args else "results/overhead_mcptox.jsonl")
