"""
Weakness-02 파일럿 분석 — 기존 로그만 사용 (LLM 호출 0회, 재실행 0회)

E1. 시뮬레이션 충실도  : shadow_decision vs final_decision (agreement, Cohen's kappa)
E5. threshold 민감도   : risk_score + label -> ROC / AUROC / TPR@FPR
E6. baseline 비교      : 방어별 TPR/FPR 한 표
E4. benign utility     : rageval 파일의 utility 지표
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(r"C:\Users\inktr\Desktop\대학원\01.학부연구생\01.연구\SimGuardAgent\results")


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def load(f: Path):
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None, []
    if isinstance(d, list):
        return {}, d
    if not isinstance(d, dict):
        return None, []
    recs = d.get("results") or d.get("cases") or []
    return d, recs if isinstance(recs, list) else []


def cohens_kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    labels = set(ca) | set(cb)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def auroc(scores: list[float], labels: list[int]) -> float:
    """rank-based AUROC (tie 보정)"""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    paired = sorted(zip(scores, labels))
    ranks, i = {}, 0
    arr = [s for s, _ in paired]
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[j + 1] == arr[i]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks.setdefault(arr[k], r)
        i = j + 1
    rank_sum = sum(ranks[s] for s in pos)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def sweep(scores: list[float], labels: list[int], thresholds=None):
    """threshold -> (TPR, FPR)"""
    if thresholds is None:
        thresholds = [round(x * 0.05, 2) for x in range(0, 21)]
    P = sum(labels)
    N = len(labels) - P
    out = []
    for t in thresholds:
        tp = sum(1 for s, l in zip(scores, labels) if l == 1 and s >= t)
        fp = sum(1 for s, l in zip(scores, labels) if l == 0 and s >= t)
        out.append((t, tp / P if P else float("nan"), fp / N if N else float("nan")))
    return out


def tpr_at_fpr(scores, labels, target_fpr):
    best = None
    for t in sorted({round(s, 4) for s in scores} | {0.0, 1.01}):
        tp = sum(1 for s, l in zip(scores, labels) if l == 1 and s >= t)
        fp = sum(1 for s, l in zip(scores, labels) if l == 0 and s >= t)
        P = sum(labels); N = len(labels) - P
        if not P or not N:
            return float("nan"), float("nan")
        f = fp / N
        if f <= target_fpr:
            r = tp / P
            if best is None or r > best[0]:
                best = (r, t)
    return best if best else (float("nan"), float("nan"))


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def jaccard(a: str, b: str) -> float:
    ta, tb = set(norm_text(a).split()), set(norm_text(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def hdr(t):
    print(f"\n{'='*78}\n{t}\n{'='*78}")


# ─────────────────────────────────────────────────────────────────────────────
# E1. 시뮬레이션 충실도
# ─────────────────────────────────────────────────────────────────────────────

def e1_fidelity():
    hdr("E1. 시뮬레이션 충실도 — shadow(시뮬레이션) vs final(실제 실행)")
    rows = []
    for f in sorted(BASE.rglob("*.json")):
        _, recs = load(f)
        if not recs or not isinstance(recs[0], dict):
            continue
        usable = [r for r in recs
                  if r.get("shadow_decision") and r.get("final_decision") and not r.get("error")]
        if len(usable) < 20:
            continue
        pairs = [(r["shadow_decision"], r["final_decision"]) for r in usable]
        n = len(pairs)
        agree = sum(1 for a, b in pairs if a == b)
        cm = Counter(pairs)
        k = cohens_kappa(pairs)

        # risk score 상관 (Pearson)
        xs = [float(r.get("shadow_risk_score") or 0) for r in usable]
        ys = [float(r.get("final_risk_score") or 0) for r in usable]
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
        dy = math.sqrt(sum((b - my) ** 2 for b in ys))
        pear = num / (dx * dy) if dx and dy else float("nan")
        mae = sum(abs(a - b) for a, b in zip(xs, ys)) / n

        # 출력 유사도
        sims = [jaccard(r.get("shadow_simulated_output", ""),
                        r.get("final_simulated_output") or r.get("response", ""))
                for r in usable if r.get("shadow_simulated_output")]
        sim = sum(sims) / len(sims) if sims else float("nan")

        rows.append({
            "file": str(f.relative_to(BASE)),
            "n": n, "agree": agree / n, "kappa": k,
            "AA": cm[("ALLOW", "ALLOW")], "BB": cm[("BLOCK", "BLOCK")],
            "AB": cm[("ALLOW", "BLOCK")], "BA": cm[("BLOCK", "ALLOW")],
            "pearson": pear, "mae": mae, "jaccard": sim,
        })

    if not rows:
        print("  shadow/final 쌍을 가진 파일 없음")
        return rows

    rows.sort(key=lambda r: -r["n"])
    print(f"\n{'파일':<62}{'n':>6}{'일치':>8}{'kappa':>8}{'r':>7}{'MAE':>7}{'출력J':>7}")
    print("-" * 105)
    for r in rows:
        name = r["file"] if len(r["file"]) <= 60 else "..." + r["file"][-57:]
        print(f"{name:<62}{r['n']:>6}{r['agree']*100:>7.1f}%{r['kappa']:>8.3f}"
              f"{r['pearson']:>7.3f}{r['mae']:>7.3f}{r['jaccard']:>7.3f}")

    tot = sum(r["n"] for r in rows)
    print("-" * 105)
    print(f"{'가중 평균':<62}{tot:>6}"
          f"{sum(r['agree']*r['n'] for r in rows)/tot*100:>7.1f}%"
          f"{sum(r['kappa']*r['n'] for r in rows)/tot:>8.3f}")

    print("\n[혼동행렬 합계]  shadow \\ final")
    AA = sum(r["AA"] for r in rows); BB = sum(r["BB"] for r in rows)
    AB = sum(r["AB"] for r in rows); BA = sum(r["BA"] for r in rows)
    print(f"              final=ALLOW   final=BLOCK")
    print(f"  shadow=ALLOW   {AA:>8}      {AB:>8}   <- 시뮬레이션이 놓친 것")
    print(f"  shadow=BLOCK   {BA:>8}      {BB:>8}")
    print(f"\n  시뮬레이션 과탐(BA, 실제는 안전한데 시뮬레이션이 위험판정): {BA}건 "
          f"({BA/(AA+AB+BA+BB)*100:.2f}%)")
    print(f"  시뮬레이션 미탐(AB, 실제는 위험한데 시뮬레이션이 통과):     {AB}건 "
          f"({AB/(AA+AB+BA+BB)*100:.2f}%)")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# E5. threshold 민감도
# ─────────────────────────────────────────────────────────────────────────────

_SPLIT_RE = re.compile(r"__(attack|benign)(-mixed)?(_[a-z0-9]+)?$", re.IGNORECASE)


def _variant_key(f: Path) -> tuple[str, str] | None:
    """파일명을 (변종 식별자, split) 으로 분해. 변종이 섞이지 않도록 파일 쌍으로 묶는다."""
    stem = f.name
    for suf in (".rageval.json", ".harmbench.json", ".allow_only.harmbench.json", ".json"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    stem = stem.replace(".allow_only", "")
    m = _SPLIT_RE.search(stem)
    if not m:
        return None
    split = m.group(1).lower()
    variant = _SPLIT_RE.sub("", stem)
    return f"{f.parent.name}/{variant}", split


def _extract(recs):
    s, y = [], []
    for r in recs:
        if r.get("error"):
            continue
        rs, lb = r.get("risk_score"), r.get("label")
        if rs is None or lb is None:
            continue
        try:
            v = float(rs)
        except (TypeError, ValueError):
            continue
        s.append(v)
        y.append(1 if str(lb).lower() in ("attack", "1", "true", "harmful") else 0)
    return s, y


def _pool_for_roc():
    """동일 변종의 attack/benign 파일 쌍만 묶는다 (변종 간 혼합 금지)."""
    by_variant = defaultdict(lambda: {"s": [], "y": [], "files": []})
    for f in sorted(BASE.rglob("*.json")):
        if ".harmbench" in f.name or ".rageval" in f.name:
            continue
        meta, recs = load(f)
        if meta is None or not recs or not isinstance(recs[0], dict):
            continue
        if "risk_score" not in recs[0]:
            continue
        if str(meta.get("mode", "")).lower() in ("baseline", "none"):
            continue
        vk = _variant_key(f)
        if vk is None:
            continue
        variant, _ = vk
        s, y = _extract(recs)
        if not s:
            continue
        by_variant[variant]["s"] += s
        by_variant[variant]["y"] += y
        by_variant[variant]["files"].append(f.name)
    return by_variant


def e5_threshold():
    hdr("E5. threshold 민감도 — 재실행 없이 기존 risk_score로 계산")
    pools = _pool_for_roc()
    usable = {k: v for k, v in pools.items()
              if len(set(v["y"])) == 2 and len(v["y"]) >= 50}
    if not usable:
        print("  attack/benign 둘 다 있는 풀이 없음 — ROC 계산 불가")
        print(f"  (단일 라벨 풀 {len(pools)}개는 TPR만 산출 가능)")
        return {}

    for key, v in sorted(usable.items(), key=lambda kv: -len(kv[1]["y"])):
        s, y = v["s"], v["y"]
        P, N = sum(y), len(y) - sum(y)
        a = auroc(s, y)
        print(f"\n■ 변종: {key}")
        print(f"  n={len(y)} (attack={P}, benign={N})   AUROC={a:.4f}")
        t1, th1 = tpr_at_fpr(s, y, 0.01)
        t5, th5 = tpr_at_fpr(s, y, 0.05)
        print(f"  TPR@FPR<=1%  = {t1*100:>5.1f}%  (threshold={th1})")
        print(f"  TPR@FPR<=5%  = {t5*100:>5.1f}%  (threshold={th5})")
        print(f"\n  {'thr':>6}{'TPR':>9}{'FPR':>9}{'Youden J':>11}")
        best = None
        for t, tpr, fpr in sweep(s, y):
            j = tpr - fpr
            mark = "  <- 현재 배포값" if abs(t - 0.70) < 1e-9 else ""
            if best is None or j > best[1]:
                best = (t, j, tpr, fpr)
            if t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0):
                print(f"  {t:>6.2f}{tpr*100:>8.1f}%{fpr*100:>8.1f}%{j:>11.3f}{mark}")
        print(f"\n  최적(Youden J) threshold = {best[0]:.2f}  "
              f"(TPR={best[2]*100:.1f}%, FPR={best[3]*100:.1f}%, J={best[1]:.3f})")

        # 점수 분포 — 앵커링/floor 확인
        c = Counter(round(x, 2) for x in s)
        top = c.most_common(8)
        print(f"  점수 최빈값 8개: {', '.join(f'{k}({v})' for k, v in top)}")
        uniq = len(set(round(x, 2) for x in s))
        print(f"  고유 점수 개수: {uniq}  (연속적이면 클수록 정상, 작으면 앵커링/floor 영향)")
    return usable


# ─────────────────────────────────────────────────────────────────────────────
# E6. baseline 비교
# ─────────────────────────────────────────────────────────────────────────────

_DEFENSE_HINTS = [
    ("llamaguard", "LlamaGuard"), ("shieldgemma", "ShieldGemma"),
    ("granite", "GraniteGuardian"), ("ragdefender", "RAGDefender"),
    ("robustrag", "RobustRAG"), ("structured_prompting", "StructuredPrompting"),
    ("capability_policy", "CapabilityPolicy"), ("llm_metadata_guard", "LLMMetadataGuard"),
    ("rule_metadata_filter", "RuleMetadataFilter"),
    ("simguard", "SimGuard"), ("baseline", "None(baseline)"),
]


def _defense_of(path: Path, meta: dict) -> str:
    n = path.name.lower()
    for hint, label in _DEFENSE_HINTS:
        if hint in n:
            return label
    return str(meta.get("mode", "?"))


def e6_baselines():
    hdr("E6. baseline 비교 — 벤치마크별 방어 성능")
    table = defaultdict(dict)
    for f in sorted(BASE.rglob("*.json")):
        if ".harmbench" in f.name or ".rageval" in f.name:
            continue
        meta, recs = load(f)
        if meta is None or not recs or not isinstance(recs[0], dict):
            continue
        if "blocked" not in recs[0] and "simguard_decision" not in recs[0]:
            continue
        bench = meta.get("benchmark", meta.get("dataset", "?"))
        if bench == "?":
            continue
        ok = [r for r in recs if not r.get("error")]
        if len(ok) < 30:
            continue
        atk = [r for r in ok if str(r.get("label", "")).lower() in ("attack", "1")]
        ben = [r for r in ok if str(r.get("label", "")).lower() in ("benign", "0")]

        def blocked(r):
            if r.get("blocked") is True:
                return True
            return str(r.get("simguard_decision", "")).upper() == "BLOCK"

        d = _defense_of(f, meta)
        ent = table[bench].setdefault(d, {"tp": 0, "P": 0, "fp": 0, "N": 0})
        ent["tp"] += sum(1 for r in atk if blocked(r)); ent["P"] += len(atk)
        ent["fp"] += sum(1 for r in ben if blocked(r)); ent["N"] += len(ben)

    for bench, defs in sorted(table.items()):
        rows = [(d, v) for d, v in defs.items() if v["P"] or v["N"]]
        if not rows:
            continue
        print(f"\n■ {bench}")
        print(f"  {'방어':<22}{'TPR':>10}{'(n)':>8}{'FPR':>10}{'(n)':>8}")
        print("  " + "-" * 58)
        for d, v in sorted(rows, key=lambda x: -(x[1]["P"] + x[1]["N"])):
            tpr = f"{v['tp']/v['P']*100:>9.1f}%" if v["P"] else f"{'-':>10}"
            fpr = f"{v['fp']/v['N']*100:>9.1f}%" if v["N"] else f"{'-':>10}"
            print(f"  {d:<22}{tpr}{v['P']:>8}{fpr}{v['N']:>8}")


# ─────────────────────────────────────────────────────────────────────────────
# E4. benign utility
# ─────────────────────────────────────────────────────────────────────────────

def e4_utility():
    hdr("E4. Benign utility — rageval 로그의 유용성 지표")
    rows = []
    for f in sorted(BASE.rglob("*.rageval.json")):
        meta, recs = load(f)
        if meta is None:
            continue
        if meta.get("eval_utility_overall") is None and meta.get("eval_utility") is None:
            continue
        ok = [r for r in recs if not r.get("error")]
        blocked = sum(1 for r in ok if r.get("blocked") is True
                      or str(r.get("simguard_decision", "")).upper() == "BLOCK")
        rows.append({
            "file": str(f.relative_to(BASE)),
            "defense": _defense_of(f, meta),
            "split": meta.get("eval_split", "?"),
            "n": len(ok),
            "block_rate": blocked / len(ok) if ok else float("nan"),
            "utility": meta.get("eval_utility_overall", meta.get("eval_utility")),
            "useful": meta.get("eval_useful"),
        })
    if not rows:
        print("  utility 지표가 있는 rageval 파일 없음")
        return
    print(f"\n{'방어':<20}{'split':<16}{'n':>6}{'차단율':>9}{'utility':>10}{'useful':>9}")
    print("-" * 72)
    for r in sorted(rows, key=lambda r: (r["split"], r["defense"])):
        u = f"{r['utility']:.3f}" if isinstance(r["utility"], (int, float)) else "-"
        us = f"{r['useful']}" if r["useful"] is not None else "-"
        print(f"{r['defense']:<20}{str(r['split']):<16}{r['n']:>6}"
              f"{r['block_rate']*100:>8.1f}%{u:>10}{us:>9}")


if __name__ == "__main__":
    e1 = e1_fidelity()
    e5_threshold()
    e6_baselines()
    e4_utility()
    hdr("완료 — 위 수치는 전부 기존 로그 기반 (LLM 호출 0, 재실행 0)")
