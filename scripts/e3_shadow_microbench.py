"""
E3 보조 — shadow 실행 구간만 격리 측정 (LLM 호출 없음)

목적: "shadow 실행 0.1ms"가 어느 기질(substrate)에서, 정확히 무엇을 포함한
      수치인지 확정한다. 기질마다 격리 수단이 달라 비용이 크게 다르므로
      단일 숫자로 일반화할 수 없다는 점을 수치로 보인다.

측정 대상
  1. ShadowMCPExecutor        — MCPTox. 순수 in-memory mock (네트워크/DB/컨테이너 없음)
  2. RuntimeShadowToolExecutor — AgentDojo 방식. Pydantic deep copy + model_dump + DeepDiff
                                  (합성 env, 레코드 수를 바꿔가며 스케일링 확인)

사용:
  python scripts/e3_shadow_microbench.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "attacks-bench" / "mcptox"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

REPEATS = 300


def timeit(fn, repeats=REPEATS):
    """워밍업 후 repeats회 측정. 중앙값/평균/p95를 초 단위로 돌려준다."""
    for _ in range(20):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return {
        "mean": statistics.fmean(samples),
        "p50": samples[len(samples) // 2],
        "p95": samples[int(len(samples) * 0.95)],
        "min": samples[0],
    }


def fmt(s: float) -> str:
    if s < 1e-3:
        return f"{s*1e6:8.1f} us"
    if s < 1.0:
        return f"{s*1e3:8.3f} ms"
    return f"{s:8.3f} s "


# ─────────────────────────────────────────────────────────────────────────────
# 1. ShadowMCPExecutor — MCPTox 실제 구현
# ─────────────────────────────────────────────────────────────────────────────

def bench_mcp():
    from run_mcptox_simguard import ShadowMCPExecutor, load_cases

    data = _ROOT / "attacks-bench" / "MCPTox-Benchmark" / "response_all.json"
    cases = load_cases(data, "attack")[:10]

    rows = []
    for case in cases[:5]:
        ex = ShadowMCPExecutor(case["query"], case["system"])
        args = {"path": "/home/user/report.txt", "content": "x" * 200}
        r = timeit(lambda: ex.shadow_execute("write_file", args))
        rows.append(r)

    agg = {k: statistics.fmean(r[k] for r in rows) for k in rows[0]}
    return agg, len(cases)


# ─────────────────────────────────────────────────────────────────────────────
# 2. RuntimeShadowToolExecutor — AgentDojo 방식 (합성 env)
# ─────────────────────────────────────────────────────────────────────────────

def bench_runtime(n_records: int):
    from pydantic import BaseModel
    from simguard.shadow_api import RuntimeShadowToolExecutor

    class Email(BaseModel):
        id: str
        sender: str
        subject: str
        body: str
        read: bool = False

    class Env(BaseModel):
        inbox: list[Email] = []
        files: dict[str, str] = {}

    env = Env(
        inbox=[
            Email(id=str(i), sender=f"user{i}@corp.com",
                  subject=f"Q{i%4} report draft",
                  body="Please review the attached figures. " * 8)
            for i in range(n_records)
        ],
        files={f"/drive/doc{i}.md": "content " * 40 for i in range(n_records // 2)},
    )

    class Runtime:
        @staticmethod
        def run_function(e, name, args):
            e.inbox.append(Email(id="new", sender="a@b.c",
                                 subject=args.get("subject", ""),
                                 body=args.get("body", "")))
            return {"ok": True}, None

    ex = RuntimeShadowToolExecutor(Runtime(), env, result_formatter=lambda r: json.dumps(r, default=str))
    args = {"subject": "status", "body": "done"}
    return timeit(lambda: ex.shadow_execute("send_email", args), repeats=60)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("E3 보조. shadow 실행 구간 격리 측정 (LLM 호출 없음)")
    print("=" * 78)
    print(f"\n측정 구간: shadow_execute() 진입 ~ 반환")
    print(f"  포함: 상태 스냅샷, 도구 디스패치, 상태 diff, 결과 직렬화")
    print(f"  제외: executor 생성, LLM 판정, commit")

    print(f"\n{'─'*78}")
    print("1. ShadowMCPExecutor — MCPTox (순수 in-memory mock)")
    print(f"{'─'*78}")
    try:
        agg, n = bench_mcp()
        print(f"  실제 MCPTox 케이스 {n}건 기준, 각 {REPEATS}회 반복")
        print(f"    p50 {fmt(agg['p50'])}   평균 {fmt(agg['mean'])}   "
              f"p95 {fmt(agg['p95'])}   최소 {fmt(agg['min'])}")
        print(f"\n  내용: 툴 이름 문자열 매칭 -> 파이썬 리스트 append -> 얕은 복사 -> json.dumps")
        print(f"        네트워크/파일시스템/DB/컨테이너 접근 없음")
    except Exception as e:
        print(f"  측정 실패: {type(e).__name__}: {e}")

    print(f"\n{'─'*78}")
    print("2. RuntimeShadowToolExecutor — AgentDojo 방식 (Pydantic deep copy + DeepDiff)")
    print(f"{'─'*78}")
    print(f"  {'env 레코드 수':>14}{'p50':>14}{'평균':>14}{'p95':>14}")
    print("  " + "-" * 56)
    try:
        for n in (10, 40, 100, 200):
            r = bench_runtime(n)
            print(f"  {n:>14}{fmt(r['p50']):>14}{fmt(r['mean']):>14}{fmt(r['p95']):>14}")
        print(f"\n  내용: model_copy(deep=True) -> run_function -> model_dump x2 -> DeepDiff")
    except Exception as e:
        print(f"  측정 실패: {type(e).__name__}: {e}")

    print(f"\n{'='*78}")
    print("결론")
    print(f"{'='*78}")
    print("  shadow 실행 비용은 기질마다 자릿수가 다르다. 단일 숫자로 일반화 불가.")
    print("    MCP in-memory mock   : 마이크로초대")
    print("    Pydantic env 복제     : 레코드 수에 비례, 밀리초대")
    print("    DB 트랜잭션 롤백      : 실제 DB 왕복 (미측정)")
    print("    Docker 컨테이너       : 초 단위 (미측정)")


if __name__ == "__main__":
    main()
