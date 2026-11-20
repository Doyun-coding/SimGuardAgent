"""
실험용: SimGuardAgent 오버헤드 계측 (E3). 배포 패키지(simguard/)에는 들어가지 않는다.

행동 1건을 판정하는 데 든 지연과 LLM 호출 수·토큰·비용을 단계별로 기록한다.
simguard 코드는 고치지 않고 바깥에서 감싼다 (install()).
  - 내부 모델 클라이언트(simguard.llm_backends.make_backend_client)를 프록시로 감싸 모든 호출을 기록한다.
  - 시뮬레이터·Judge·실행기 메서드를 감싸 단계(predict_result, predict_next_action, judge, live_exec_real)를 나눈다.

사용 (SIMGUARD_PROFILE=1 일 때만 동작):
    SIMGUARD_PROFILE=1 SIMGUARD_PROFILE_OUT=results/overhead.jsonl python ...

    from scripts.simguard_profiling import install, action
    install()                                     # SimGuardAgentInterceptor를 만들기 전에 부른다
    with action(benchmark="mcptox", item_id="3"):
        await SimGuardAgentInterceptor().intercept_output(...)

집계: python scripts/e3_overhead_report.py results/overhead.jsonl
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
import threading
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 활성화 스위치
# ─────────────────────────────────────────────────────────────────────────────

def enabled() -> bool:
    return os.getenv("SIMGUARD_PROFILE", "").strip().lower() not in ("", "0", "false", "no")


def _out_path() -> Path:
    p = os.getenv("SIMGUARD_PROFILE_OUT", "results/overhead.jsonl")
    return Path(p)


# ─────────────────────────────────────────────────────────────────────────────
# 컨텍스트
# ─────────────────────────────────────────────────────────────────────────────

_stage: contextvars.ContextVar[str] = contextvars.ContextVar("simguard_stage", default="other")
_calls: contextvars.ContextVar[list | None] = contextvars.ContextVar("simguard_calls", default=None)
_stages: contextvars.ContextVar[dict | None] = contextvars.ContextVar("simguard_stages", default=None)

_write_lock = threading.Lock()


@contextlib.contextmanager
def action(**meta):
    """액션 1건의 계측 범위. 종료 시 JSONL 한 줄을 append 한다."""
    if not enabled():
        yield None
        return
    calls: list[dict] = []
    stages: dict[str, float] = {}
    tok_c = _calls.set(calls)
    tok_s = _stages.set(stages)
    t0 = time.perf_counter()
    err = None
    try:
        yield calls
    except Exception as e:  # 실패한 액션도 비용은 발생했으므로 기록한다
        err = f"{type(e).__name__}: {e}"[:200]
        raise
    finally:
        wall = time.perf_counter() - t0
        _calls.reset(tok_c)
        _stages.reset(tok_s)
        rec = {
            **meta,
            "wall_s": round(wall, 4),
            "n_llm_calls": len(calls),
            "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in calls),
            "completion_tokens": sum(c.get("completion_tokens") or 0 for c in calls),
            "total_tokens": sum(c.get("total_tokens") or 0 for c in calls),
            "cost_usd": sum(c.get("cost_usd") or 0.0 for c in calls),
            "llm_s": round(sum(c.get("latency_s", 0.0) for c in calls), 4),
            # 비-LLM 단계는 마이크로초대라 소수 4자리로 반올림하면 최소 논제로 값에
            # 뭉개진다. 단계 시간은 6자리까지 남긴다.
            "stage_s": {k: round(v, 6) for k, v in stages.items()},
            "calls": calls,
            "error": err,
        }
        _append(rec)


@contextlib.contextmanager
def stage(name: str):
    """단계 구분 + 벽시계 시간 누적."""
    if not enabled():
        yield
        return
    tok = _stage.set(name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        _stage.reset(tok)
        st = _stages.get()
        if st is not None:
            st[name] = st.get(name, 0.0) + dt


def _append(rec: dict) -> None:
    path = _out_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with _write_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        pass  # 계측 실패가 실험을 죽이지 않는다


def _record_call(model, latency_s, usage, ok, stage_name=None):
    calls = _calls.get()
    if calls is None:
        return
    pt = ct = tt = cost = reasoning = None
    if usage is not None:
        pt = getattr(usage, "prompt_tokens", None)
        ct = getattr(usage, "completion_tokens", None)
        tt = getattr(usage, "total_tokens", None)
        # OpenRouter 계열은 usage.cost 로 실제 과금액을 돌려준다(표준 필드가
        # 아니므로 model_extra 에 들어갈 수 있다).
        extra = getattr(usage, "model_extra", None) or {}
        cost = getattr(usage, "cost", None) or extra.get("cost")
        details = getattr(usage, "completion_tokens_details", None) or extra.get(
            "completion_tokens_details"
        )
        if details is not None:
            reasoning = (
                getattr(details, "reasoning_tokens", None)
                if not isinstance(details, dict)
                else details.get("reasoning_tokens")
            )
    calls.append({
        "stage": stage_name or _stage.get(),
        "model": model,
        "latency_s": round(latency_s, 4),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
        "reasoning_tokens": reasoning,
        "cost_usd": cost,
        "ok": ok,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 클라이언트 프록시 — chat.completions.create 만 가로챈다
# ─────────────────────────────────────────────────────────────────────────────

class _CompletionsProxy:
    def __init__(self, inner):
        self._inner = inner

    async def create(self, **kwargs):
        t0 = time.perf_counter()
        try:
            resp = await self._inner.create(**kwargs)
        except Exception:
            _record_call(kwargs.get("model"), time.perf_counter() - t0, None, False)
            raise
        _record_call(
            kwargs.get("model"),
            time.perf_counter() - t0,
            getattr(resp, "usage", None),
            True,
        )
        return resp

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _ChatProxy:
    def __init__(self, inner):
        self._inner = inner
        self.completions = _CompletionsProxy(inner.completions)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _ClientProxy:
    def __init__(self, inner):
        self._inner = inner
        self.chat = _ChatProxy(inner.chat)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def instrument(client):
    """계측이 켜져 있을 때만 프록시로 감싼다."""
    if not enabled():
        return client
    try:
        return _ClientProxy(client)
    except Exception:
        return client


# ─────────────────────────────────────────────────────────────────────────────
# simguard 바깥에서 감싸기
# ─────────────────────────────────────────────────────────────────────────────

# (모듈, 클래스, 메서드, 단계 이름)
_STAGES = (
    ("simguard.simulation.backends.tool_call_simulator", "ToolCallSimulator", "_predict_tool_result", "predict_result"),
    ("simguard.simulation.backends.tool_call_simulator", "ToolCallSimulator", "_predict_next_action", "predict_next_action"),
    ("simguard.simulation.backends.behavior_simulator", "BehaviorSimulator", "_simulate_plan", "predict_result"),
    ("simguard.simulation.backends.execution_simulator", "ExecutionSimulator", "_predict", "predict_result"),
    ("simguard.simulation.backends.execution_simulator", "ExecutionSimulator", "_predict_next_action", "predict_next_action"),
    ("simguard.simulation.judges.tool_call_judge", "ToolCallJudge", "judge", "judge"),
    ("simguard.simulation.judges.behavior_judge", "BehaviorJudge", "judge", "judge"),
    ("simguard.simulation.judges.execution_judge", "ExecutionJudge", "judge", "judge"),
    ("simguard.live_db", "DatabaseExecutor", "live_execute", "live_exec_real"),
    ("simguard.internal_database", "InternalDatabase", "live_execute", "live_exec_real"),
    ("simguard.live_http", "LiveReadExecutor", "live_execute", "live_exec_real"),
)
_installed = False


def _with_stage(method, name):
    import functools
    import inspect

    if inspect.iscoroutinefunction(method):
        @functools.wraps(method)
        async def wrapped(*args, **kwargs):
            with stage(name):
                return await method(*args, **kwargs)
    else:
        @functools.wraps(method)
        def wrapped(*args, **kwargs):
            with stage(name):
                return method(*args, **kwargs)
    return wrapped


def install() -> None:
    """SIMGUARD_PROFILE=1이면 simguard의 내부 모델 클라이언트와 단계 메서드를 감싼다. 여러 번 불러도 한 번만 감싼다."""
    global _installed
    if _installed or not enabled():
        return
    import importlib

    import simguard.llm_backends as backends

    make = backends.make_backend_client
    backends.make_backend_client = lambda role=None: instrument(make(role))
    for module, cls, method, name in _STAGES:
        owner = getattr(importlib.import_module(module), cls)
        setattr(owner, method, _with_stage(getattr(owner, method), name))
    _installed = True
