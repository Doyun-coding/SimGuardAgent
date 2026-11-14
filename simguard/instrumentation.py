"""
SimGuard 오버헤드 계측 (E3)

목적: 액션 1건을 방어하는 데 드는 지연(latency)과 추론 비용(token)을
      단계별로 분해해 기록한다.

설계 원칙
  - 기본 비활성. SIMGUARD_PROFILE=1 일 때만 동작하며, 꺼져 있으면
    래핑도 하지 않으므로 런타임 경로에 영향이 없다.
  - LLM 호출은 전부 make_client() 가 만든 클라이언트를 지나가므로,
    그 하나만 프록시로 감싸면 모든 judge/simulator 호출이 잡힌다.
  - 단계 구분은 contextvar 로 전파한다. 호출부는 stage("...") 로 감싸기만
    하면 되고, 그 안에서 일어난 LLM 호출이 자동으로 해당 단계에 귀속된다.

사용:
    SIMGUARD_PROFILE=1 SIMGUARD_PROFILE_OUT=results/overhead.jsonl python ...

  코드에서:
    from simguard.instrumentation import action, stage
    with action(benchmark="mcptox", item_id="3"):
        with stage("shadow_exec"):
            ...
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
