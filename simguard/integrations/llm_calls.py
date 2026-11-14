"""
범용 연결부: OpenAI SDK와 LiteLLM 호출.

전용 연결부가 없는 에이전트 프레임워크도 대부분 OpenAI SDK나 LiteLLM으로 LLM을 호출한다.
protect()를 한 번 부르면 이 프로세스의 모든 호출에서 LLM 출력이 에이전트에게 돌아가기
전에 SimGuardAgent Interceptor가 판정한다. 타겟 에이전트에 프롬프트나 도구를 추가하지 않는다.

  import simguard
  simguard.protect()          # 에이전트 프레임워크를 import하기 전에 부른다

  ...에이전트 실행...

BLOCK이면 출력을 차단 문구로 바꾸고 도구 호출을 지운다. on_block="raise"이면 SimGuardBlocked를
일으킨다. 스트리밍 응답은 끝까지 받아 판정한 뒤 내보낸다.

도구 결과는 shadow가 있으면 shadow에서 실제로 실행한 결과, 없으면 ToolCallSimulator가 예상한 결과다.
shadow는 protect(database=..., databases=..., shadow_executors=...)로 넘긴다. 넘기지 않으면 내부 모델이
결과를 예상한다. 실제 도구로 shadow 실행하는 전용 연결부도 있다 (예: integrations.agentdojo).
지원하지 않는 호출: OpenAI Responses API, Anthropic SDK.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import functools
import json
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from typing import Any

from simguard.interceptor import BLOCKED_MESSAGE, SimGuardAgentInterceptor
from simguard.models import Decision, InterceptionResult

# SimGuardAgent 자신의 LLM 호출(Simulator, Judge)과 LiteLLM 내부의 OpenAI SDK 호출은 판정하지 않는다.
_BYPASS: ContextVar[bool] = ContextVar("simguard_bypass", default=False)
_settings: dict[str, Any] = {
    "on_block": "replace", "on_result": None, "target_model": None,
    "execution_environment": None, "shadow_executors": None, "database": None, "databases": None,
}


def _tool_calls(output: dict) -> list[tuple[str, Any]]:
    calls = []
    for call in output.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = None
        if function.get("name"):
            calls.append((function["name"], arguments))
    return calls


def _database_executor(database: Any) -> Any:
    if callable(getattr(database, "shadow_execute", None)):
        return database
    from simguard.shadow_db import DatabaseShadowExecutor

    return DatabaseShadowExecutor(database, copy=True)


def _shadow_executors(output: dict) -> dict | None:
    """이 출력의 도구 호출에 쓸 shadow. 우선순위: shadow_executors > databases[도구] > database(SQL 호출)."""
    executors = dict(_settings["shadow_executors"] or {})
    for name, executor in (_settings["databases"] or {}).items():
        executors.setdefault(name, executor)
    if _settings["database"] is not None:
        from simguard.shadow_db import is_sql_call

        for name, arguments in _tool_calls(output):
            if name not in executors and is_sql_call(arguments):
                executors[name] = _settings["database"]
    return executors or None


def _same_model(requested: Any, target: str) -> bool:
    """같은 모델의 다른 표기(openrouter/qwen/x, qwen/x)도 같은 모델로 본다."""
    if not isinstance(requested, str):
        return False
    requested, target = requested.strip().lower(), target.strip().lower()
    return requested.endswith(target) or target.endswith(requested)


def _is_target(kwargs: dict) -> bool:
    target = _settings["target_model"]
    return target is None or _same_model(kwargs.get("model"), target)
_originals: dict[tuple[Any, str], Any] = {}


class SimGuardBlocked(Exception):
    """on_block="raise"일 때 BLOCK된 출력 대신 일어난다."""

    def __init__(self, result: InterceptionResult) -> None:
        super().__init__(BLOCKED_MESSAGE)
        self.result = result


def _plain(value: Any) -> Any:
    """SDK 객체를 판정에 넘길 dict와 list로 바꾼다."""
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _request(kwargs: dict, args: tuple = (), messages_position: int | None = None) -> tuple[list, list | None]:
    messages = kwargs.get("messages")
    if messages is None and messages_position is not None and len(args) > messages_position:
        messages = args[messages_position]
    tools = _plain(kwargs.get("tools"))
    return list(_plain(messages) or []), tools if isinstance(tools, list) and tools else None


async def _judge(messages: list, tools: list | None, output: dict) -> InterceptionResult:
    token = _BYPASS.set(True)
    environment = _settings["execution_environment"]
    try:
        # 판정기의 LLM 클라이언트는 이벤트 루프에 묶이므로 호출마다 새로 만든다.
        result = await SimGuardAgentInterceptor().intercept_output(
            messages, output, tools,
            shadow_executors=_shadow_executors(output),
            execution_session_factory=environment.factory(messages) if environment is not None else None,
        )
        if environment is not None:
            environment.record(messages, result)
        return result
    finally:
        _BYPASS.reset(token)


def _judge_sync(messages: list, tools: list | None, output: dict) -> InterceptionResult:
    coro = _judge(messages, tools, output)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # 이벤트 루프 안에서 동기 호출이 온 경우 다른 스레드에서 판정한다.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _report(results: list[InterceptionResult]) -> None:
    callback = _settings["on_result"]
    if callback is not None:
        for result in results:
            callback(result)
    blocked = next((result for result in results if result.decision == Decision.BLOCK), None)
    if blocked is not None and _settings["on_block"] == "raise":
        raise SimGuardBlocked(blocked)


def _block_message(message: Any) -> None:
    message.content = BLOCKED_MESSAGE
    for name in ("tool_calls", "function_call"):
        if getattr(message, name, None) is not None:
            setattr(message, name, None)


def _apply(response: Any, results: list[InterceptionResult]) -> Any:
    _report(results)
    for choice, result in zip(response.choices, results):
        if result.decision == Decision.BLOCK:
            _block_message(choice.message)
            choice.finish_reason = "stop"
    return response


def _outputs(response: Any) -> list[dict]:
    outputs = []
    for choice in response.choices:
        output = _plain(choice.message)
        outputs.append(output if isinstance(output, dict) else {"content": str(output)})
    return outputs


# ---------------------------------------------------------------- streaming

def _snapshot(chunk: Any) -> Any:
    # 스트림 구현이 청크 객체를 재사용해도 받은 그대로 다시 내보낼 수 있게 복사한다.
    return copy.deepcopy(chunk)


def _get(value: Any, name: str) -> Any:
    """SDK 객체와 dict를 같은 방식으로 읽는다. LiteLLM 스트림은 delta가 dict일 수 있다."""
    if isinstance(value, Mapping):
        return value.get(name)
    found = getattr(value, name, None)
    if found is None:
        found = (getattr(value, "model_extra", None) or {}).get(name)
    return found


def _set(value: Any, name: str, item: Any) -> None:
    if isinstance(value, dict):
        value[name] = item
    else:
        setattr(value, name, item)


def _merge(chunks: list) -> dict[int, dict]:
    """스트림 청크를 선택지별 출력 메시지로 합친다."""
    merged: dict[int, dict] = {}
    for chunk in chunks:
        for choice in _get(chunk, "choices") or []:
            slot = merged.setdefault(_get(choice, "index") or 0, {"content": "", "reasoning": "", "calls": {}})
            delta = _get(choice, "delta")
            if delta is None:
                continue
            slot["content"] += _get(delta, "content") or ""
            reasoning = _get(delta, "reasoning_content") or _get(delta, "reasoning")
            if isinstance(reasoning, str):
                slot["reasoning"] += reasoning
            for call in _get(delta, "tool_calls") or []:
                entry = slot["calls"].setdefault(
                    _get(call, "index") or 0,
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                entry["id"] = _get(call, "id") or entry["id"]
                function = _get(call, "function")
                if function is not None:
                    entry["function"]["name"] += _get(function, "name") or ""
                    entry["function"]["arguments"] += _get(function, "arguments") or ""
    outputs = {}
    for index, slot in merged.items():
        output: dict = {"role": "assistant", "content": slot["content"]}
        if slot["calls"]:
            output["tool_calls"] = [slot["calls"][key] for key in sorted(slot["calls"])]
        if slot["reasoning"]:
            output["reasoning"] = slot["reasoning"]
        outputs[index] = output
    return outputs


def _replace_blocked(chunks: list, blocked: set[int]) -> list:
    """BLOCK된 선택지는 차단 문구 한 번으로 바꾸고 나머지 내용은 내보내지 않는다."""
    if not blocked:
        return chunks
    emitted: set[int] = set()
    kept_chunks = []
    for chunk in chunks:
        choices = _get(chunk, "choices") or []
        if not choices:
            kept_chunks.append(chunk)
            continue
        kept = []
        for choice in choices:
            index = _get(choice, "index") or 0
            if index not in blocked:
                kept.append(choice)
            elif index not in emitted:
                emitted.add(index)
                delta = _get(choice, "delta")
                _set(delta, "content", BLOCKED_MESSAGE)
                if _get(delta, "tool_calls") is not None:
                    _set(delta, "tool_calls", None)
                _set(choice, "finish_reason", "stop")
                kept.append(choice)
        if kept:
            _set(chunk, "choices", kept)
            kept_chunks.append(chunk)
    return kept_chunks


def _decide_stream(chunks: list, outputs: dict[int, dict], results: list[InterceptionResult]) -> list:
    _report(results)
    blocked = {index for index, result in zip(outputs, results) if result.decision == Decision.BLOCK}
    return _replace_blocked(chunks, blocked)


class _GuardedStream:
    """동기 스트림. 처음 읽을 때 끝까지 받아 판정한다."""

    def __init__(self, stream: Any, messages: list, tools: list | None) -> None:
        self._stream, self._messages, self._tools = stream, messages, tools
        self._iterator = None

    def _ready(self):
        if self._iterator is None:
            chunks = [_snapshot(chunk) for chunk in self._stream]
            outputs = _merge(chunks)
            results = [_judge_sync(self._messages, self._tools, output) for output in outputs.values()]
            self._iterator = iter(_decide_stream(chunks, outputs, results))
        return self._iterator

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._ready())

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        close = getattr(self._stream, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _GuardedAsyncStream:
    """비동기 스트림. 처음 읽을 때 끝까지 받아 판정한다."""

    def __init__(self, stream: Any, messages: list, tools: list | None) -> None:
        self._stream, self._messages, self._tools = stream, messages, tools
        self._iterator = None

    async def _ready(self):
        if self._iterator is None:
            chunks = [_snapshot(chunk) async for chunk in self._stream]
            outputs = _merge(chunks)
            results = [await _judge(self._messages, self._tools, output) for output in outputs.values()]
            self._iterator = iter(_decide_stream(chunks, outputs, results))
        return self._iterator

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(await self._ready())
        except StopIteration:
            raise StopAsyncIteration from None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    async def close(self) -> None:
        close = getattr(self._stream, "close", None) or getattr(self._stream, "aclose", None)
        if callable(close):
            outcome = close()
            if asyncio.iscoroutine(outcome):
                await outcome

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


# ---------------------------------------------------------------- wrappers

def _guard_sync(original: Callable, messages_position: int | None = None, bypass_inner: bool = False) -> Callable:
    @functools.wraps(original)
    def guarded(*args, **kwargs):
        if _BYPASS.get() or not _is_target(kwargs):
            return original(*args, **kwargs)
        messages, tools = _request(kwargs, args, messages_position)
        token = _BYPASS.set(True) if bypass_inner else None
        try:
            response = original(*args, **kwargs)
        finally:
            if token is not None:
                _BYPASS.reset(token)
        if kwargs.get("stream"):
            return _GuardedStream(response, messages, tools)
        return _apply(response, [_judge_sync(messages, tools, output) for output in _outputs(response)])

    guarded.__simguard_original__ = original
    return guarded


def _guard_async(original: Callable, messages_position: int | None = None, bypass_inner: bool = False) -> Callable:
    @functools.wraps(original)
    async def guarded(*args, **kwargs):
        if _BYPASS.get() or not _is_target(kwargs):
            return await original(*args, **kwargs)
        messages, tools = _request(kwargs, args, messages_position)
        token = _BYPASS.set(True) if bypass_inner else None
        try:
            response = await original(*args, **kwargs)
        finally:
            if token is not None:
                _BYPASS.reset(token)
        if kwargs.get("stream"):
            return _GuardedAsyncStream(response, messages, tools)
        return _apply(response, [await _judge(messages, tools, output) for output in _outputs(response)])

    guarded.__simguard_original__ = original
    return guarded


def _patch(owner: Any, name: str, wrap: Callable[[Callable], Callable]) -> None:
    if (owner, name) in _originals:
        return
    original = getattr(owner, name)
    _originals[(owner, name)] = original
    setattr(owner, name, wrap(original))


def protect(
    on_block: str = "replace",
    on_result: Callable[[InterceptionResult], None] | None = None,
    litellm: bool = True,
    target_model: str | None = None,
    execution_environment: Any = None,
    shadow_executors: Mapping[str, Any] | None = None,
    database: Any = None,
    databases: Mapping[str, Any] | None = None,
    live_reads: bool = True,
    live_read_hosts: Any = None,
    live_read_ports: Any = (80, 443),
) -> None:
    """이 프로세스의 OpenAI SDK와 LiteLLM 호출에 SimGuardAgent Interceptor를 연결한다.

    on_block: "replace"는 출력을 차단 문구로 바꾸고, "raise"는 SimGuardBlocked를 일으킨다.
    on_result: 판정마다 InterceptionResult를 받는 함수 (기록용).
    target_model: 지정하면 이 모델로 가는 호출만 판정한다. 지정하지 않으면 모든 호출을 판정한다.
    execution_environment: 에이전트가 코드를 실행하는 작업 파일 (simguard.ExecutionEnvironment).
    shadow_executors: {도구 이름: shadow 실행기}. 여기 있는 도구는 LLM 예상 대신 shadow 환경에서
        실제로 실행한 결과로 판정한다 (simguard.DatabaseShadowExecutor, simguard.MCPShadowExecutor,
        또는 shadow_execute를 구현한 객체). 없는 도구는 내부 모델이 결과를 예상한다.
    database: 에이전트가 쓰는 DB. 인자에 SQL 문이 있는 도구 호출은 이 DB에서 실행한 결과로 판정한다.
        SQLite 경로, SQLAlchemy URL이나 Engine, 또는 DB 연결을 만드는 함수(lambda: psycopg.connect(DSN)).
        SQLite는 판정 순간의 상태를 복사해 복사본에서, 그 외 DB는 트랜잭션 안에서 실행하고 되돌린다.
    databases: DB가 여럿일 때 {도구 이름: DB}. 값은 database와 같은 형식이다.
    live_reads: True면 이 프로세스에 있는 실제 도구를 GET 전용 모드로 실행해 실제 결과로 판정한다.
        GET·HEAD가 아닌 HTTP 요청, 파일 쓰기, 프로세스 실행, HTTP가 아닌 포트 연결이 나오면 멈추고
        내부 모델이 결과를 예상한다 (simguard.shadow_http).
    live_read_hosts: GET을 보내도 되는 도메인 목록. None이면 모든 도메인.
    live_read_ports: GET 전용 실행에서 연결해도 되는 포트.
    여러 번 불러도 한 번만 연결되고 설정만 바뀐다.
    """
    if on_block not in {"replace", "raise"}:
        raise ValueError('on_block must be "replace" or "raise"')
    for name, executor in (shadow_executors or {}).items():
        if not callable(getattr(executor, "shadow_execute", None)):
            raise TypeError(f"shadow executor for {name!r} has no shadow_execute(tool_name, tool_arguments)")
    from simguard.shadow_http import configure_live_reads

    configure_live_reads(live_reads, live_read_hosts, live_read_ports)
    _settings.update(on_block=on_block, on_result=on_result, target_model=target_model,
                     execution_environment=execution_environment,
                     shadow_executors=dict(shadow_executors) if shadow_executors else None,
                     database=_database_executor(database) if database is not None else None,
                     databases={name: _database_executor(db) for name, db in databases.items()} if databases else None)

    from openai.resources.chat.completions import AsyncCompletions, Completions

    _patch(Completions, "create", _guard_sync)
    _patch(AsyncCompletions, "create", _guard_async)

    from simguard.tool_discovery import track_mcp

    # 에이전트가 MCP 서버에서 받아 가는 도구를 기록해 실제 도구를 찾을 수 있게 한다.
    track_mcp()

    if litellm:
        try:
            import litellm as litellm_module
        except ImportError:
            return
        # LiteLLM은 내부에서 OpenAI SDK를 부를 수 있으므로 그 안쪽 호출은 다시 판정하지 않는다.
        _patch(litellm_module, "completion", lambda original: _guard_sync(original, 1, bypass_inner=True))
        _patch(litellm_module, "acompletion", lambda original: _guard_async(original, 1, bypass_inner=True))


def unprotect() -> None:
    """protect()로 연결한 것을 모두 되돌린다."""
    for (owner, name), original in list(_originals.items()):
        setattr(owner, name, original)
    _originals.clear()
    from simguard.tool_discovery import untrack_mcp

    untrack_mcp()
