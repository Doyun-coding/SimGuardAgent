"""
범용 연결부: 타겟 모델 호출을 가로챈다.

전용 연결부가 없는 에이전트도 결국 어떤 SDK나 로컬 모델로 LLM을 부른다. protect()를 한 번 부르면
이 프로세스의 그 호출에서 LLM 출력이 에이전트에게 돌아가기 전에 SimGuardAgent Interceptor가 판정한다.
타겟 에이전트에 프롬프트나 도구를 추가하지 않는다.

  import simguard
  simguard.protect()                                   # 설치된 방법을 모두 연결 (providers="auto")
  simguard.protect(providers=["anthropic", "vllm"])    # 쓰는 방법만
  simguard.protect(llm_calls=[simguard.LLMCall(...)])  # 목록에 없는 방법

  ...에이전트 실행...

가로채는 방법 (simguard.integrations.llm_providers)
  openai        OpenAI SDK. OpenAI 호환 주소면 모두 (OpenAI, OpenRouter, LiteLLM 프록시, GPU 서버의 vLLM·SGLang·TGI, Ollama /v1)
  litellm       litellm.completion / acompletion
  anthropic     Anthropic SDK messages.create
  ollama        ollama.chat, Client.chat, AsyncClient.chat
  vllm          vllm.LLM.chat (이 프로세스의 GPU)
  transformers  transformers text-generation pipeline (이 프로세스의 GPU)
아직 import되지 않은 모듈은 import되는 순간 연결한다. 로컬 모델이 도구 호출을 글로 내면
(<tool_call>{...}</tool_call> 등) 도구 호출로 읽어 판정한다.

BLOCK이면 출력을 차단 문구로 바꾸고 도구 호출을 지운다. on_block="raise"이면 SimGuardBlocked를
일으킨다. 스트리밍 응답은 끝까지 받아 판정한 뒤 내보낸다.

도구 결과는 판정 전에 실제로 실행할 수 있으면 그 결과, 없으면 ToolCallSimulator가 예상한 결과다.
  실제 도구  protect(tools=...)로 알려 주거나 자동으로 찾은 도구를 GET 전용 모드로 실행 (live_reads)
  SQL 호출   protect(database=..., databases=...)로 알려 준 에이전트 DB의 복사본·롤백 트랜잭션에서 실행.
             DB를 알려 주지 않으면 SimGuardAgent 내부 SQLite에서 실행
그 외 호출과 실행을 멈춘 호출(GET이 아닌 요청, 파일 쓰기 등)은 내부 모델이 결과를 예상한다.
지원하지 않는 호출: OpenAI Responses API, Anthropic messages.stream() 도우미 (LLMCall로 알려 줄 수 있다).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import copy
import functools
import importlib.abc
import importlib.util
import inspect
import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from typing import Any

from simguard.integrations.llm_providers import PROVIDERS, LLMCall
from simguard.interceptor import BLOCKED_MESSAGE, SimGuardAgentInterceptor
from simguard.models import Decision, InterceptionResult

# SimGuardAgent 자신의 LLM 호출(Simulator, Judge)과 SDK 안쪽에서 다시 일어나는 호출은 판정하지 않는다.
_BYPASS: ContextVar[bool] = ContextVar("simguard_bypass", default=False)
_settings: dict[str, Any] = {
    "on_block": "replace", "on_result": None, "target_model": None,
    "database": None, "databases": None, "execution_environment": None,
}


@contextlib.contextmanager
def bypass():
    """이 안에서 일어나는 LLM 호출은 판정하지 않는다."""
    token = _BYPASS.set(True)
    try:
        yield
    finally:
        _BYPASS.reset(token)


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
    if callable(getattr(database, "live_execute", None)):
        return database
    from simguard.live_db import DatabaseExecutor

    return DatabaseExecutor(database, copy=True)


def _execution_environment(workdir: Any):
    if workdir is None:
        return None
    from simguard.execution_environment import ExecutionEnvironment

    return workdir if isinstance(workdir, ExecutionEnvironment) else ExecutionEnvironment(workdir=workdir)


def _tool_executors(output: dict) -> dict | None:
    """이 출력의 도구 호출에 쓸 에이전트 DB. 우선순위: databases[도구] > database(SQL 호출)."""
    executors = dict(_settings["databases"] or {})
    if _settings["database"] is not None:
        from simguard.live_db import is_sql_call

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


def _is_target(model: Any) -> bool:
    target = _settings["target_model"]
    return target is None or _same_model(model, target)


_originals: dict[tuple[Any, str], Any] = {}


class SimGuardBlocked(Exception):
    """on_block="raise"일 때, 또는 차단 문구로 바꿀 수 없는 스트림이 BLOCK일 때 일어난다."""

    def __init__(self, result: InterceptionResult) -> None:
        super().__init__(BLOCKED_MESSAGE)
        self.result = result


async def _judge(messages: list, tools: list | None, output: dict) -> InterceptionResult:
    with bypass():
        # 판정기의 LLM 클라이언트는 이벤트 루프에 묶이므로 호출마다 새로 만든다.
        return await SimGuardAgentInterceptor().intercept_output(
            messages, output, tools,
            tool_executors=_tool_executors(output),
            execution_environment=_settings["execution_environment"],
        )


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


def _items(call: LLMCall, response: Any, args: tuple, kwargs: dict, messages: list) -> list[tuple[list, dict]]:
    items = []
    for item in call.outputs(response, args, kwargs):
        conversation, output = item if isinstance(item, tuple) else (messages, item)
        items.append((conversation, output if isinstance(output, dict) else {"content": str(output)}))
    return items


def _apply(call: LLMCall, response: Any, results: list[InterceptionResult]) -> Any:
    _report(results)
    for index, result in enumerate(results):
        if result.decision == Decision.BLOCK:
            replaced = call.block(response, index)
            if replaced is not None:
                response = replaced
    return response


# ---------------------------------------------------------------- streaming

def _snapshot(chunk: Any) -> Any:
    # 스트림 구현이 청크 객체를 재사용해도 받은 그대로 다시 내보낼 수 있게 복사한다.
    return copy.deepcopy(chunk)


def _merge_stream(call: LLMCall, chunks: list) -> dict[int, dict]:
    if call.merge_stream is None:
        raise NotImplementedError(f"SimGuardAgent cannot judge a streamed {call.name} response (no merge_stream)")
    return call.merge_stream(chunks)


def _decide_stream(call: LLMCall, chunks: list, outputs: dict[int, dict], results: list[InterceptionResult]) -> list:
    _report(results)
    blocked = {index for index, result in zip(outputs, results) if result.decision == Decision.BLOCK}
    if not blocked:
        return chunks
    if call.replace_stream is None:
        raise SimGuardBlocked(next(result for result in results if result.decision == Decision.BLOCK))
    return call.replace_stream(chunks, blocked)


class _GuardedStream:
    """동기 스트림. 처음 읽을 때 끝까지 받아 판정한다."""

    def __init__(self, stream: Any, call: LLMCall, messages: list, tools: list | None) -> None:
        self._stream, self._call, self._messages, self._tools = stream, call, messages, tools
        self._iterator = None

    def _ready(self):
        if self._iterator is None:
            chunks = [_snapshot(chunk) for chunk in self._stream]
            outputs = _merge_stream(self._call, chunks)
            results = [_judge_sync(self._messages, self._tools, output) for output in outputs.values()]
            self._iterator = iter(_decide_stream(self._call, chunks, outputs, results))
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

    def __init__(self, stream: Any, call: LLMCall, messages: list, tools: list | None) -> None:
        self._stream, self._call, self._messages, self._tools = stream, call, messages, tools
        self._iterator = None

    async def _ready(self):
        if self._iterator is None:
            chunks = [_snapshot(chunk) async for chunk in self._stream]
            outputs = _merge_stream(self._call, chunks)
            results = [await _judge(self._messages, self._tools, output) for output in outputs.values()]
            self._iterator = iter(_decide_stream(self._call, chunks, outputs, results))
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

def _inner(call: LLMCall):
    return bypass() if call.bypass_inner else contextlib.nullcontext()


def _guard(call: LLMCall) -> Callable[[Callable], Callable]:
    def wrap(original: Callable) -> Callable:
        is_async = call.is_async if call.is_async is not None else inspect.iscoroutinefunction(original)

        def judged(args: tuple, kwargs: dict) -> bool:
            return not _BYPASS.get() and _is_target(call.model(args, kwargs))

        def streamed(args: tuple, kwargs: dict) -> bool:
            return call.stream is not None and call.stream(args, kwargs)

        if is_async:
            @functools.wraps(original)
            async def guarded(*args, **kwargs):
                if not judged(args, kwargs):
                    return await original(*args, **kwargs)
                messages, tools = call.request(args, kwargs)
                with _inner(call):
                    response = await original(*args, **kwargs)
                if streamed(args, kwargs):
                    return _GuardedAsyncStream(response, call, messages, tools)
                items = _items(call, response, args, kwargs, messages)
                return _apply(call, response, [await _judge(m, tools, output) for m, output in items])
        else:
            @functools.wraps(original)
            def guarded(*args, **kwargs):
                if not judged(args, kwargs):
                    return original(*args, **kwargs)
                messages, tools = call.request(args, kwargs)
                with _inner(call):
                    response = original(*args, **kwargs)
                if streamed(args, kwargs):
                    return _GuardedStream(response, call, messages, tools)
                items = _items(call, response, args, kwargs, messages)
                return _apply(call, response, [_judge_sync(m, tools, output) for m, output in items])

        return guarded
    return wrap


def _patch(owner: Any, name: str, wrap: Callable[[Callable], Callable]) -> None:
    if (owner, name) in _originals:
        return
    original = getattr(owner, name)
    _originals[(owner, name)] = original
    setattr(owner, name, wrap(original))


def _connect(call: LLMCall) -> None:
    _patch(call.owner, call.name, _guard(call))


# ---------------------------------------------------------------- 호출 방법 연결

_connected: set[str] = set()


def _connect_provider(name: str) -> None:
    if name in _connected:
        return
    _connected.add(name)
    for call in PROVIDERS[name][1]():
        _connect(call)


class _ConnectOnImport(importlib.abc.MetaPathFinder):
    """아직 import되지 않은 SDK는 import가 끝나는 순간 연결한다 (무거운 GPU 라이브러리를 미리 불러오지 않는다)."""

    def __init__(self) -> None:
        self.pending: dict[str, list[str]] = {}

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if fullname not in self.pending:
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        exec_module = getattr(getattr(spec, "loader", None), "exec_module", None)
        if exec_module is None:
            return spec
        names = self.pending.pop(fullname)

        def exec_and_connect(module: Any) -> None:
            exec_module(module)
            for name in names:
                _connect_provider(name)

        spec.loader.exec_module = exec_and_connect
        return spec


_hook = _ConnectOnImport()


def _provider_names(providers: str | Iterable[str]) -> tuple[list[str], bool]:
    if isinstance(providers, str):
        if providers.strip().lower() == "auto":
            return list(PROVIDERS), False
        providers = [name for name in providers.split(",") if name.strip()]
    names = [str(name).strip().lower() for name in providers]
    unknown = [name for name in names if name not in PROVIDERS]
    if unknown:
        raise ValueError(f"unknown LLM provider(s) {unknown}; choose from {', '.join(PROVIDERS)} or pass llm_calls=")
    return names, True


def _connect_providers(providers: str | Iterable[str]) -> None:
    names, explicit = _provider_names(providers)
    for name in names:
        module = PROVIDERS[name][0]
        package = module.split(".")[0]
        if module in sys.modules:
            _connect_provider(name)
        elif importlib.util.find_spec(package) is None:
            if explicit:
                raise ImportError(f"LLM provider {name!r} is not installed (package {package!r})")
        elif name not in _connected and name not in _hook.pending.get(module, []):
            _hook.pending.setdefault(module, []).append(name)
    if _hook.pending and _hook not in sys.meta_path:
        sys.meta_path.insert(0, _hook)


def protect(
    on_block: str = "replace",
    on_result: Callable[[InterceptionResult], None] | None = None,
    providers: str | Iterable[str] | None = None,
    llm_calls: Iterable[LLMCall] = (),
    target_model: str | None = None,
    database: Any = None,
    databases: Mapping[str, Any] | None = None,
    live_reads: bool = True,
    live_read_hosts: Any = None,
    live_read_ports: Any = (80, 443),
    tools: Any = None,
    workdir: Any = None,
) -> None:
    """이 프로세스의 타겟 모델 호출에 SimGuardAgent Interceptor를 연결한다.

    providers: 가로챌 호출 방법. 주지 않으면 SIMGUARD_PROVIDERS 환경 변수, 그것도 없으면 "auto".
        "auto"는 설치된 방법을 모두 연결한다.
        목록으로 고를 수 있다: openai, litellm, anthropic, ollama, vllm, transformers.
        목록으로 고른 방법이 설치돼 있지 않으면 ImportError를 일으킨다.
    llm_calls: 목록에 없는 호출 방법 (simguard.LLMCall). 다른 SDK나 직접 만든 generate 함수 등.
    on_block: "replace"는 출력을 차단 문구로 바꾸고, "raise"는 SimGuardBlocked를 일으킨다.
    on_result: 판정마다 InterceptionResult를 받는 함수 (기록용).
    target_model: 지정하면 이 모델로 가는 호출만 판정한다. 지정하지 않으면 모든 호출을 판정한다.
    database: 에이전트가 쓰는 DB. 인자에 SQL 문이 있는 도구 호출은 이 DB에서 실행한 결과로 판정한다.
        SQLite 경로, SQLAlchemy URL이나 Engine, 또는 DB 연결을 만드는 함수(lambda: psycopg.connect(DSN)).
        SQLite는 판정 순간의 상태를 복사해 복사본에서, 그 외 DB는 트랜잭션 안에서 실행하고 되돌린다.
    databases: DB가 여럿일 때 {도구 이름: DB}. 값은 database와 같은 형식이다.
        DB를 알려 주지 않으면 SQL 호출은 SimGuardAgent 내부 SQLite에서 실행한다 (simguard.internal_database).
    live_reads: True면 이 프로세스에 있는 실제 도구를 GET 전용 모드로 실행해 실제 결과로 판정한다.
        GET·HEAD가 아닌 HTTP 요청, 파일 쓰기, 프로세스 실행, HTTP가 아닌 포트 연결이 나오면 멈추고
        내부 모델이 결과를 예상한다 (simguard.live_http).
    live_read_hosts: GET을 보내도 되는 도메인 목록. None이면 모든 도메인.
    live_read_ports: GET 전용 실행에서 연결해도 되는 포트.
    tools: 에이전트가 쓰는 실제 도구. 목록 또는 {이름: 도구}. 알려 준 도구는 live_reads가 실제로 실행한다.
        도구 객체와 Python 함수를 공통 규칙으로 읽는다 (simguard.as_real_tool). 이름을 읽을 수 없는
        도구는 {이름: 도구}로 넘긴다.
    workdir: 에이전트가 코드를 실행하는 작업 폴더, 또는 simguard.ExecutionEnvironment. 주면 코드·셸 행동을
        그 폴더의 파일을 복사해 넣은 Docker 컨테이너에서 실행해 본다. 주지 않으면 빈 컨테이너에서 실행한다.
    여러 번 불러도 한 번만 연결되고 설정만 바뀐다.
    """
    if on_block not in {"replace", "raise"}:
        raise ValueError('on_block must be "replace" or "raise"')
    from simguard.live_http import configure_live_reads
    from simguard.tool_discovery import register_tools

    configure_live_reads(live_reads, live_read_hosts, live_read_ports)
    register_tools(tools)
    _settings.update(on_block=on_block, on_result=on_result, target_model=target_model,
                     database=_database_executor(database) if database is not None else None,
                     databases={name: _database_executor(db) for name, db in databases.items()} if databases else None,
                     execution_environment=_execution_environment(workdir))

    _connect_providers(providers if providers is not None else os.getenv("SIMGUARD_PROVIDERS", "auto"))
    for call in llm_calls:
        _connect(call)

    from simguard.tool_discovery import track_mcp

    # 에이전트가 MCP 서버에서 받아 가는 도구를 기록해 실제 도구를 찾을 수 있게 한다.
    track_mcp()


def unprotect() -> None:
    """protect()로 연결한 것을 모두 되돌린다."""
    for (owner, name), original in list(_originals.items()):
        setattr(owner, name, original)
    _originals.clear()
    _connected.clear()
    _hook.pending.clear()
    if _hook in sys.meta_path:
        sys.meta_path.remove(_hook)
    from simguard.tool_discovery import untrack_mcp

    untrack_mcp()
