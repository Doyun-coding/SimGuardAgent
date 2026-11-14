"""
타겟 에이전트가 실제로 가진 도구를 이름으로 찾아온다.

LLM 요청에는 도구 설명만 있고 실행할 코드는 없다. 에이전트와 같은 프로세스에 있는 도구 객체를
찾아 이름, 설명, 인자 스키마, 실제 실행 함수를 RealTool 하나로 묶는다.

  tool = simguard.find_tool("send_email")      # 없거나 같은 이름이 여럿이면 None
  tool.parameters                              # JSON 스키마
  tool.call({"to": "...", "body": "..."})      # 실제 도구를 실행한다 (부작용 주의)

지원:
  LangChain  BaseTool (@tool, StructuredTool, BaseTool 하위 클래스)
  smolagents Tool (@tool, Tool 하위 클래스)
  MCP        ClientSession.list_tools()로 에이전트가 받아 간 도구 (track_mcp() 이후)
  그 외      register_tool()로 등록한 Python 함수

프레임워크 모듈이 이미 import된 경우에만 그 프레임워크의 도구를 찾는다. SimGuardAgent가
프레임워크를 대신 import하지 않는다. 같은 이름의 서로 다른 도구가 여럿이면 어느 것인지 알 수
없으므로 찾지 못한 것으로 본다.

주의: RealTool.call()은 격리 없이 실제 도구를 실행한다. 판정용 shadow 실행에 쓰려면 도구 종류에
맞는 격리(트랜잭션 롤백, Docker, mock)가 필요하다.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import inspect
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RealTool:
    name: str
    description: str
    parameters: dict
    framework: str
    target: Any = field(repr=False)
    _invoke: Callable[[dict], Any] = field(repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, repr=False)

    async def acall(self, arguments: Mapping[str, Any] | None = None) -> Any:
        """실제 도구를 실행한다. 비동기 도구는 원래 이벤트 루프에서 실행한다."""
        result = self._invoke(dict(arguments or {}))
        if not inspect.isawaitable(result):
            return result
        running = asyncio.get_running_loop()
        if self._loop is None or self._loop is running:
            return await result
        future = asyncio.run_coroutine_threadsafe(result, self._loop)
        return await asyncio.wrap_future(future)

    def call(self, arguments: Mapping[str, Any] | None = None) -> Any:
        """실제 도구를 실행한다 (동기)."""
        result = self._invoke(dict(arguments or {}))
        if not inspect.isawaitable(result):
            return result
        if self._loop is not None and self._loop.is_running():
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is self._loop:
                raise RuntimeError(f"{self.name} is async; use 'await tool.acall(...)' inside its event loop")
            return asyncio.run_coroutine_threadsafe(result, self._loop).result()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(result)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, result).result()


# ---------------------------------------------------------------- registered functions

_registered: dict[str, RealTool] = {}
_lock = threading.Lock()


def _function_schema(function: Callable) -> dict:
    properties, required = {}, []
    for name, parameter in inspect.signature(function).parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        properties[name] = {}
        if parameter.default is parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def register_tool(
    function: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict | None = None,
):
    """프레임워크 없이 만든 도구 함수를 등록한다. 데코레이터로도 쓸 수 있다.

    @simguard.register_tool
    def send_email(to: str, body: str) -> str: ...
    """
    def register(target: Callable) -> Callable:
        tool_name = name or target.__name__
        tool = RealTool(
            name=tool_name,
            description=description if description is not None else inspect.getdoc(target) or "",
            parameters=parameters or _function_schema(target),
            framework="function",
            target=target,
            _invoke=lambda arguments: target(**arguments),
        )
        with _lock:
            _registered[tool_name] = tool
        return target

    return register(function) if function is not None else register


def unregister_tool(name: str) -> None:
    with _lock:
        _registered.pop(name, None)


# ---------------------------------------------------------------- MCP

_mcp_tools: dict[str, list[RealTool]] = {}
_mcp_original: Callable | None = None


def _mcp_tool(session: Any, tool: Any, loop: asyncio.AbstractEventLoop) -> RealTool:
    return RealTool(
        name=tool.name,
        description=tool.description or "",
        parameters=dict(tool.inputSchema or {}),
        framework="mcp",
        target=session,
        _invoke=lambda arguments: session.call_tool(tool.name, arguments),
        _loop=loop,
    )


def track_mcp() -> bool:
    """MCP 클라이언트가 list_tools()로 받아 가는 도구를 기록한다. mcp가 없으면 False."""
    global _mcp_original
    try:
        from mcp import ClientSession
    except ImportError:
        return False
    if _mcp_original is not None:
        return True
    original = ClientSession.list_tools

    async def list_tools(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        loop = asyncio.get_running_loop()
        with _lock:
            for tool in result.tools:
                entries = [entry for entry in _mcp_tools.get(tool.name, []) if entry.target is not self]
                _mcp_tools[tool.name] = [*entries, _mcp_tool(self, tool, loop)]
        return result

    list_tools.__simguard_original__ = original
    ClientSession.list_tools = list_tools
    _mcp_original = original
    return True


def untrack_mcp() -> None:
    global _mcp_original
    if _mcp_original is None:
        return
    from mcp import ClientSession

    ClientSession.list_tools = _mcp_original
    _mcp_original = None
    with _lock:
        _mcp_tools.clear()


# ---------------------------------------------------------------- framework objects

def _langchain_tool(tool: Any) -> RealTool:
    schema = getattr(tool, "tool_call_schema", None)
    if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
        parameters = schema.model_json_schema()
    elif isinstance(schema, dict):
        parameters = schema
    else:
        parameters = {"type": "object", "properties": dict(getattr(tool, "args", {}) or {})}
    return RealTool(
        name=tool.name,
        description=tool.description or "",
        parameters=parameters,
        framework="langchain",
        target=tool,
        _invoke=tool.invoke,
    )


def _smolagents_tool(tool: Any) -> RealTool:
    inputs = dict(getattr(tool, "inputs", {}) or {})
    parameters = {
        "type": "object",
        "properties": {key: {k: v for k, v in spec.items() if k in {"type", "description"}} for key, spec in inputs.items()},
        "required": [key for key, spec in inputs.items() if not spec.get("nullable")],
    }
    return RealTool(
        name=tool.name,
        description=tool.description or "",
        parameters=parameters,
        framework="smolagents",
        target=tool,
        _invoke=lambda arguments: tool(**arguments),
    )


def _framework_kinds() -> list[tuple[type, Callable[[Any], RealTool]]]:
    """이미 import된 프레임워크의 도구 클래스만 돌려준다."""
    kinds = []
    if "langchain_core.tools" in sys.modules:
        kinds.append((sys.modules["langchain_core.tools"].BaseTool, _langchain_tool))
    if "smolagents" in sys.modules:
        kinds.append((sys.modules["smolagents"].Tool, _smolagents_tool))
    return kinds


def _identity(tool: RealTool) -> Any:
    """같은 도구를 가리키는 객체는 하나로 센다 (LangChain StructuredTool은 감싼 함수로)."""
    return getattr(tool.target, "func", None) or getattr(tool.target, "coroutine", None) or tool.target


def _unique(candidates: list[RealTool]) -> RealTool | None:
    distinct = {id(_identity(tool)): tool for tool in candidates}
    return next(iter(distinct.values())) if len(distinct) == 1 else None


def discover_tools() -> dict[str, list[RealTool]]:
    """지금 프로세스에 있는 도구를 이름별로 모두 모은다."""
    found: dict[str, list[RealTool]] = {}
    with _lock:
        for name, tool in _registered.items():
            found.setdefault(name, []).append(tool)
        for name, tools in _mcp_tools.items():
            found.setdefault(name, []).extend(tools)
    kinds = _framework_kinds()
    if kinds:
        classes = tuple(kind for kind, _ in kinds)
        for obj in gc.get_objects():
            if isinstance(obj, classes) and isinstance(getattr(obj, "name", None), str):
                for kind, wrap in kinds:
                    if isinstance(obj, kind):
                        try:
                            tool = wrap(obj)
                        except Exception:
                            break
                        found.setdefault(tool.name, []).append(tool)
                        break
    return found


def find_tools(names: list[str] | None = None) -> dict[str, RealTool]:
    """이름마다 실제 도구 하나를 찾는다. 없거나 같은 이름이 여럿이면 빠진다."""
    found = discover_tools()
    wanted = found if names is None else {name: found.get(name, []) for name in names}
    tools = {name: _unique(candidates) for name, candidates in wanted.items() if candidates}
    return {name: tool for name, tool in tools.items() if tool is not None}


def find_tool(name: str) -> RealTool | None:
    return find_tools([name]).get(name)

