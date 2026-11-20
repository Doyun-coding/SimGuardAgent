"""
타겟 에이전트가 실제로 가진 도구를 이름으로 찾아온다.

LLM 요청에는 도구 설명만 있고 실행할 코드는 없다. 연결하는 곳에서 알려 준 도구 객체를 이름, 설명,
인자 스키마, 실제 실행 함수를 묶은 RealTool 하나로 감싼다. 특정 에이전트 프레임워크를 가정하지 않는다.

  simguard.protect(tools=[send_email, search_hotels])        # 목록 (이름은 도구에서 읽는다)
  simguard.protect(tools={"send_email": my_send_email})       # {이름: 도구}
  simguard.register_tools([...])                              # protect 밖에서 등록

  tool = simguard.find_tool("send_email")      # 없거나 같은 이름이 여럿이면 None
  tool.call({"to": "...", "body": "..."})      # 실제 도구를 실행한다 (부작용 주의)

도구 객체는 공통 규칙으로 읽는다 (as_real_tool):
  이름  name 속성 > 함수 이름.  설명  description 속성 > docstring.
  인자  parameters / input_schema / tool_call_schema / args_schema / inputs 속성 > 함수 시그니처.
  실행  invoke(인자) 메서드 > obj(**인자).
도구에서 이름을 읽을 수 없으면 {이름: 도구}로 넘긴다.
MCP 도구: ClientSession.list_tools()로 에이전트가 받아 간 도구를 기록한다 (track_mcp()).

같은 이름의 서로 다른 도구가 여럿이면 어느 것인지 알 수 없으므로 찾지 못한 것으로 본다.
주의: RealTool.call()은 격리 없이 실제 도구를 실행한다. 판정 전 실행은 live_http(GET 전용)가 감싼다.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
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
    function: Any = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict | None = None,
):
    """도구 하나를 등록한다. 데코레이터로도 쓸 수 있다.

    Python 함수와 도구 객체를 받는다 (읽는 규칙은 as_real_tool).

    @simguard.register_tool
    def send_email(to: str, body: str) -> str: ...
    """
    def register(target: Any) -> Any:
        tool = as_real_tool(target, name=name)
        if description is not None or parameters is not None:
            tool = RealTool(
                name=tool.name,
                description=description if description is not None else tool.description,
                parameters=parameters or tool.parameters,
                framework=tool.framework,
                target=tool.target,
                _invoke=tool._invoke,
                _loop=tool._loop,
            )
        with _lock:
            _registered[tool.name] = tool
        return target

    return register(function) if function is not None else register


def register_tools(tools: Any) -> list[str]:
    """도구 여럿을 등록한다. 목록이면 이름을 도구에서 읽고, {이름: 도구}면 그 이름으로 등록한다."""
    if tools is None:
        return []
    items = list(tools.items()) if isinstance(tools, Mapping) else [(None, tool) for tool in tools]
    wrapped = [(name, as_real_tool(tool, name=name)) for name, tool in items]
    seen: dict[str, int] = {}
    for _, tool in wrapped:
        seen[tool.name] = seen.get(tool.name, 0) + 1
    clashes = sorted(name for name, count in seen.items() if count > 1)
    if clashes:
        # 예: 안쪽 함수 이름(execute)이 이름으로 읽힌 도구들. 잘못된 도구를 실행하지 않도록 등록하지 않는다.
        raise ValueError(f"several tools read the same name {clashes}; pass them as {{name: tool}}")
    for (name, tool), (_, real) in zip(items, wrapped):
        register_tool(tool, name=real.name)
    return [real.name for _, real in wrapped]


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


# ---------------------------------------------------------------- tool objects

def _schema(value: Any) -> dict | None:
    """JSON 스키마(dict) 또는 스키마를 내보내는 클래스(model_json_schema)를 dict로 바꾼다."""
    if isinstance(value, Mapping):
        return dict(value)
    export = getattr(value, "model_json_schema", None) or getattr(value, "schema", None)
    if callable(export):
        try:
            schema = export()
        except Exception:
            return None
        return dict(schema) if isinstance(schema, Mapping) else None
    return None


def _inputs_schema(inputs: Mapping) -> dict:
    """{인자: {"type", "description", "nullable"}} 형식의 인자 목록을 JSON 스키마로 바꾼다."""
    properties, required = {}, []
    for key, spec in inputs.items():
        spec = spec if isinstance(spec, Mapping) else {}
        properties[key] = {k: v for k, v in spec.items() if k in {"type", "description", "enum", "items"}}
        if not spec.get("nullable") and not spec.get("optional"):
            required.append(key)
    return {"type": "object", "properties": properties, "required": required}


def _parameters(obj: Any, call: Callable) -> dict:
    """도구의 인자 스키마. 도구가 스키마를 갖고 있으면 그것을, 없으면 호출 함수의 시그니처를 쓴다."""
    for attribute in ("parameters", "input_schema", "tool_call_schema", "args_schema"):
        schema = _schema(getattr(obj, attribute, None))
        if schema and schema.get("properties") is not None:
            return schema
    inputs = getattr(obj, "inputs", None)
    if isinstance(inputs, Mapping) and inputs:
        return _inputs_schema(inputs)
    try:
        return _function_schema(call)
    except (TypeError, ValueError):
        return {"type": "object", "properties": {}}


def _invoker(obj: Any) -> Callable[[dict], Any]:
    """도구를 인자 dict로 실행하는 방법. invoke(인자) 메서드가 있으면 그것을, 없으면 obj(**인자)로 부른다."""
    invoke = getattr(obj, "invoke", None)
    if callable(invoke):
        return lambda arguments: invoke(arguments)
    if callable(obj):
        return lambda arguments: obj(**arguments)
    raise TypeError(f"not a tool: {obj!r} (no invoke() and not callable)")


def as_real_tool(obj: Any, name: str | None = None) -> RealTool:
    """도구 객체 하나를 RealTool로 감싼다. 특정 프레임워크를 가정하지 않는다.

    이름      name > 도구의 name 속성 > 함수 이름 (__name__)
    설명      description 속성 > docstring
    인자      parameters / input_schema / tool_call_schema / args_schema 속성 (JSON 스키마 또는
              model_json_schema()를 가진 클래스) > inputs 속성 ({인자: {"type", ...}}) > 호출 함수의 시그니처
    실행      invoke(인자) 메서드 > obj(**인자)
    이름을 읽을 수 없는 도구(예: 이름이 안쪽 함수 이름인 도구)는 {이름: 도구}로 넘긴다.
    """
    if isinstance(obj, RealTool):
        tool = obj
    else:
        invoke = _invoker(obj)
        call = obj if inspect.isfunction(obj) or inspect.ismethod(obj) else getattr(obj, "forward", None) or obj
        own_name = getattr(obj, "name", None)
        tool = RealTool(
            name=name or (own_name if isinstance(own_name, str) and own_name else getattr(obj, "__name__", "")),
            description=str(getattr(obj, "description", None) or inspect.getdoc(obj) or ""),
            parameters=_parameters(obj, call),
            framework="function" if inspect.isroutine(obj) else "object",
            target=obj,
            _invoke=invoke,
        )
        if not tool.name or tool.name == "<lambda>":
            raise ValueError(f"cannot read a name for tool {obj!r}; pass it as {{name: tool}}")
    if name and name != tool.name:
        tool = RealTool(name=name, description=tool.description, parameters=tool.parameters,
                        framework=tool.framework, target=tool.target, _invoke=tool._invoke, _loop=tool._loop)
    return tool


def _unique(candidates: list[RealTool]) -> RealTool | None:
    distinct = {id(tool.target): tool for tool in candidates}
    return next(iter(distinct.values())) if len(distinct) == 1 else None


def discover_tools() -> dict[str, list[RealTool]]:
    """알려 준 도구(register_tool·protect(tools=...))와 에이전트가 MCP 서버에서 받은 도구를 이름별로 모은다."""
    found: dict[str, list[RealTool]] = {}
    with _lock:
        for name, tool in _registered.items():
            found.setdefault(name, []).append(tool)
        for name, tools in _mcp_tools.items():
            found.setdefault(name, []).extend(tools)
    return found


def find_tools(names: list[str] | None = None) -> dict[str, RealTool]:
    """이름마다 실제 도구 하나를 찾는다. 없거나 같은 이름이 여럿이면 빠진다."""
    found = discover_tools()
    wanted = found if names is None else {name: found.get(name, []) for name in names}
    tools = {name: _unique(candidates) for name, candidates in wanted.items() if candidates}
    return {name: tool for name, tool in tools.items() if tool is not None}


def find_tool(name: str) -> RealTool | None:
    return find_tools([name]).get(name)

