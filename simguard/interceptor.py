"""
SimGuardAgent Interceptor

타겟 에이전트의 LLM 출력(Plan)을 실행 전에 가로챈다.

  intercept_output(대화, 출력, 도구)
    -> response_actions        출력 하나를 Plan 하나로 읽어 실행하려는 행동으로 바꾼다
    -> Simulation Manager      행동마다 한 경로로 보낸다 (Behavior / ToolCall / Execution)
    -> Simulator -> Judge      시뮬레이션한 결과를 판정한다
    -> Decision Controller     ALLOW / BLOCK
"""
from __future__ import annotations

import ast
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence

from simguard.decision.controller import DecisionController
from simguard.models import Action, ActionType, AgentPlan, Decision, InterceptionResult, SimulationTrace
from simguard.simulation.manager import SimulationManager
from simguard.internal_database import internal_database_for
from simguard.shadow_http import live_read_executor_for
from simguard.shadow_api import ShadowToolExecutor


BLOCKED_MESSAGE = "This action was blocked by SimGuardAgent."
_SHELL_TOOLS = {
    "bash", "shell", "run_bash", "execute_bash", "run_shell",
    "terminal", "cmd", "powershell", "system_exec", "exec",
}
_CODE_TOOLS = {
    "python", "execute_python", "python_repl", "run_python", "python_interpreter",
    "execute_code", "run_code", "code_interpreter", "code_execution",
    "execute_javascript", "run_node", "run_js",
}
_SHELL_NAME_TOKENS = {"bash", "sh", "shell", "zsh", "terminal", "cmd", "command", "powershell"}
_CODE_NAME_TOKENS = {"python", "python3", "ipython", "javascript", "js", "node", "code", "repl"}
_SCRIPT_NAME_TOKENS = {"javascript", "js", "node"}
_RUN_NAME_TOKENS = {"run", "exec", "execute", "eval"}
_CODE_ARGUMENT_KEYS = ("command", "commands", "cmd", "code", "script", "input", "source")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_MAX_PLAN_TEXT = 20000
# 타겟 입력을 시뮬레이터·Judge에 넘길 때의 최대 길이(문자). 이보다 짧으면 그대로 넘기고,
# 길면 오래된 도구 결과부터 줄인다. 내부 모델의 입력 한도에 맞춰 조정한다.
_MAX_CONVERSATION_CHARS = int(os.getenv("SIMGUARD_MAX_CONVERSATION_CHARS", "120000"))
_SHORTENED_RESULT_CHARS = 1000
_REASONING_KEYS = ("reasoning_content", "reasoning")
# ㄹ 받침으로 끝나는 한글 음절 ("보낼 예정", "삭제할 계획")
_RIEUL_ENDING = "".join(chr(code) for code in range(0xAC00, 0xD7A4) if (code - 0xAC00) % 28 == 8)
# 앞으로 할 행동을 서술하는 표현. 애매하면 Plan으로 본다.
_FUTURE_ACTION_RE = re.compile(
    r"\b(?:i|we)(?:'ll|'m going to|'re going to|\s+(?:will|should|am going to|are going to|plan to|intend to|need to|want to))\b"
    r"|\blet(?:\s+me|'s)\b"
    r"|\bwill\s+now\b"
    r"|\bnext\s+step\b"
    r"|^\s*(?:plan|next\s+steps?|thought|action)\s*:"
    r"|(?<!알)(?<!모르)(?<!좋)(?<!감사하)겠(?:습니다|어요|다)"
    r"|[할볼갈줄릴을쓸낼올]게(?:요|\s|[.!,]|$)"
    rf"|[{_RIEUL_ENDING}]\s*(?:예정|계획)"
    r"|려고\s*(?:합니다|해요|한다|함)"
    r"|^\s*(?:계획|다음\s*단계)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
_SHELL_COMMAND = (
    r"(?:sudo|rm|curl|wget|chmod|chown|bash|sh|nc|cat|ls|mv|cp"
    r"|python3?\s+(?:-c|\S+\.py)|pip3?\s+install|git\s+\w+)\b"
)
_FENCE_RE = re.compile(r"```([\w+-]*)[ \t]*\n?(.*?)```", re.DOTALL)
_SHELL_LINE_RE = re.compile(rf"(?m)^[ \t]*(?:\$[ \t]+)?({_SHELL_COMMAND}[^\n]*)$")
_INLINE_SHELL_RE = re.compile(rf"`({_SHELL_COMMAND}[^`]*)`")
_PYTHON_SOURCE_RE = re.compile(r"(?m)^\s*(?:import\s+\w|from\s+\w+\s+import|def\s+\w+\(|print\()")
_PYTHON_FENCES = {"python", "py", "python3"}
_SCRIPT_FENCES = {"javascript", "js", "node"}
_SHELL_FENCES = {"bash", "sh", "shell", "zsh", "console"}
_JSON_NAME_KEYS = ("name", "tool", "tool_name", "action")
_JSON_ARGUMENT_KEYS = ("arguments", "args", "parameters", "input", "action_input")
_JSON_AFTER_NAME_RE = re.compile(
    r"[`\"']?\s*(?:(?:action[ _]input|arguments?|args|input|with(?:\s+(?:arguments?|args|input))?)?\s*:?\s*)(?:```\w*\s*)?(?=\{)",
    re.IGNORECASE,
)


def _code_in_plan(text: str) -> dict | None:
    for language, body in _FENCE_RE.findall(text):
        language, body = language.lower(), body.strip()
        if not body:
            continue
        if language in _PYTHON_FENCES or (not language and _PYTHON_SOURCE_RE.search(body)):
            return {"code": body, "language": "python"}
        if language in _SCRIPT_FENCES:
            return {"code": body, "language": "javascript"}
        if language in _SHELL_FENCES or (not language and _SHELL_LINE_RE.search(body)):
            return {"code": body, "language": "shell"}
    command = _SHELL_LINE_RE.search(text) or _INLINE_SHELL_RE.search(text)
    if command:
        return {"code": command.group(1).strip(), "language": "shell"}
    return None


def _closing(text: str, start: int, opening: str, closing: str) -> int:
    """start의 여는 괄호와 짝이 맞는 닫는 괄호 위치를 찾는다. 없으면 -1."""
    depth, quote, index = 0, None, start
    while index < len(text):
        char = text[index]
        if quote:
            if char == "\\":
                index += 1
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _call_arguments(source: str, parameters: Sequence[str]) -> dict | None:
    """함수 호출 형태의 인자를 실행하지 않고 읽는다. 값을 확정할 수 없으면 None."""
    try:
        call = ast.parse(f"_({source})", mode="eval").body
        if not isinstance(call, ast.Call) or any(item.arg is None for item in call.keywords):
            return None
        values = [ast.literal_eval(item) for item in call.args]
        named = {item.arg: ast.literal_eval(item.value) for item in call.keywords}
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return None
    if len(values) > len(parameters) or Ellipsis in values or Ellipsis in named.values():
        return None
    return {**dict(zip(parameters, values)), **named}


def _arguments_after(plan: str, end: int, parameters: Sequence[str]) -> tuple[dict | None, int]:
    """Plan에서 도구 이름 바로 뒤에 적힌 인자를 읽는다."""
    if plan[end:end + 1] == "(":
        close = _closing(plan, end, "(", ")")
        if close != -1:
            return _call_arguments(plan[end + 1:close], parameters), close + 1
    elif plan[end:end + 1] == "[":
        close = _closing(plan, end, "[", "]")
        if close != -1 and parameters:
            return {parameters[0]: plan[end + 1:close].strip()}, close + 1
    else:
        match = _JSON_AFTER_NAME_RE.match(plan, end)
        if match:
            try:
                value, stop = json.JSONDecoder().raw_decode(plan, match.end())
            except json.JSONDecodeError:
                return None, end
            if isinstance(value, dict):
                return value, stop
    return None, end


def _json_tool_calls(plan: str, tools: Mapping[str, Sequence[str]]) -> list[tuple[int, int, str, dict | None]]:
    """Plan에 JSON으로 적힌 등록 도구 호출을 찾는다."""
    decoder, found, position = json.JSONDecoder(), [], plan.find("{")
    while position != -1:
        try:
            value, end = decoder.raw_decode(plan, position)
        except json.JSONDecodeError:
            position = plan.find("{", position + 1)
            continue
        if isinstance(value, dict):
            name = next((value[key] for key in _JSON_NAME_KEYS if isinstance(value.get(key), str) and value[key] in tools), None)
            if name:
                arguments = next((value[key] for key in _JSON_ARGUMENT_KEYS if isinstance(value.get(key), dict)), None)
                found.append((position, end, name, arguments))
        position = plan.find("{", end)
    return found


def plan_tool_calls(plan: str, tools: Mapping[str, Sequence[str]]) -> list[dict]:
    """Plan에 적힌 등록 도구를 순서대로 찾는다.

    tools는 {도구 이름: 인자 이름 목록}이다. tool_arguments가 None이면 Plan에
    이름만 있고 실제 실행에 쓸 인자는 확정하지 못한 것이다.
    """
    if not tools:
        return []
    found = _json_tool_calls(plan, tools)
    spans = [(start, end) for start, end, _, _ in found]
    pattern = re.compile(
        r"(?<![\w-])(" + "|".join(re.escape(name) for name in sorted(tools, key=len, reverse=True)) + r")(?![\w-])"
    )
    position = 0
    while match := pattern.search(plan, position):
        if any(start <= match.start() < end for start, end in spans):
            position = match.end()
            continue
        name = match.group(1)
        arguments, stop = _arguments_after(plan, match.end(), tools[name])
        found.append((match.start(), stop, name, arguments))
        position = max(stop, match.end())

    calls: list[dict] = []
    named_with_arguments = {name for _, _, name, arguments in found if arguments is not None}
    for _, _, name, arguments in sorted(found, key=lambda item: item[0]):
        call = {"tool_name": name, "tool_arguments": arguments}
        if arguments is None and name in named_with_arguments:
            continue
        if call not in calls:
            calls.append(call)
    return calls


def _name_tokens(name: str) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", name.lower()))


def _code_tool_kind(name: str) -> str | None:
    """코드나 셸을 실행하는 도구이면 "shell" 또는 "code"를 돌려준다."""
    name = name.lower()
    if name in _SHELL_TOOLS:
        return "shell"
    if name in _CODE_TOOLS:
        return "code"
    tokens = _name_tokens(name)
    if tokens & _RUN_NAME_TOKENS:
        if tokens & _SHELL_NAME_TOKENS:
            return "shell"
        if tokens & _CODE_NAME_TOKENS:
            return "code"
    return None


def _code_argument(arguments: dict) -> str:
    """코드 실행 도구의 인자에서 실행할 코드를 꺼낸다. 없으면 빈 문자열."""
    for key in _CODE_ARGUMENT_KEYS:
        value = arguments.get(key)
        if isinstance(value, (list, tuple)):
            value = "\n".join(str(item) for item in value)
        if value:
            return str(value)
    strings = [value for key, value in arguments.items() if key != "plan" and isinstance(value, str) and value.strip()]
    return strings[0] if len(strings) == 1 else ""


def _capture(action: Action) -> Action:
    """가로챈 행동의 내용을 읽어 SimulationManager가 볼 Action 필드를 채운다."""
    if action.action_type or action.code is not None:
        return action
    if action.tool_name:
        kind = _code_tool_kind(action.tool_name)
        if kind is None:
            return action
        args = action.tool_arguments
        code = _code_argument(args)
        if not code:
            # 코드 실행 도구인데 실행할 코드를 읽지 못했다. 빈 코드를 샌드박스에서 돌리지 않는다.
            return action.model_copy(update={"action_type": ActionType.TOOL_CALL})
        default_language = "javascript" if _name_tokens(action.tool_name) & _SCRIPT_NAME_TOKENS else "python"
        language = "shell" if kind == "shell" else str(args.get("language") or default_language)
        return action.model_copy(update={"code": code, "language": language})
    return action


class SimGuardAgentInterceptor:
    def __init__(self):
        self._manager    = SimulationManager()
        self._controller = DecisionController()

    async def intercept(self, action: Action) -> SimulationTrace:
        trace          = await self._manager.simulate(_capture(action))
        trace.decision = self._controller.decide(trace.risk_score)
        return trace

    async def intercept_output(
        self,
        messages: Sequence[dict],
        output: dict | str,
        tools: Sequence[dict] | Mapping[str, Sequence[str]] | None = None,
        shadow_executors: Mapping[str, ShadowToolExecutor] | None = None,
        execution_session: object = None,
        execution_session_factory: Callable[[], Awaitable[object]] | None = None,
    ) -> InterceptionResult:
        """타겟 LLM의 출력 하나를 에이전트가 실행하거나 사용자에게 보내기 전에 평가한다.

        어떤 에이전트, 어떤 데이터든 이 함수 하나로 같은 과정을 거친다.
          messages: 타겟이 LLM에 보낸 대화 (OpenAI 메시지 형식)
          output:   LLM의 출력 메시지 {"content", "tool_calls", "reasoning"} 또는 문자열
          tools:    타겟이 LLM에 준 도구 정의 목록, 또는 {도구 이름: 인자 이름 목록}

        출력 하나는 Plan 하나이고, Simulation Manager가 Plan이 실행하려는 내용에 따라 한 경로로
        보낸다. 도구 호출은 ToolCallSimulator, 코드는 ExecutionSimulator, 행동 없이 할 일을 서술한
        Plan은 BehaviorSimulator로 시뮬레이션한 뒤 각 Judge가 판정한다. 완성된 답변은
        BehaviorJudge만 거친다. 도구 호출이 여러 개면 하나라도 BLOCK일 때 Plan 전체가 BLOCK이고,
        판정 중 오류가 나도 BLOCK이다.
        """
        if isinstance(output, str):
            output = {"content": output}
        descriptions = None
        if tools is not None and not isinstance(tools, Mapping):
            descriptions = _tool_descriptions(list(tools))
            tools = declared_tools({"tools": list(tools)})
        result = InterceptionResult()
        try:
            for action in response_actions(list(messages), dict(output), tools, shadow_executors, descriptions):
                if action.code and action.execution_session is None:
                    if execution_session is None and execution_session_factory is not None:
                        execution_session = await execution_session_factory()
                    action = action.model_copy(update={"execution_session": execution_session})
                trace = await self.intercept(action)
                result.traces.append(trace)
                result.risk_score = max(result.risk_score, float(trace.risk_score))
                if trace.decision == Decision.BLOCK:
                    result.decision = Decision.BLOCK
                    return result
        except Exception as exc:
            result.decision, result.risk_score = Decision.BLOCK, 1.0
            result.error = f"{type(exc).__name__}: {str(exc)[:300]}"
        return result

    @staticmethod
    def _plan_actions(
        base: dict,
        plan_text: str,
        tools: Mapping[str, Sequence[str]],
        shadow_executor: ShadowToolExecutor | None,
    ) -> list[Action]:
        """Plan 내용을 Simulation Manager가 분류할 행동들로 나눈다."""
        code_in_plan = _code_in_plan(plan_text)
        code_actions: list[Action] = []
        tool_actions: list[Action] = []
        for call in plan_tool_calls(plan_text, tools):
            arguments = call["tool_arguments"]
            action = Action(
                **base,
                tool_name=call["tool_name"],
                tool_arguments=arguments if arguments is not None else {"plan": plan_text},
            )
            captured = _capture(action)
            if captured.code:
                if all(captured.code != known.code for known in code_actions):
                    code_actions.append(captured)
            elif captured.action_type == ActionType.TOOL_CALL:
                if not code_in_plan:
                    tool_actions.append(captured)
            elif arguments is not None and shadow_executor is not None:
                tool_actions.append(action.model_copy(update={"shadow_executor": shadow_executor}))
            else:
                tool_actions.append(action)
        if code_in_plan and not code_actions:
            code_actions.append(Action(**base, **code_in_plan))
        return (code_actions + tool_actions) or [Action(**base, action_type=ActionType.CONVERSATIONAL)]



def declared_tools(request_body: dict) -> dict[str, list[str]]:
    """LLM 요청의 도구 정의에서 {도구 이름: 인자 이름 목록}을 읽는다."""
    tools: dict[str, list[str]] = {}
    definitions = [item.get("function") or item for item in request_body.get("tools") or [] if isinstance(item, dict)]
    definitions += [item for item in request_body.get("functions") or [] if isinstance(item, dict)]
    for definition in definitions:
        if isinstance(definition, dict) and isinstance(definition.get("name"), str):
            schema = definition.get("parameters") or {}
            properties = schema.get("properties") if isinstance(schema, dict) else None
            tools[definition["name"]] = list(properties) if isinstance(properties, dict) else []
    return tools


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part["text"] for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def _tool_call_arguments(arguments) -> dict:
    if isinstance(arguments, dict):
        return dict(arguments)
    if not arguments:
        return {}
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    return {"raw": arguments}


def _reasoning_text(message: dict) -> str:
    """LLM 응답의 reasoning 필드를 읽는다."""
    for key in _REASONING_KEYS:
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    details = message.get("reasoning_details")
    if not isinstance(details, list):
        return ""
    parts = [
        item.get("text") or item.get("summary")
        for item in details
        if isinstance(item, dict)
    ]
    return "\n".join(part for part in parts if isinstance(part, str) and part.strip())


def _describes_future_action(text: str) -> bool:
    """코드 블록을 뺀 본문이 앞으로 할 행동을 서술하면 Plan이다."""
    return bool(_FUTURE_ACTION_RE.search(_FENCE_RE.sub("", text)))


def _render_message(message: dict, result_limit: int | None = None) -> str:
    role = str(message.get("role") or "unknown")
    text = _THINK_RE.sub("", _message_text(message.get("content"))).strip()
    if role in {"tool", "function"}:
        name = message.get("name") or message.get("tool_call_id") or ""
        if result_limit is not None and len(text) > 2 * result_limit:
            omitted = len(text) - 2 * result_limit
            text = f"{text[:result_limit]}\n...[{omitted} characters omitted]...\n{text[-result_limit:]}"
        header = f"[tool result: {name}]" if name else "[tool result]"
        return f"{header}\n{text}"
    lines = [f"[{role}]"] + ([text] if text else [])
    calls = list(message.get("tool_calls") or [])
    if message.get("function_call"):
        calls.append({"function": message["function_call"]})
    for call in calls:
        function = call.get("function") or call.get("custom") or {}
        arguments = function.get("arguments", function.get("input"))
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False) if arguments else ""
        lines.append(f"-> call {function.get('name') or call.get('name')}({arguments})")
    return "\n".join(lines)


def render_conversation(messages: Sequence[dict], limit: int = _MAX_CONVERSATION_CHARS) -> str:
    """타겟 에이전트의 모델이 받은 입력(시스템 프롬프트, 이전 턴을 포함한 대화, 도구 호출과 결과)을
    순서대로 적는다.

    전체가 limit을 넘을 때만 줄인다. 먼저 오래된 도구 결과를 앞뒤만 남기고 줄이고, 그래도 넘으면
    시스템 프롬프트와 사용자 메시지는 남긴 채 오래된 에이전트 출력과 도구 결과를 뺀다.
    """
    messages = [item for item in messages if isinstance(item, dict)]
    blocks = [_render_message(item) for item in messages]

    def size() -> int:
        return sum(len(block) + 2 for block in blocks if block)

    for index, message in enumerate(messages):
        if size() <= limit:
            break
        if message.get("role") in {"tool", "function"}:
            blocks[index] = _render_message(message, _SHORTENED_RESULT_CHARS)
    for index, message in enumerate(messages[:-1]):
        if size() <= limit:
            break
        if message.get("role") not in {"system", "developer", "user"}:
            blocks[index] = ""
    text = "\n\n".join(block for block in blocks if block)
    return text if len(text) <= limit else text[-limit:]


def _tool_descriptions(tools: Sequence[dict]) -> dict[str, str]:
    """도구 정의 목록에서 {도구 이름: 설명과 인자 스키마}를 읽는다."""
    descriptions: dict[str, str] = {}
    for item in tools:
        definition = (item.get("function") or item) if isinstance(item, dict) else None
        if isinstance(definition, dict) and isinstance(definition.get("name"), str):
            text = str(definition.get("description") or "")
            schema = definition.get("parameters")
            if schema:
                text = f"{text}\nParameters: {json.dumps(schema, ensure_ascii=False)}".strip()
            descriptions[definition["name"]] = text
    return descriptions


def _target_plan(plan_text: str, user_request: str) -> AgentPlan:
    return AgentPlan(
        agent_role="Target agent",
        agent_persona="general-purpose agent",
        sub_queries=[plan_text],
        research_focus=user_request,
    )


def _behavior_plan_action(parts: Sequence[str], user_request: str, conversation: str | None) -> Action:
    """실행되지 않는 Plan 텍스트를 BehaviorSimulator로 보낼 행동으로 만든다."""
    return Action(
        user_request=user_request,
        agent_plan=_target_plan("\n\n".join(parts)[-_MAX_PLAN_TEXT:], user_request),
        conversation=conversation,
        action_type=ActionType.CONVERSATIONAL,
    )


def response_actions(
    messages: list[dict],
    message: dict,
    tools: Mapping[str, Sequence[str]] | None = None,
    shadow_executors: Mapping[str, ShadowToolExecutor] | None = None,
    descriptions: Mapping[str, str] | None = None,
) -> list[Action]:
    """타겟 LLM의 출력 하나(Plan 하나)를 Simulation Manager가 라우팅할 행동으로 바꾼다.

    Plan은 실행하려는 내용에 따라 한 경로로만 간다.
      도구 호출              -> 호출마다 ToolCallSimulator (코드 실행 도구는 ExecutionSimulator)
      코드, 셸               -> ExecutionSimulator
      행동 없이 할 일을 서술  -> BehaviorSimulator
      완성된 답변             -> 시뮬레이션할 것이 없으므로 BehaviorJudge만
      허용된 도구·코드 실행 뒤의 텍스트 답변 -> 이미 판정한 행동의 결과이므로 판정하지 않음
    Plan의 텍스트와 reasoning은 따로 시뮬레이션하지 않고 행동의 agent_plan으로 붙는다.
    """
    executors = shadow_executors or {}
    signatures = [f"{name}({', '.join(parameters)})" for name, parameters in (tools or {}).items()]
    messages = [item for item in messages if isinstance(item, dict)]
    user_positions = [index for index, item in enumerate(messages) if item.get("role") == "user"]
    last_user = user_positions[-1] if user_positions else -1
    source = messages[last_user] if user_positions else (messages[0] if messages else {})
    user_request = _message_text(source.get("content"))

    # 시뮬레이터와 Judge는 타겟의 모델이 이 출력을 내기 전에 받은 입력 전체를 같이 본다.
    conversation = render_conversation(messages) or None

    content = _message_text(message.get("content"))
    text = _THINK_RE.sub("", content).strip()
    thoughts = [_reasoning_text(message), *(block.strip() for block in _THINK_RE.findall(content))]
    reasoning = "\n\n".join(thought for thought in thoughts if thought)
    plan_text = "\n\n".join(part for part in (reasoning, text) if part)[-_MAX_PLAN_TEXT:]
    base = {
        "user_request": user_request,
        "agent_plan": _target_plan(plan_text, user_request) if plan_text else None,
        "available_tools": signatures,
        "conversation": conversation,
    }

    # 도구를 호출하는 Plan
    calls = list(message.get("tool_calls") or [])
    if message.get("function_call"):
        calls.append({"function": message["function_call"]})
    if calls:
        actions = []
        for call in calls:
            function = call.get("function") or call.get("custom") or {}
            name = str(function.get("name") or call.get("name") or "unknown_tool")
            arguments = _tool_call_arguments(function.get("arguments", function.get("input")))
            # shadow가 없는 SQL 도구 호출은 SimGuardAgent 내부 DB에서 실행해 본다.
            shadow = executors.get(name) or internal_database_for(messages, descriptions, arguments)
            action = _capture(Action(
                **base, tool_name=name, tool_arguments=arguments, shadow_executor=shadow,
                tool_description=(descriptions or {}).get(name),
            ))
            if action.shadow_executor is None and action.code is None:
                # 이 프로세스에 실제 도구가 있으면 GET 전용 모드로 실행해 본다. GET·HEAD가 아닌 요청이나
                # 다른 부작용이 나오면 멈추고 내부 모델이 예상한다.
                live = live_read_executor_for(name, arguments)
                if live is not None:
                    action = action.model_copy(update={"shadow_executor": live})
            if action.code is None and action.action_type is None:
                action = action.model_copy(update={"action_type": ActionType.TOOL_CALL})
            actions.append(action)
        return actions

    if not text and not reasoning:
        return []

    # 텍스트로 도구나 코드를 실행하겠다고 쓴 Plan (텍스트 형식으로 행동하는 에이전트)
    if text and tools and plan_tool_calls(text, tools):
        actions = []
        for action in SimGuardAgentInterceptor._plan_actions(base, text, tools, None):
            known_arguments = action.tool_arguments != {"plan": text}
            if action.tool_name in executors and action.code is None and action.action_type is None and known_arguments:
                action = action.model_copy(update={"shadow_executor": executors[action.tool_name]})
            actions.append(action)
        return actions
    if text and not tools and _FENCE_RE.search(text):
        code = _code_in_plan(text)
        if code:
            return [Action(**base, **code)]

    # 실행할 행동이 없는 출력. 이 작업에서 이미 허용되어 실행된 도구·코드가 있으면, 그 결과를 전하는
    # 답변이므로 다시 판정하지 않는다.
    if any(item.get("role") in {"tool", "function"} for item in messages[last_user + 1:]):
        return []
    if text and not _describes_future_action(text):
        return [Action(
            user_request=user_request,
            final_response=text,
            conversation=conversation,
            action_type=ActionType.FINAL_RESPONSE,
        )]
    if (text and _describes_future_action(text)) or (not text and _describes_future_action(reasoning)):
        return [_behavior_plan_action([plan_text], user_request, conversation)]
    return []
