"""
AgentDojo 연결부.

SimGuardAgent Interceptor를 AgentDojo 파이프라인 요소로 끼운다. LLM이 출력을 내면
도구가 실행되거나 답변이 나가기 전에 판정하고, BLOCK이면 그 출력을 차단 문구로 바꾼다.
도구 호출이 사라지므로 ToolsExecutor는 아무것도 실행하지 않는다.

  from simguard.integrations.agentdojo import with_simguard
  pipeline = with_simguard(pipeline)          # LLM 요소마다 바로 뒤에 넣는다

  # 또는 직접 배치
  pipeline = AgentPipeline([
      SystemMessage(...), InitQuery(), llm, SimGuardAgentInterceptor(),
      ToolsExecutionLoop([ToolsExecutor(), llm, SimGuardAgentInterceptor()]),
  ])

도구 결과는 내부 모델이 앞 단계의 실제 도구 결과와 인자를 보고 예상한다. AgentDojo 환경은
ALLOW된 뒤 ToolsExecutor가 실행할 때만 바뀐다.
판정 결과는 extra_args["simguard_results"]에 InterceptionResult로 쌓인다.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionReturnType, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, get_text_content_as_str, text_content_block_from_string

from simguard.interceptor import BLOCKED_MESSAGE
from simguard.interceptor import SimGuardAgentInterceptor as _Interceptor
from simguard.execution_environment import ExecutionEnvironment
from simguard.models import Decision

_LLM_MODULE = "agentdojo.agent_pipeline.llms"


def _run(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _tool_call(call: Any, position: int) -> dict:
    return {
        "id": call.id or f"call_{position}",
        "type": "function",
        "function": {"name": call.function, "arguments": json.dumps(dict(call.args), ensure_ascii=False, default=str)},
    }


def _to_openai(message: ChatMessage) -> dict:
    """AgentDojo 메시지를 OpenAI 메시지 형식으로 바꾼다."""
    text = get_text_content_as_str(message.get("content") or [])
    if message["role"] == "tool":
        error = message.get("error")
        return {
            "role": "tool",
            "tool_call_id": message.get("tool_call_id"),
            "content": f"{text}\nError: {error}".strip() if error else text,
        }
    converted = {"role": message["role"], "content": text}
    if message["role"] == "assistant" and message.get("tool_calls"):
        converted["tool_calls"] = [_tool_call(call, index) for index, call in enumerate(message["tool_calls"])]
    return converted


def _tool_definition(function: Any) -> dict:
    return {
        "type": "function",
        "function": {
            "name": function.name,
            "description": function.description,
            "parameters": function.parameters.model_json_schema(),
        },
    }


class SimGuardAgentInterceptor(BasePipelineElement):
    """LLM 요소 바로 뒤에 두는 SimGuardAgent Interceptor."""

    name = "simguard_agent_interceptor"

    def __init__(
        self,
        tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str,
        execution_environment: ExecutionEnvironment | None = None,
    ) -> None:
        self.tool_output_formatter = tool_output_formatter
        self.execution_environment = execution_environment

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        extra_args = dict(extra_args or {})
        if not messages or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args

        history = [_to_openai(message) for message in messages[:-1]]
        output = _to_openai(messages[-1])
        tools = [_tool_definition(function) for function in runtime.functions.values()]
        # 판정기의 LLM 클라이언트는 이벤트 루프에 묶이므로 호출마다 새로 만든다.
        result = _run(_Interceptor().intercept_output(
            history, output, tools, execution_environment=self.execution_environment,
            retrieved_context=extra_args.get("retrieved_context"),
        ))
        extra_args["simguard_results"] = [*extra_args.get("simguard_results", []), result]
        if result.decision == Decision.BLOCK:
            blocked = ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string(BLOCKED_MESSAGE)],
                tool_calls=None,
            )
            messages = [*messages[:-1], blocked]
        return query, runtime, env, messages, extra_args


def _is_llm(element: BasePipelineElement) -> bool:
    kind = type(element)
    return kind.__module__.startswith(_LLM_MODULE) and "ToolFilter" not in kind.__name__


def _guard(
    elements: Sequence[BasePipelineElement],
    make: Callable[[], SimGuardAgentInterceptor],
    is_llm: Callable[[BasePipelineElement], bool],
) -> list[BasePipelineElement]:
    guarded: list[BasePipelineElement] = []
    for position, element in enumerate(elements):
        if isinstance(element, ToolsExecutionLoop):
            element.elements = _guard(element.elements, make, is_llm)
        guarded.append(element)
        following = elements[position + 1] if position + 1 < len(elements) else None
        if is_llm(element) and not isinstance(following, SimGuardAgentInterceptor):
            guarded.append(make())
    return guarded


def with_simguard(
    pipeline: AgentPipeline,
    tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str,
    is_llm: Callable[[BasePipelineElement], bool] = _is_llm,
    execution_environment: ExecutionEnvironment | None = None,
) -> AgentPipeline:
    """파이프라인의 LLM 요소마다 바로 뒤에 SimGuardAgent Interceptor를 넣는다.

    AgentDojo 기본 LLM 요소는 자동으로 찾는다. 직접 만든 LLM 요소를 쓰면 is_llm으로 알려 준다.
    execution_environment: 에이전트가 코드를 실행하는 작업 파일. 주면 코드 행동을 그 파일 위에서 실행해 본다.
    파이프라인의 다른 요소와 이름은 그대로 둔다.
    """
    def make() -> SimGuardAgentInterceptor:
        return SimGuardAgentInterceptor(tool_output_formatter, execution_environment)

    guarded = _guard(pipeline.elements, make, is_llm)
    if not any(isinstance(element, SimGuardAgentInterceptor) for element in guarded):
        raise ValueError("no LLM element found in the pipeline; pass is_llm to identify it")
    pipeline.elements = guarded
    return pipeline
