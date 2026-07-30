"""
AgentDojo adapter for SimGuardAgent tool-call defense experiments.

This keeps AgentDojo's benchmark suites, attacks, tasks, tools, and evaluators
intact, but replaces the tool executor with a SimGuard-aware executor when
``--defense simguard`` or ``--defense simguard_tool_filter`` is selected.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Literal

import yaml

_ADAPTER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ADAPTER_DIR.parents[1]
for _path in (_ADAPTER_DIR / "src", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from agentdojo.agent_pipeline.agent_pipeline import (  # noqa: E402
    TOOL_FILTER_PROMPT,
    AgentPipeline,
    PipelineConfig,
    get_llm,
    load_system_message,
)
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement  # noqa: E402
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage  # noqa: E402
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM, OpenAILLMToolFilter  # noqa: E402
from agentdojo.agent_pipeline.tool_execution import (  # noqa: E402
    ToolsExecutionLoop,
    ToolsExecutor,
    is_string_list,
    tool_result_to_str,
)
from agentdojo.benchmark import (  # noqa: E402
    SuiteResults,
    benchmark_suite_with_injections,
    benchmark_suite_without_injections,
)
from agentdojo.functions_runtime import (  # noqa: E402
    EmptyEnv,
    Env,
    FunctionReturnType,
    FunctionsRuntime,
)
from agentdojo.logging import OutputLogger  # noqa: E402
from agentdojo.models import MODEL_PROVIDERS, ModelsEnum  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite, get_suites  # noqa: E402
from agentdojo.types import ChatMessage, ChatToolResultMessage, text_content_block_from_string  # noqa: E402


SIMGUARD_DEFENSES = {"simguard", "simguard_tool_filter"}
BUILTIN_DEFENSES = {
    "none",
    "tool_filter",
    "transformers_pi_detector",
    "spotlighting_with_delimiting",
    "repeat_user_prompt",
}
DEFENSE_CHOICES = sorted(BUILTIN_DEFENSES | SIMGUARD_DEFENSES)

MODE_TO_DEFENSE = {
    "baseline": "none",
    "none": "none",
    "simguard": "simguard",
    "simguard_tool_filter": "simguard_tool_filter",
    "tool_filter": "tool_filter",
    "repeat_user_prompt": "repeat_user_prompt",
    "spotlighting_with_delimiting": "spotlighting_with_delimiting",
    "transformers_pi_detector": "transformers_pi_detector",
}
MODE_CHOICES = sorted(MODE_TO_DEFENSE)
BENCHMARK_CHOICES = ["workspace", "banking", "slack", "travel", "all"]

OPENAI_BASE_URL_CANDIDATES = [
    "http://localhost:10000/v1",
    "http://localhost:8000/v1",
    "http://localhost:8001/v1",
    "http://localhost:8002/v1",
    "http://localhost:8003/v1",
]


def _discover_openai_base_url(explicit_url: str | None) -> str:
    if explicit_url:
        return explicit_url
    import urllib.request

    for base_url in OPENAI_BASE_URL_CANDIDATES:
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=2) as response:
                if response.status == 200:
                    print(f"[agentdojo] auto-detected OpenAI-compatible endpoint: {base_url}")
                    return base_url
        except Exception:
            continue
    candidates = ", ".join(OPENAI_BASE_URL_CANDIDATES)
    raise RuntimeError(
        "No OpenAI-compatible endpoint found. "
        f"Tried: {candidates}. Pass --openai-base-url http://HOST:PORT/v1."
    )


def _discover_model_id(base_url: str, explicit_model_id: str | None) -> str:
    if explicit_model_id:
        return explicit_model_id
    import urllib.request

    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not discover model id from {base_url}/models: {exc}") from exc

    models = payload.get("data") or []
    if not models or not models[0].get("id"):
        raise RuntimeError(f"No models returned by {base_url}/models")
    model_id = models[0]["id"]
    print(f"[agentdojo] auto-detected model id: {model_id}")
    return model_id


def _install_openai_completion_limits(max_tokens: int | None, timeout: float | None) -> None:
    if max_tokens is None and timeout is None:
        return
    import agentdojo.agent_pipeline.llms.openai_llm as openai_llm_module

    def limited_chat_completion_request(
        client,
        model,
        messages,
        tools,
        reasoning_effort,
        temperature=0.0,
    ):
        kwargs = {
            "model": model,
            "messages": messages,
            "tools": tools or openai_llm_module.NOT_GIVEN,
            "tool_choice": "auto" if tools else openai_llm_module.NOT_GIVEN,
            "temperature": temperature if temperature is not None else openai_llm_module.NOT_GIVEN,
            "reasoning_effort": reasoning_effort or openai_llm_module.NOT_GIVEN,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if timeout is not None:
            kwargs["timeout"] = timeout
        return client.chat.completions.create(**kwargs)

    openai_llm_module.chat_completion_request = limited_chat_completion_request

    def limited_tool_filter_query(self, query, runtime, env=openai_llm_module.EmptyEnv(), messages=[], extra_args={}):
        messages = [
            *messages,
            openai_llm_module.ChatUserMessage(
                role="user",
                content=[openai_llm_module.text_content_block_from_string(self.prompt)],
            ),
        ]
        openai_messages = [openai_llm_module._message_to_openai(message, self.model) for message in messages]
        openai_tools = [openai_llm_module._function_to_openai(tool) for tool in runtime.functions.values()]
        kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "tools": openai_tools or openai_llm_module.NOT_GIVEN,
            "tool_choice": "none",
            "temperature": self.temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = min(max_tokens, 512)
        if timeout is not None:
            kwargs["timeout"] = timeout
        completion = self.client.chat.completions.create(**kwargs)
        output = openai_llm_module._openai_to_assistant_message(completion.choices[0].message)

        new_tools = {}
        for tool_name, tool in runtime.functions.items():
            if output["content"] is not None and tool_name in openai_llm_module.get_text_content_as_str(output["content"]):
                new_tools[tool_name] = tool

        runtime.update_functions(new_tools)
        messages = [*messages, output]
        return query, runtime, env, messages, extra_args

    openai_llm_module.OpenAILLMToolFilter.query = limited_tool_filter_query


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # AgentDojo's executor is synchronous. If a future caller invokes it from an
    # existing event loop, use a short-lived private loop rather than re-entering.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _normalize_args(args: object) -> dict:
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except Exception:
            return {"raw": args}
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    return {"raw": args}


class SimGuardToolsExecutor(BasePipelineElement):
    """AgentDojo executor that judges concrete shadow execution consequences."""

    name = "simguard_tools_executor"

    def __init__(self, tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str) -> None:
        self.output_formatter = tool_output_formatter

    def _guard_and_execute(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env,
        tool_name: str,
        tool_args: dict,
    ):
        from simguard import RuntimeShadowToolExecutor, SimGuardInterceptor

        shadow_executor = RuntimeShadowToolExecutor(runtime, env, self.output_formatter)
        interceptor = SimGuardInterceptor()
        return _run_async(
            interceptor.guard_shadow_tool_call(
                user_request=query,
                tool_name=tool_name,
                tool_arguments=tool_args,
                shadow_executor=shadow_executor,
            )
        )

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if extra_args is None:
            extra_args = {}
        if len(messages) == 0:
            return query, runtime, env, messages, extra_args
        if messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        if messages[-1]["tool_calls"] is None or len(messages[-1]["tool_calls"]) == 0:
            return query, runtime, env, messages, extra_args

        from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME

        tool_call_results = []
        simguard_events = extra_args.setdefault("simguard_events", [])
        for tool_call in messages[-1]["tool_calls"]:
            tool_name = tool_call.function
            tool_args = _normalize_args(tool_call.args)

            if tool_name == EMPTY_FUNCTION_NAME:
                tool_call_results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string("")],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error="Empty function name provided. Provide a valid function name.",
                    )
                )
                continue

            if tool_name not in (tool.name for tool in runtime.functions.values()):
                tool_call_results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string("")],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error=f"Invalid tool {tool_name} provided.",
                    )
                )
                continue

            for arg_k, arg_v in list(tool_args.items()):
                if isinstance(arg_v, str) and is_string_list(arg_v):
                    from ast import literal_eval

                    tool_args[arg_k] = literal_eval(arg_v)
            tool_call.args = tool_args

            guarded = self._guard_and_execute(query, runtime, env, tool_name, tool_args)
            trace = guarded.trace
            risk_score = float(getattr(trace, "risk_score", 0.0) or 0.0)
            decision_value = getattr(trace.decision, "value", str(trace.decision))
            simguard_events.append(
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "decision": decision_value,
                    "risk_score": risk_score,
                    "committed": guarded.committed,
                    "shadow_error": guarded.shadow.error,
                    "state_diff": guarded.shadow.state_diff[:1000],
                }
            )
            print(f"[SimGuard][AgentDojo] {decision_value} risk={risk_score:.2f} tool={tool_name}")

            tool_call_results.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string(guarded.result)],
                    tool_call_id=tool_call.id,
                    tool_call=tool_call,
                    error=guarded.error,
                )
            )

        return query, runtime, env, [*messages, *tool_call_results], extra_args


def build_pipeline(
    model: str,
    model_id: str | None,
    defense: str,
    tool_delimiter: str,
    system_message_name: str | None,
    system_message: str | None,
    tool_output_format: Literal["yaml", "json"] | None,
) -> AgentPipeline:
    defense = "none" if defense is None else defense
    loaded_system_message = system_message if system_message is not None else load_system_message(system_message_name)
    provider = MODEL_PROVIDERS[ModelsEnum(model)]
    llm = get_llm(provider, model, model_id, tool_delimiter)
    llm_call_model = getattr(llm, "model", None) or model_id
    pipeline_model_name = model if isinstance(model, str) else getattr(llm, "name", None)

    tool_output_formatter = partial(tool_result_to_str, dump_fn=json.dumps) if tool_output_format == "json" else tool_result_to_str
    system_message_component = SystemMessage(loaded_system_message)
    init_query_component = InitQuery()

    if defense not in SIMGUARD_DEFENSES:
        if defense == "tool_filter":
            if not isinstance(llm, OpenAILLM):
                raise ValueError("tool_filter requires an OpenAI-compatible/OpenAI LLM")
            tools_loop = ToolsExecutionLoop([ToolsExecutor(tool_output_formatter), llm])
            pipeline = AgentPipeline([
                system_message_component,
                init_query_component,
                OpenAILLMToolFilter(TOOL_FILTER_PROMPT, llm.client, llm_call_model),
                llm,
                tools_loop,
            ])
            pipeline.name = f"{pipeline_model_name}-{defense}"
            return pipeline
        return AgentPipeline.from_config(
            PipelineConfig(
                llm=model,
                model_id=model_id,
                defense=None if defense == "none" else defense,
                tool_delimiter=tool_delimiter,
                system_message_name=system_message_name,
                system_message=system_message,
                tool_output_format=tool_output_format,
            )
        )

    tools_loop = ToolsExecutionLoop([SimGuardToolsExecutor(tool_output_formatter), llm])

    elements = [system_message_component, init_query_component]
    if defense == "simguard_tool_filter":
        if not isinstance(llm, OpenAILLM):
            raise ValueError("simguard_tool_filter requires an OpenAI-compatible/OpenAI LLM")
        if llm_call_model is None:
            raise ValueError("simguard_tool_filter requires a named model")
        elements.append(OpenAILLMToolFilter(TOOL_FILTER_PROMPT, llm.client, llm_call_model))

    elements.extend([llm, tools_loop])
    pipeline = AgentPipeline(elements)
    pipeline.name = f"{pipeline_model_name}-{defense}"
    return pipeline



def _resolve_suites(args: argparse.Namespace) -> tuple[str, ...]:
    if args.suite:
        return tuple(args.suite)
    if args.benchmark == "all":
        return tuple(get_suites(args.benchmark_version).keys())
    return (args.benchmark,)


def _write_summary(results: SuiteResults, args: argparse.Namespace) -> dict:
    utility_values = list(results["utility_results"].values())
    injection_values = list(results["security_results"].values())
    summary = {
        "agent": "agentdojo",
        "benchmark": args.benchmark,
        "suites": list(_resolve_suites(args)),
        "mode": args.mode,
        "label": args.label,
        "defense": args.defense,
        "attack": args.attack,
        "benchmark_version": args.benchmark_version,
        "limit": args.limit,
        "workers": args.workers,
        "model": args.model,
        "model_id": args.model_id,
        "openai_base_url": args.openai_base_url,
        "logdir": str(args.logdir),
        "num_cases": len(utility_values),
        "utility_success": sum(v is True for v in utility_values),
        "utility_rate": (sum(v is True for v in utility_values) / len(utility_values)) if utility_values else None,
        "injection_success": sum(v is True for v in injection_values),
        "asr": (sum(v is True for v in injection_values) / len(injection_values)) if injection_values else None,
    }
    if args.results:
        path = Path(args.results)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary

def run_benchmark(args: argparse.Namespace) -> SuiteResults:
    args.openai_base_url = _discover_openai_base_url(args.openai_base_url)
    os.environ["OPENAI_COMPATIBLE_BASE_URL"] = args.openai_base_url
    os.environ.setdefault("OPENAI_BASE_URL", args.openai_base_url)
    args.model_id = _discover_model_id(args.openai_base_url, args.model_id)
    if args.model_id:
        os.environ.setdefault("SHADOW_MODEL", args.model_id)
        os.environ.setdefault("JUDGE_MODEL", args.model_id)
    if args.openai_api_key:
        os.environ["OPENAI_COMPATIBLE_API_KEY"] = args.openai_api_key
        os.environ.setdefault("OPENAI_API_KEY", args.openai_api_key)
    _install_openai_completion_limits(args.llm_max_tokens, args.llm_timeout)

    suites = _resolve_suites(args)
    combined = SuiteResults(utility_results={}, security_results={}, injection_tasks_utility_results={})

    for suite_name in suites:
        suite = get_suite(args.benchmark_version, suite_name)
        selected_user_tasks = tuple(args.user_task)
        if args.limit is not None:
            if selected_user_tasks:
                selected_user_tasks = selected_user_tasks[: args.limit]
            else:
                selected_user_tasks = tuple(list(suite.user_tasks)[: args.limit])
            print(f"Limiting suite={suite_name} to {len(selected_user_tasks)} user tasks")
        pipeline = build_pipeline(
            model=args.model,
            model_id=args.model_id,
            defense=args.defense,
            tool_delimiter=args.tool_delimiter,
            system_message_name=args.system_message_name,
            system_message=args.system_message,
            tool_output_format=args.tool_output_format,
        )

        print(f"Running AgentDojo suite={suite_name} model={args.model} defense={args.defense} attack={args.attack}")
        with OutputLogger(str(args.logdir)):
            if args.attack is None:
                result = benchmark_suite_without_injections(
                    pipeline,
                    suite,
                    logdir=args.logdir,
                    force_rerun=args.force_rerun,
                    user_tasks=selected_user_tasks or None,
                    benchmark_version=args.benchmark_version,
                )
            else:
                from agentdojo.attacks.attack_registry import load_attack

                attacker = load_attack(args.attack, suite, pipeline)
                result = benchmark_suite_with_injections(
                    pipeline,
                    suite,
                    attacker,
                    logdir=args.logdir,
                    force_rerun=args.force_rerun,
                    user_tasks=selected_user_tasks or None,
                    injection_tasks=args.injection_task or None,
                    benchmark_version=args.benchmark_version,
                )

        for key, value in result["utility_results"].items():
            combined["utility_results"][(f"{suite_name}_{key[0]}", key[1])] = value
        for key, value in result["security_results"].items():
            combined["security_results"][(f"{suite_name}_{key[0]}", key[1])] = value
        for key, value in result["injection_tasks_utility_results"].items():
            combined["injection_tasks_utility_results"][f"{suite_name}_{key}"] = value

    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AgentDojo with optional SimGuard tool-call defense.")
    parser.add_argument("--benchmark", default="workspace", choices=BENCHMARK_CHOICES, help="AgentDojo suite to run, or all.")
    parser.add_argument("--mode", default="simguard", choices=MODE_CHOICES, help="baseline, simguard, or a defense method.")
    parser.add_argument("--label", default="attack", choices=["attack", "benign", "all"], help="Compatibility label. attack=runs injection benchmark, benign=runs no-attack utility, all=runs injection benchmark with utility side results.")
    parser.add_argument("-s", "--suite", action="append", default=[], help="Suite name. Can be repeated.")
    parser.add_argument("-ut", "--user-task", action="append", default=[], help="User task ID. Can be repeated.")
    parser.add_argument("-it", "--injection-task", action="append", default=[], help="Injection task ID. Can be repeated.")
    parser.add_argument("--attack", default="tool_knowledge", help="AgentDojo attack name. Use 'none' to disable.")
    parser.add_argument("--defense", default=None, choices=DEFENSE_CHOICES, help="Legacy alias. Prefer --mode.")
    parser.add_argument("--model", default="openai-compatible", choices=[value for value in ModelsEnum])
    parser.add_argument("--model-id", default=os.getenv("AGENTDOJO_MODEL_ID"))
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_COMPATIBLE_BASE_URL"))
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_COMPATIBLE_API_KEY", "dummy"))
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--tool-delimiter", default="tool")
    parser.add_argument("--system-message-name", default=None)
    parser.add_argument("--system-message", default=None)
    parser.add_argument("--tool-output-format", choices=["yaml", "json"], default=None)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N user tasks in each selected suite.")
    parser.add_argument("--llm-max-tokens", type=int, default=int(os.getenv("AGENTDOJO_LLM_MAX_TOKENS", "1024")))
    parser.add_argument("--llm-timeout", type=float, default=float(os.getenv("AGENTDOJO_LLM_TIMEOUT", "300")))
    parser.add_argument("--logdir", type=Path, default=None, help="Trace log directory. Defaults to runs/agentdojo_<benchmark>_<mode>.")
    parser.add_argument("--results", "--result", dest="results", default=None, help="Summary JSON output path.")
    parser.add_argument("--workers", type=int, default=1, help="Accepted for CLI compatibility. AgentDojo runs one suite process here.")
    parser.add_argument("-f", "--force-rerun", action="store_true")
    args = parser.parse_args()
    if args.defense is None:
        args.defense = MODE_TO_DEFENSE[args.mode]
    else:
        args.mode = "baseline" if args.defense == "none" else args.defense
    if args.logdir is None:
        args.logdir = Path(f"./runs/agentdojo_{args.benchmark}_{args.mode}_{args.label}")
    if args.label == "benign":
        args.attack = "none"
    if args.attack == "none":
        args.attack = None
    if len(args.suite) > 1 and args.user_task:
        parser.error("--user-task can only be used with one --suite")
    return args


def main() -> None:
    args = parse_args()
    results = run_benchmark(args)
    summary = _write_summary(results, args)
    if summary["utility_rate"] is not None:
        print(f"Average utility: {summary['utility_rate'] * 100:.2f}%")
    if summary["asr"] is not None and args.attack is not None:
        print(f"ASR / injection success: {summary['asr'] * 100:.2f}%")
    if args.results:
        print(f"Summary written to: {args.results}")


if __name__ == "__main__":
    main()
