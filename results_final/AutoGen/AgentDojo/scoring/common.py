"""Shared pieces for running benchmarks on the real AutoGen agent (AgentChat AssistantAgent) with SimGuardAgent.

Same connection as agents/autogen/run_autogen_simguard.py: simguard.protect(providers=["openai"]) is installed before
AutoGen is imported, so every model output is judged before AutoGen sees it; AutoGen's code is not changed.
Target = simulator = qwen/qwen3.5-9b (thinking off), judge = openai/gpt-oss-120b. Run through
experiments/full_runs/launch.py so every LLM call is logged with its role.
"""
from __future__ import annotations

import contextvars
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TARGET = os.getenv("AGENTDOJO_MODEL_ID", "openrouter/qwen/qwen3.5-9b")
BASE = (os.getenv("OPENAI_COMPATIBLE_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
KEY = os.getenv("OPENAI_COMPATIBLE_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
MODE = os.getenv("AUTOGEN_MODE", "simguard")  # simguard | none

_judged: contextvars.ContextVar[list | None] = contextvars.ContextVar("autogen_judged", default=None)


def _on_result(result) -> None:
    items = _judged.get()
    if items is not None:
        items.append(result)


def install(**protect_kwargs: Any) -> None:
    """Connect SimGuardAgent (call before importing AutoGen). MODE=none leaves the agent undefended."""
    if MODE == "simguard":
        import simguard
        simguard.protect(providers=["openai"], on_result=_on_result, target_model=TARGET, **protect_kwargs)


def make_client():
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    return OpenAIChatCompletionClient(
        model=TARGET, base_url=BASE, api_key=KEY, temperature=0,
        model_info={"vision": False, "function_calling": True, "json_output": True, "family": "unknown",
                    "structured_output": False},
        extra_body={"reasoning": {"effort": "none", "exclude": True}},  # target thinking off, as in the other runs
    )


async def run_agent(task: str, system: str | None = None, tools: list | None = None, max_tool_iterations: int = 10) -> dict:
    """Run one AutoGen AssistantAgent episode. Returns messages, final text, SimGuardAgent verdicts and timing."""
    from autogen_agentchat.agents import AssistantAgent
    judged: list = []
    token = _judged.set(judged)
    client = make_client()
    started, error, messages = time.time(), None, []
    try:
        kwargs = {"model_client": client, "tools": tools or [], "max_tool_iterations": max_tool_iterations}
        if system:
            kwargs["system_message"] = system
        agent = AssistantAgent("assistant", **kwargs)
        result = await agent.run(task=task)
        messages = list(result.messages)
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        await client.close()
        _judged.reset(token)
    traces = [t for item in judged for t in item.traces]
    return {
        "messages": messages, "final": str(messages[-1].content) if messages else "", "error": error,
        "blocked": any(item.decision.value == "BLOCK" for item in judged) if MODE == "simguard" else None,
        "judgments": len(judged),
        # tool calls the model proposed, with SimGuardAgent's decision; a blocked call never reaches AutoGen's messages
        "proposed": [{"tool": t.action.tool_name, "arguments": t.action.tool_arguments, "decision": t.decision.value}
                     for t in traces if t.action.tool_name],
        "events": [f"{e.type}: {e.evidence[:200]}" for t in traces if t.decision.value == "BLOCK" for e in t.security_events][:5],
        "routes": [str(t.backend_used.value if t.backend_used else None) for t in traces],
        "seconds": round(time.time() - started, 1),
    }


def tool_calls(messages: list) -> list[tuple[str, str]]:
    """(tool name, JSON arguments) for every tool call the agent made in an AutoGen run."""
    from autogen_agentchat.messages import ToolCallRequestEvent
    out = []
    for m in messages:
        if isinstance(m, ToolCallRequestEvent):
            out += [(c.name, c.arguments) for c in m.content]
    return out
