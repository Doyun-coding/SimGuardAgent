"""
LangGraph 벤치마크 어댑터

agents/langgraph/agent.py 의 에이전트를 감싸
SimGuard / Llama Guard / GPT-OSS Safeguard / baseline 모드를 지원한다.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv
from .base import AgentAdapter
from .smolagents_agent import LLAMA_GUARD_MODEL, GPT_SAFEGUARD_MODEL, _call_guard_api

load_dotenv()

TIMEOUT = 180

_REPO_ROOT = Path(__file__).parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# agents/langgraph 를 경로에 추가
_AGENT_ROOT = _REPO_ROOT / "agents" / "langgraph"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from agent import create_agent, create_llm, create_tools, run_agent as _run_agent_sync


def _get_initial_plan(task: str, llm, tools) -> list[str]:
    """LLM 1회 호출로 예상 tool_calls 추출 (실제 실행 없음)."""
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke([{"role": "user", "content": task}])

    queries = [task]
    for tc in (response.tool_calls or []):
        args = tc.get("args", {})
        q = args.get("query") or args.get("__arg1") or ""
        if q:
            queries.append(str(q))
    return queries


async def _run_simguard(task: str, graph, llm, tools) -> dict:
    from simguard.models import AgentPlan
    from simguard.simworld.shadow_runner import run_shadow_execution
    from simguard.analyzer.jailbreak_detector import analyze_jailbreak
    from simguard.decision.decision_controller import make_decision

    # 1. 플랜 추출
    sub_queries = await asyncio.get_event_loop().run_in_executor(
        None, _get_initial_plan, task, llm, tools
    )

    plan = AgentPlan(
        agent_role="LangGraph ReAct Agent",
        agent_persona="web search agent",
        sub_queries=sub_queries,
        research_focus=task,
    )

    # 2. SimWorld 실행
    shadow = await run_shadow_execution(task, plan)

    # 3. Jailbreak 분석
    events, risk_score = await analyze_jailbreak(task, plan, shadow)

    # 4. 결정
    decision = make_decision(risk_score)

    if decision == "BLOCK":
        return {
            "report":            None,
            "simguard_decision": decision,
            "risk_score":        risk_score,
            "error":             None,
        }

    # ALLOW → 실제 에이전트 실행
    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _run_agent_sync, graph, task),
            timeout=TIMEOUT,
        )
        return {
            "report":            result,
            "simguard_decision": decision,
            "risk_score":        risk_score,
            "error":             None,
        }
    except asyncio.TimeoutError:
        return {"report": None, "simguard_decision": decision, "risk_score": risk_score, "error": "timeout"}
    except Exception as e:
        return {"report": None, "simguard_decision": decision, "risk_score": risk_score, "error": str(e)[:200]}


class LanggraphAdapter(AgentAdapter):
    name = "langgraph"

    def __init__(self):
        self._local = threading.local()

    def _get_components(self):
        if not hasattr(self._local, "graph"):
            llm   = create_llm()
            tools = create_tools()
            graph, _, _ = create_agent(llm, tools)
            self._local.graph = graph
            self._local.llm   = llm
            self._local.tools = tools
        return self._local.graph, self._local.llm, self._local.tools

    async def query(self, task: str, mode: str, **kwargs) -> dict:
        graph, llm, tools = self._get_components()

        if mode == "simguard":
            try:
                return await asyncio.wait_for(_run_simguard(task, graph, llm, tools), timeout=TIMEOUT)
            except asyncio.TimeoutError:
                return {"report": None, "simguard_decision": None, "risk_score": None, "error": "timeout"}
            except Exception as e:
                return {"report": None, "simguard_decision": None, "risk_score": None, "error": str(e)[:200]}

        if mode in ("llamaguard", "gptsafeguard"):
            guard_model = LLAMA_GUARD_MODEL if mode == "llamaguard" else GPT_SAFEGUARD_MODEL
            verdict, is_blocked, error = await _call_guard_api(guard_model, task)
            if error:
                return {"report": None, "simguard_decision": None, "risk_score": None,
                        "error": f"{mode}_api: {error}", "guard_verdict": None}
            return {
                "report":            None,
                "simguard_decision": "BLOCK" if is_blocked else "ALLOW",
                "risk_score":        1.0 if is_blocked else 0.0,
                "error":             None,
                "guard_verdict":     verdict,
            }

        # baseline
        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, _run_agent_sync, graph, task),
                timeout=TIMEOUT,
            )
            return {"report": result, "simguard_decision": None, "risk_score": None, "error": None}
        except asyncio.TimeoutError:
            return {"report": None, "simguard_decision": None, "risk_score": None, "error": "timeout"}
        except Exception as e:
            return {"report": None, "simguard_decision": None, "risk_score": None, "error": str(e)[:200]}
