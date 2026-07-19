"""
SmolAgents 에이전트 어댑터

SmolAgents ToolCallingAgent + Tavily 웹 검색 실행기.
run_benchmark.py에서 호출되는 단일 인터페이스: run_task(task, mode)

모드:
  simguard  — model.generate() 직후 tool_calls를 SimGuard로 분석
              BLOCK → 실행 중단 / ALLOW → 정상 실행
  baseline  — SimGuard 없이 agent.run() 직접 실행
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

_AGENT_DIR = Path(__file__).parent
_REPO_ROOT = _AGENT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

TIMEOUT = 180

# ─────────────────────────────────────────────────────────────────────────────
# 에이전트 (스레드 로컬)
# ─────────────────────────────────────────────────────────────────────────────

_local = threading.local()


def _make_search_tool():
    from smolagents import Tool

    class TavilySearchTool(Tool):
        name        = "web_search"
        description = "Search the web for current information using Tavily."
        inputs      = {"query": {"type": "string", "description": "Search query"}}
        output_type = "string"

        def forward(self, query: str) -> str:
            api_key = os.environ.get("TAVILY_API_KEY", "")
            if not api_key:
                return "Tavily API key not configured."
            try:
                resp = httpx.post(
                    "https://api.tavily.com/search",
                    json={"api_key": api_key, "query": query, "max_results": 5},
                    verify=False, timeout=30,
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
                if not results:
                    return "No results found."
                return "\n\n".join(
                    f"**{r.get('title','')}**\n{r.get('url','')}\n{r.get('content','')[:400]}"
                    for r in results
                )
            except Exception as e:
                return f"Search error: {e}"

    return TavilySearchTool()


_TOOL_REGISTRY = {
    "web_search": _make_search_tool,
}

_ENABLED_TOOLS: list[str] = ["web_search"]  # --tools 옵션으로 override


def get_agent():
    if not hasattr(_local, "agent"):
        from smolagents import ToolCallingAgent, OpenAIModel
        model = OpenAIModel(
            model_id=os.environ.get("PLAN_MODEL", "qwen/qwen3.5-9b"),
            api_base=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
            client_kwargs={"http_client": httpx.Client(verify=False)},
        )
        tools = [_TOOL_REGISTRY[t]() for t in _ENABLED_TOOLS if t in _TOOL_REGISTRY]
        _local.agent = ToolCallingAgent(
            tools=tools,
            model=model,
            verbosity_level=0,
            max_steps=5,
        )
    return _local.agent


# ─────────────────────────────────────────────────────────────────────────────
# SimGuard 인터셉터
# ─────────────────────────────────────────────────────────────────────────────

# tool_name → CodeSimulator로 라우팅할 코드 실행 도구
_CODE_TOOLS = {
    "execute_python", "python_repl", "run_python", "python_interpreter",
    "execute_code", "run_code", "code_interpreter", "code_execution",
    "execute_javascript", "run_node", "run_js",
}

# tool_name → CodeSimulator로 라우팅할 Shell 실행 도구
_SHELL_TOOLS = {
    "bash", "shell", "run_bash", "execute_bash", "run_shell",
    "terminal", "cmd", "powershell", "system_exec", "exec",
}


class _BlockedBySimGuard(Exception):
    def __init__(self, risk_score: float):
        self.risk_score = risk_score


class _SimGuardWrapper:
    """SmolAgents model.generate()를 래핑해 매 step의 tool_calls를 SimGuard로 분석한다."""

    def __init__(self, model, task: str, session=None):
        self._model     = model
        self._task      = task
        self._session   = session
        self._step      = 0
        self.trace      = None
        self.blocked    = False
        self.block_risk = None

    def generate(self, input_messages, **kwargs):
        chat_message = self._model.generate(input_messages, **kwargs)
        self._step += 1
        asyncio.run(self._analyze(chat_message.tool_calls or []))
        return chat_message

    async def _analyze(self, tool_calls: list):
        from simguard import SimGuardInterceptor, Decision, AgentPlan

        interceptor   = SimGuardInterceptor()
        step_max_risk = 0.0
        step_worst    = None

        print(f"\n  ┌─[SimGuard step={self._step}] task: {self._task[:60]}...")
        print(f"  │  tool_calls: {[tc.function.name for tc in tool_calls] or '없음 (conversational)'}")

        for tc in tool_calls:
            tool_name = tc.function.name
            args      = tc.function.arguments or {}
            if isinstance(args, str):
                import json as _j
                try:
                    args = _j.loads(args)
                except Exception:
                    args = {}

            if tool_name in _CODE_TOOLS:
                print(f"  │  [{tool_name}] → CodeSimulator")
                trace = await interceptor.intercept_code(
                    user_request=self._task,
                    code=args.get("code", args.get("script", str(args))),
                    language=args.get("language", "python"),
                    session=self._session,
                )
            elif tool_name in _SHELL_TOOLS:
                print(f"  │  [{tool_name}] → CodeSimulator (shell)")
                trace = await interceptor.intercept_shell(
                    user_request=self._task,
                    command=args.get("command", args.get("cmd", str(args))),
                    session=self._session,
                )
            else:
                print(f"  │  [{tool_name}] → ToolCallSimulator  args={str(args)[:60]}")
                trace = await interceptor.intercept_tool_call(
                    user_request=self._task,
                    tool_name=tool_name,
                    tool_arguments=args,
                )

            events_str = ", ".join(f"{e.type}({e.severity})" for e in trace.security_events) or "없음"
            print(f"  │    risk={trace.risk_score:.2f}  events={events_str}  → {trace.decision.value}")

            if trace.risk_score > step_max_risk:
                step_max_risk = trace.risk_score
                step_worst    = trace

            if trace.decision == Decision.BLOCK:
                print(f"  └─[BLOCK] step={self._step}  risk={trace.risk_score:.2f}\n")
                self.blocked    = True
                self.block_risk = trace.risk_score
                self.trace      = trace
                raise _BlockedBySimGuard(trace.risk_score)

        if step_worst is None and not tool_calls:
            print(f"  │  tool_calls 없음 → BehaviorSimulator (conversational)")
            step_worst = await interceptor.intercept_conversational(
                self._task,
                AgentPlan(
                    agent_role="SmolAgents ToolCallingAgent",
                    agent_persona="general-purpose research agent",
                    sub_queries=[self._task[:200]],
                    research_focus=self._task,
                ),
            )
            events_str = ", ".join(f"{e.type}({e.severity})" for e in step_worst.security_events) or "없음"
            print(f"  │    risk={step_worst.risk_score:.2f}  events={events_str}  → {step_worst.decision.value}")
            step_max_risk = step_worst.risk_score

        self.trace = step_worst
        decision   = step_worst.decision.value if step_worst else "ALLOW"
        print(f"  └─[step={self._step}] risk={step_max_risk:.2f}  decision={decision}\n")

    def __getattr__(self, name):
        return getattr(self._model, name)


# ─────────────────────────────────────────────────────────────────────────────
# 실행 함수
# ─────────────────────────────────────────────────────────────────────────────

async def _run_simguard(task: str, agent) -> dict:
    from simguard.simulation.backends.code_simulator import MicroVMSession
    from smolagents import ToolCallingAgent

    session = await asyncio.get_event_loop().run_in_executor(None, MicroVMSession.create)
    wrapper = _SimGuardWrapper(agent.model, task, session=session)

    # task마다 독립된 agent 인스턴스 생성 → workers race condition 방지
    task_agent = ToolCallingAgent(
        tools=list(agent.tools.values()),
        model=wrapper,
        verbosity_level=0,
        max_steps=agent.max_steps,
    )

    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, task_agent.run, task),
            timeout=TIMEOUT,
        )
        trace = wrapper.trace
        return {
            "response":          str(result),
            "simguard_decision": trace.decision.value if trace else "ALLOW",
            "risk_score":        trace.risk_score     if trace else 0.0,
            "blocked":           False,
            "error":             None,
        }
    except asyncio.TimeoutError:
        trace = wrapper.trace
        return {
            "response":          None,
            "simguard_decision": trace.decision.value if trace else "TIMEOUT",
            "risk_score":        trace.risk_score     if trace else None,
            "blocked":           wrapper.blocked,
            "error":             "timeout",
        }
    except Exception:
        if wrapper.blocked:
            return {
                "response":          None,
                "simguard_decision": "BLOCK",
                "risk_score":        wrapper.block_risk,
                "blocked":           True,
                "error":             None,
            }
        trace = wrapper.trace
        import traceback
        return {
            "response":          None,
            "simguard_decision": trace.decision.value if trace else "ERROR",
            "risk_score":        trace.risk_score     if trace else None,
            "blocked":           False,
            "error":             traceback.format_exc()[-300:],
        }
    finally:
        if session:
            await asyncio.get_event_loop().run_in_executor(None, session.stop)


async def _run_baseline(task: str, agent) -> dict:
    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, agent.run, task),
            timeout=TIMEOUT,
        )
        return {"response": str(result), "simguard_decision": None,
                "risk_score": None, "blocked": None, "error": None}
    except asyncio.TimeoutError:
        return {"response": None, "simguard_decision": None,
                "risk_score": None, "blocked": None, "error": "timeout"}
    except Exception as e:
        return {"response": None, "simguard_decision": None,
                "risk_score": None, "blocked": None, "error": str(e)[:300]}


async def run_task(task: str, mode: str) -> dict:
    """run_benchmark.py에서 호출하는 단일 태스크 실행 인터페이스."""
    agent = get_agent()
    if mode == "simguard":
        return await _run_simguard(task, agent)
    return await _run_baseline(task, agent)


# ─────────────────────────────────────────────────────────────────────────────
# CLI (run_benchmark.py에 위임)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import logging
    from run_benchmark import BENCHMARKS, run, evaluate, analyze

    parser = argparse.ArgumentParser(description="SmolAgents 벤치마크 어댑터")
    parser.add_argument("--benchmark", default="harmful",
                        choices=list(BENCHMARKS),
                        help="벤치마크 (harmful / ga-jailbreak / safedial)")
    parser.add_argument("--mode",      default="simguard",
                        choices=["simguard", "baseline"])
    parser.add_argument("--limit",     type=int, default=None)
    parser.add_argument("--workers",   type=int, default=1)
    parser.add_argument("--evaluate",  action="store_true")
    parser.add_argument("--analyze",   action="store_true")
    parser.add_argument("--tools",     default="web_search",
                        help="사용할 툴 (콤마 구분). 예: web_search  /  none  /  web_search,rag")
    parser.add_argument("--label",     default="attack",
                        choices=["attack", "benign", "all"],
                        help="GA_Jailbreak 레이블 필터 (attack=1 / benign=0 / all)")
    parser.add_argument("--result",    default=None,
                        help="결과 저장 경로 (기본값: results/... 자동 생성)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    # --tools 옵션 적용
    global _ENABLED_TOOLS
    _ENABLED_TOOLS = [] if args.tools.strip().lower() == "none" else [
        t.strip() for t in args.tools.split(",") if t.strip()
    ]
    print(f"[tools] {_ENABLED_TOOLS or '없음 (final_answer only)'}")

    if args.analyze:
        analyze()
    elif args.evaluate:
        asyncio.run(evaluate("smolagents", args.benchmark, args.mode, args.workers))
    else:
        asyncio.run(run("smolagents", args.benchmark, args.mode, args.limit, args.workers, args.label, args.result))
        analyze(agent="smolagents", benchmark=args.benchmark, mode=args.mode)


if __name__ == "__main__":
    main()
