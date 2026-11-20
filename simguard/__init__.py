"""
SimGuardAgent: 타겟 에이전트의 Plan을 실행 전에 시뮬레이션하고 판정한다.

  await SimGuardAgentInterceptor().intercept_output(messages, output, tools)   # 핵심 입구
  simguard.integrations.agentdojo.with_simguard(pipeline)                      # AgentDojo
  simguard.protect()                                                            # 설치된 SDK·로컬 GPU 모델 호출 (providers=)
  protect(llm_calls=[LLMCall(...)])                                             # 목록에 없는 LLM 호출 방법
  protect(tools=[send_email, search_hotels])                                    # 자동으로 못 찾는 실제 도구 알려 주기
  protect(database=lambda: psycopg.connect(DSN))                                # 에이전트의 DB에서 SQL 호출 판정
  protect(workdir="/home/me/project")                                           # 코드를 에이전트 작업 파일 위에서 실행
"""
from simguard.interceptor import BLOCKED_MESSAGE, SimGuardAgentInterceptor
from simguard.models import (
    Action, ActionType, Decision, InterceptionResult,
    SecurityEvent, SimulationBackend, SimulationTrace,
)
from simguard.live_api import LiveToolExecutor, ToolRunResult
from simguard.execution_environment import ExecutionEnvironment
from simguard.live_db import DatabaseExecutor
from simguard.live_http import LiveReadExecutor
from simguard.integrations.llm_calls import SimGuardBlocked, protect, unprotect
from simguard.integrations.llm_providers import LLMCall, parse_text_tool_calls
from simguard.tool_discovery import RealTool, discover_tools, find_tool, find_tools, register_tool, register_tools, as_real_tool, track_mcp

__all__ = [
    "SimGuardAgentInterceptor", "BLOCKED_MESSAGE",
    "Action", "ActionType", "Decision", "InterceptionResult",
    "SecurityEvent", "SimulationBackend", "SimulationTrace",
    "ToolRunResult", "LiveToolExecutor", "DatabaseExecutor", "LiveReadExecutor", "ExecutionEnvironment",
    "protect", "unprotect", "SimGuardBlocked", "LLMCall", "parse_text_tool_calls",
    "RealTool", "discover_tools", "find_tool", "find_tools", "register_tool", "register_tools", "as_real_tool", "track_mcp",
]
