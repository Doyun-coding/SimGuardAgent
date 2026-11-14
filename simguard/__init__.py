"""
SimGuardAgent: 타겟 에이전트의 Plan을 실행 전에 시뮬레이션하고 판정한다.

  await SimGuardAgentInterceptor().intercept_output(messages, output, tools)   # 핵심 입구
  simguard.integrations.agentdojo.with_simguard(pipeline)                      # AgentDojo
  simguard.protect()                                                            # OpenAI SDK, LiteLLM
  ExecutionEnvironment(workdir=...)                                             # 코드 실행 에이전트의 작업 파일
  protect(shadow_executors={"execute_sql": DatabaseShadowExecutor("app.db")})   # 도구를 shadow 환경에서 실행
  protect(database=lambda: psycopg.connect(DSN))                                # 에이전트의 DB에서 SQL 호출 판정
"""
from simguard.interceptor import BLOCKED_MESSAGE, SimGuardAgentInterceptor
from simguard.execution_environment import ExecutionEnvironment
from simguard.models import (
    Action, ActionType, AgentPlan, Decision, InterceptionResult,
    SecurityEvent, SimulationBackend, SimulationTrace,
)
from simguard.shadow_api import RuntimeShadowToolExecutor, ShadowExecutionResult, ShadowToolExecutor
from simguard.shadow_db import DatabaseShadowExecutor
from simguard.shadow_mcp import MCPShadowExecutor
from simguard.shadow_http import LiveReadExecutor
from simguard.integrations.llm_calls import SimGuardBlocked, protect, unprotect
from simguard.tool_discovery import RealTool, discover_tools, find_tool, find_tools, register_tool, track_mcp

__all__ = [
    "SimGuardAgentInterceptor", "BLOCKED_MESSAGE", "ExecutionEnvironment",
    "Action", "ActionType", "AgentPlan", "Decision", "InterceptionResult",
    "SecurityEvent", "SimulationBackend", "SimulationTrace",
    "RuntimeShadowToolExecutor", "ShadowExecutionResult", "ShadowToolExecutor",
    "DatabaseShadowExecutor", "MCPShadowExecutor", "LiveReadExecutor",
    "protect", "unprotect", "SimGuardBlocked",
    "RealTool", "discover_tools", "find_tool", "find_tools", "register_tool", "track_mcp",
]
