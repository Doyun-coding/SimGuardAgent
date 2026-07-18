"""
SimGuard Interceptor

에이전트 액션의 메인 진입점.

흐름:
  Action
    -> Simulation Manager  (백엔드 분류 + 시뮬레이션)
    -> Risk Analyzer       (trace 검토)
    -> Decision Controller (ALLOW / BLOCK)
    -> SimulationTrace     (결과 반환)
"""
from __future__ import annotations

from simguard.analyzer.risk_analyzer import RiskAnalyzer
from simguard.decision.controller import DecisionController
from simguard.models import Action, ActionType, AgentPlan, SimulationTrace
from simguard.simulation.manager import SimulationManager


class SimGuardInterceptor:
    def __init__(self):
        self._manager    = SimulationManager()
        self._analyzer   = RiskAnalyzer()
        self._controller = DecisionController()

    async def intercept(self, action: Action) -> SimulationTrace:
        trace          = await self._manager.simulate(action)
        trace          = await self._analyzer.analyze(trace)
        trace.decision = self._controller.decide(trace.risk_score)
        return trace

    async def _intercept_with_session(self, action: Action, session) -> SimulationTrace:
        """세션 컨테이너를 직접 사용해 CodeSimulator를 호출한다."""
        from simguard.simulation.backends.code_simulator import CodeSimulator
        trace = await CodeSimulator().simulate(action, session=session)
        trace = await self._analyzer.analyze(trace)
        trace.decision = self._controller.decide(trace.risk_score)
        return trace

    # ------------------------------------------------------------------
    # 편의 메서드: smolagents / AgentDojo 어댑터에서 호출
    # ------------------------------------------------------------------

    async def intercept_conversational(
        self,
        user_request: str,
        agent_plan: AgentPlan | None = None,
    ) -> SimulationTrace:
        return await self.intercept(Action(
            user_request=user_request,
            agent_plan=agent_plan,
        ))

    async def intercept_code(
        self,
        user_request: str,
        code: str,
        language: str = "python",
        session=None,
    ) -> SimulationTrace:
        action = Action(user_request=user_request, code=code, language=language)
        if session:
            return await self._intercept_with_session(action, session)
        return await self.intercept(action)

    async def intercept_shell(
        self,
        user_request: str,
        command: str,
        session=None,
    ) -> SimulationTrace:
        """Shell 명령을 MicroVM에서 격리 실행하고 위협을 분석한다."""
        action = Action(
            user_request=user_request,
            code=command,
            language="bash",
            action_type=ActionType.SHELL,
        )
        if session:
            return await self._intercept_with_session(action, session)
        return await self.intercept(action)

    async def intercept_tool_call(
        self,
        user_request: str,
        tool_name: str,
        tool_arguments: dict,
        tool_category: str | None = None,
    ) -> SimulationTrace:
        return await self.intercept(Action(
            user_request=user_request,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            tool_category=tool_category,
        ))
