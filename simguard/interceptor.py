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
from simguard.shadow_api import GuardedToolCallResult, ShadowToolExecutor


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

    async def intercept_axes(
        self,
        user_query: str,
        contexts: str,
        answer: str = "",
        profile: str = "rag_poison",
        clean_answer: str | None = None,
        ground_truth: str | None = None,
    ) -> SimulationTrace:
        """다축 독립 평가. 공격 유형에 맞는 프로필로 필요한 축만 활성화한다.

        축 점수는 평균 내지 않는다. 활성 축 중 하나라도 임계를 넘으면 BLOCK이다.
        clean_answer(정상 문맥으로 만든 답)를 함께 주면 오염이 결론을 바꿨는지
        반사실 비교로 판정할 수 있어 integrity 정확도가 크게 오른다.
        """
        from simguard.models import SimulationBackend
        from simguard.simulation.judges.axis_judge import AxisJudge

        events, score, detail = await AxisJudge(profile).judge(
            user_query=user_query,
            contexts=contexts,
            answer=answer,
            clean_answer=clean_answer,
            ground_truth=ground_truth,
        )
        trace = SimulationTrace(
            action=Action(user_request=user_query, retrieved_context=contexts),
            backend_used=SimulationBackend.LLM_ONLY,
            simulated_output=answer,
            security_events=events,
            risk_score=score,
        )
        trace = await self._analyzer.analyze(trace)
        trace.decision = self._controller.decide(trace.risk_score)
        # 축별 상세는 trace에 담을 자리가 없어 호출자가 쓰도록 반환값에 붙인다
        trace.axis_detail = detail
        return trace

    async def intercept_retrieval(
        self,
        user_request: str,
        tool_name: str,
        retrieved_context: str,
        tool_arguments: dict | None = None,
        agent_plan: AgentPlan | None = None,
    ) -> SimulationTrace:
        """RAG·Memory로 실제 조회된 내용을 근거로 에이전트 행동을 평가한다.

        읽기 연산은 호출 자체가 피해가 아니라 가져온 내용이 위험하므로,
        응답을 상상하는 ToolCallSimulator 대신 BehaviorSimulator 경로를 탄다.
        retrieved_context는 벤치마크에서는 데이터셋이, 배포에서는 RAG 시스템이 채운다.
        """
        return await self.intercept(Action(
            user_request=user_request,
            agent_plan=agent_plan,
            tool_name=tool_name,
            tool_arguments=tool_arguments or {},
            retrieved_context=retrieved_context,
        ))

    async def intercept_plan(
        self,
        user_request: str,
        plan_text: str,
    ) -> SimulationTrace:
        """에이전트가 실제로 생성한 계획을 시뮬레이션한 뒤 평가한다.

        흐름: 실제 계획 -> BehaviorSimulator(계획대로 실행하면 나올 출력)
              -> BehaviorJudge(그 출력을 평가) -> ALLOW / BLOCK
        (final_answer·retrieval 경로는 AxisJudge 다축 평가를 쓴다)

        계획을 상상해 만들지 않고 방금 생성된 원문을 시뮬레이터에 넘기는 것이 핵심이다.
        """
        plan = AgentPlan(
            agent_role="SmolAgents ToolCallingAgent",
            agent_persona="general-purpose agent",
            sub_queries=[plan_text],
            research_focus=user_request,
        )
        return await self.intercept(Action(
            user_request=user_request,
            agent_plan=plan,
        ))

    async def intercept_final_answer(
        self,
        user_request: str,
        answer: str,
        agent_plan: AgentPlan | None = None,
    ) -> SimulationTrace:
        """에이전트의 실제 최종 응답을 시뮬레이션 없이 그대로 평가한다.

        final_answer는 툴 호출이 아니라 최종 출력이므로, 응답을 상상해 만들 필요 없이
        손에 있는 내용을 직접 judge에 넘긴다.
        """
        from simguard.models import SimulationBackend
        from simguard.simulation.judges.axis_judge import AxisJudge

        events, score, detail = await AxisJudge("jailbreak").judge(
            user_query=user_request, contexts="", answer=answer,
        )
        trace = SimulationTrace(
            action=Action(
                user_request=user_request,
                tool_name="final_answer",
                tool_arguments={"answer": answer},
            ),
            backend_used=SimulationBackend.LLM_ONLY,
            simulated_output=answer,
            security_events=events,
            risk_score=score,
        )
        trace = await self._analyzer.analyze(trace)
        trace.decision = self._controller.decide(trace.risk_score)
        trace.axis_detail = detail
        return trace

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

    async def intercept_observed_tool_call(
        self,
        user_request: str,
        tool_name: str,
        tool_arguments: dict,
        observed_result: str,
        state_diff: str = "",
        tool_error: str | None = None,
        tool_category: str | None = None,
    ) -> SimulationTrace:
        """실제 shadow 실행으로 얻은 tool 결과와 상태 변화를 근거로 판정한다."""
        import json

        observed_context = json.dumps(
            {
                "tool_result": observed_result,
                "tool_error": tool_error,
                "state_diff": state_diff,
            },
            ensure_ascii=False,
        )
        return await self.intercept(Action(
            user_request=user_request,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            tool_category=tool_category,
            retrieved_context=observed_context,
            action_type=ActionType.TOOL_CALL,
        ))

    async def guard_shadow_tool_call(
        self,
        user_request: str,
        tool_name: str,
        tool_arguments: dict,
        shadow_executor: ShadowToolExecutor,
        tool_category: str | None = None,
    ) -> GuardedToolCallResult:
        """SimGuard 내부에서 shadow 실행, 판정, commit/replay까지 수행한다."""
        shadow = shadow_executor.shadow_execute(tool_name, tool_arguments)
        trace = await self.intercept_observed_tool_call(
            user_request=user_request,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            observed_result=shadow.result,
            state_diff=shadow.state_diff,
            tool_error=shadow.error,
            tool_category=tool_category,
        )

        decision_value = getattr(trace.decision, "value", str(trace.decision)).upper()
        if decision_value == "BLOCK":
            return GuardedToolCallResult(
                trace=trace,
                result=f"Tool call blocked by SimGuardAgent. risk_score={trace.risk_score:.2f}",
                error="BlockedBySimGuard",
                committed=False,
                shadow=shadow,
            )

        result, error = shadow_executor.commit(tool_name, tool_arguments)
        return GuardedToolCallResult(
            trace=trace,
            result=result,
            error=error,
            committed=True,
            shadow=shadow,
        )
