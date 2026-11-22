"""
Simulation Manager

액션을 분류하고 적절한 시뮬레이션 백엔드로 라우팅한다.

분류 기준:
  FINAL_RESPONSE                      -> BehaviorJudge  (시뮬레이션할 Plan이 없다)
  CONVERSATIONAL                      -> BehaviorSimulator
  CODE_EXECUTION / SHELL              -> ExecutionSimulator  (Docker)
  TOOL_CALL                           -> ToolCallSimulator

분류는 Action에 채워진 필드만 본다:
  1. action_type 있음                 -> 그대로 사용
  2. code 있고 language가 셸           -> SHELL
  3. code 있음 (언어 무관)             -> CODE_EXECUTION
     표에 있는 언어는 그 언어의 컨테이너에서 실행하고, 없는 언어는 내부 모델이 실행 결과를 예측한다
     (simguard/languages.py)
  4. tool_name 있음                   -> TOOL_CALL (RAG 검색, 메모리 읽기·쓰기 포함)
  5. 그 외                            -> CONVERSATIONAL

TOOL_CALL이면 도구 종류(tool_category)도 정한다. 애플리케이션이 준 값을 쓰고, 없으면 도구 이름으로 추정한다
(memory·remember·store·recall -> memory 등). memory이면 ToolCallJudge가 메모리 점검을 더해 판정한다.
애플리케이션이 DB를 넘겨(protect(database=...)) 그 DB 사본에서 SQL을 실행한 호출은 ToolCallJudge가 DB 점검을 더한다.
FINAL_RESPONSE에 검색 문서(retrieved_context)가 있으면 BehaviorJudge가 RAG 점검을 더해 판정한다.
"""
from __future__ import annotations

from simguard.models import Action, ActionType, SimulationBackend, SimulationTrace
from simguard.simulation.backends.behavior_simulator import BehaviorSimulator
from simguard.simulation.backends.tool_call_simulator import ToolCallSimulator, infer_tool_category
from simguard.simulation.judges.behavior_judge import BehaviorJudge

from simguard import languages


class SimulationManager:
    def __init__(self):
        self._llm = BehaviorSimulator()
        self._tool = ToolCallSimulator()
        self._behavior_judge = BehaviorJudge()
        # Code/Shell execution is expensive and may require Docker. Load it only
        # when a code or shell action is actually evaluated. VESSL prompt-level
        # behavior experiments do not need this backend.
        self._execution = None

    def _classify(self, action: Action) -> ActionType:
        if action.action_type:
            return action.action_type
        if action.code is not None and languages.is_shell(action.language):
            return ActionType.SHELL
        if action.code:
            return ActionType.CODE_EXECUTION
        if action.tool_name:
            # RAG 검색, 메모리 읽기·쓰기도 다른 도구 호출과 같다. 가져온 문서와 메모리는 타겟 모델의
            # 다음 입력으로 들어가고, 그 입력으로 낸 출력을 다시 판정한다.
            return ActionType.TOOL_CALL
        return ActionType.CONVERSATIONAL

    def _execution_simulator(self):
        if self._execution is None:
            from simguard.simulation.backends.execution_simulator import ExecutionSimulator

            self._execution = ExecutionSimulator()
        return self._execution

    async def simulate(self, action: Action) -> SimulationTrace:
        action_type = self._classify(action)

        if action_type == ActionType.FINAL_RESPONSE:
            events, risk_score, axis_detail = await self._behavior_judge.judge(
                user_query=action.user_request,
                contexts=action.retrieved_context or "",
                answer=action.final_response,
                conversation=action.conversation,
            )
            return SimulationTrace(
                action=action,
                backend_used=SimulationBackend.LLM_ONLY,
                simulated_output=action.final_response,
                security_events=events,
                risk_score=risk_score,
                axis_detail=axis_detail,
            )

        if action_type in (ActionType.CODE_EXECUTION, ActionType.SHELL):
            return await self._execution_simulator().simulate(action)

        if action_type == ActionType.TOOL_CALL:
            if not action.tool_category:
                action = action.model_copy(update={"tool_category": infer_tool_category(action.tool_name or "")})
            return await self._tool.simulate(action)

        return await self._llm.simulate(action)
