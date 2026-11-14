from __future__ import annotations
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field
import uuid


class ActionType(str, Enum):
    CONVERSATIONAL = "conversational"   # -> LLM-only Simulation
    CODE_EXECUTION = "code"             # -> MicroVM Execution
    SHELL          = "shell"            # -> MicroVM Execution
    TOOL_CALL      = "tool_call"        # -> Mock-tool Simulation
    FINAL_RESPONSE = "final_response"   # -> Simulator 없이 BehaviorJudge


class SimulationBackend(str, Enum):
    LLM_ONLY  = "llm_only"
    ISOLATED  = "isolated"
    MOCK_TOOL = "mock_tool"


class Decision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class AgentPlan(BaseModel):
    agent_role:     str
    agent_persona:  str
    sub_queries:    List[str]
    research_focus: str


class SecurityEvent(BaseModel):
    type:     str
    location: str   # "plan" | "output" | "code" | "tool_call"
    severity: str   # "high" | "medium" | "low"
    evidence: str


class Action(BaseModel):
    user_request:    str
    agent_plan:      Optional[AgentPlan] = None
    # FINAL_RESPONSE: Plan 없이 이미 생성된 타겟의 답변
    final_response:  Optional[str]       = None
    # CODE_EXECUTION
    code:            Optional[str]       = None
    language:        str                 = "python"
    # TOOL_CALL
    tool_name:       Optional[str]       = None
    tool_arguments:  dict                = Field(default_factory=dict)
    tool_category:   Optional[str]       = None  # api|mcp|rag|browser|email|database|calendar|memory
    # 타겟 에이전트가 쓸 수 있는 도구 ("이름(인자, ...)"). 다음 행동을 시뮬레이션할 때 쓴다.
    available_tools: List[str]           = Field(default_factory=list)
    # 현재 호출한 도구의 설명 (타겟이 LLM에 준 도구 정의에서 읽는다)
    tool_description: Optional[str]      = None
    # 데이터셋이나 RAG 시스템이 따로 넘긴 검색 문서. 답변·Plan을 판정할 때 근거 문서로 쓴다.
    # 에이전트가 도구로 읽은 결과는 conversation에 들어 있다.
    retrieved_context: Optional[str]      = None
    # 타겟 에이전트의 모델이 이 행동을 내기 전에 받은 입력 전체 (시스템 프롬프트, 이전 턴을 포함한
    # 대화, 도구 호출과 결과). 시뮬레이터와 Judge가 타겟과 같은 정보를 보고 판단하게 한다.
    conversation:    Optional[str]       = None
    action_type:     Optional[ActionType] = None  # None -> manager가 자동 판별
    # Simulator가 행동을 실제로 돌려 볼 환경. 직렬화하지 않는다.
    shadow_executor:   Any = Field(default=None, exclude=True)
    execution_session: Any = Field(default=None, exclude=True)


class SimulationTrace(BaseModel):
    id:               str = Field(default_factory=lambda: f"sim_{uuid.uuid4().hex[:8]}")
    action:           Action
    backend_used:     SimulationBackend
    simulated_output: str                = ""
    security_events:  List[SecurityEvent] = []
    risk_score:       float              = 0.0
    decision:         Decision           = Decision.ALLOW
    # BehaviorJudge가 채우는 축별 상세 판정 (평균 내지 않고 축마다 독립 보관)
    axis_detail:      dict               = Field(default_factory=dict)
    # ToolCallSimulator가 shadow 환경에서 실제 도구를 실행한 결과. 기록용.
    shadow:           Any                = Field(default=None, exclude=True)


class InterceptionResult(BaseModel):
    """타겟 LLM 출력 하나에 대한 SimGuardAgent의 판정."""
    decision:   Decision              = Decision.ALLOW
    risk_score: float                 = 0.0
    # 평가한 행동마다 하나씩. BLOCK이 나오면 그 뒤 행동은 평가하지 않는다.
    traces:     List[SimulationTrace] = Field(default_factory=list)
    # 판정이 끝나지 않은 이유. 값이 있으면 decision은 BLOCK이다.
    error:      Optional[str]         = None
