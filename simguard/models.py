from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid


class ActionType(str, Enum):
    CONVERSATIONAL = "conversational"   # -> LLM-only Simulation
    CODE_EXECUTION = "code"             # -> MicroVM Execution
    SHELL          = "shell"            # -> MicroVM Execution
    TOOL_CALL      = "tool_call"        # -> Mock-tool Simulation


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


class ToolCall(BaseModel):
    tool_name:    str
    arguments:    dict  = {}
    result:       str   = ""
    risk_score:   float = 0.0
    is_dangerous: bool  = False
    threat_type:  str   = "none"
    reasoning:    str   = ""


class Action(BaseModel):
    user_request:    str
    agent_plan:      Optional[AgentPlan] = None
    # CODE_EXECUTION
    code:            Optional[str]       = None
    language:        str                 = "python"
    # TOOL_CALL
    tool_name:       Optional[str]       = None
    tool_arguments:  dict                = Field(default_factory=dict)
    tool_category:   Optional[str]       = None  # api|mcp|rag|browser|email|database|calendar|memory
    action_type:     Optional[ActionType] = None  # None -> manager가 자동 판별


class SimulationTrace(BaseModel):
    id:               str = Field(default_factory=lambda: f"sim_{uuid.uuid4().hex[:8]}")
    action:           Action
    backend_used:     SimulationBackend
    simulated_output: str                = ""
    security_events:  List[SecurityEvent] = []
    risk_score:       float              = 0.0
    decision:         Decision           = Decision.ALLOW
