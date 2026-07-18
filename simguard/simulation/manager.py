"""
Simulation Manager

액션을 분류하고 적절한 시뮬레이션 백엔드로 라우팅한다.

분류 기준:
  CONVERSATIONAL                      -> BehaviorSimulator
  CODE_EXECUTION / SHELL              -> CodeSimulator  (Docker → subprocess fallback)
  TOOL_CALL                           -> ToolCallSimulator

자동 분류 (action_type 미지정 시):
  action.code 있음                    -> CODE_EXECUTION
  action.tool_name 있음               -> TOOL_CALL
  action.language in (bash/sh/shell)  -> SHELL
  그 외                               -> CONVERSATIONAL
"""
from __future__ import annotations

from simguard.models import Action, ActionType, SimulationTrace
from simguard.simulation.backends.behavior_simulator import BehaviorSimulator
from simguard.simulation.backends.code_simulator import CodeSimulator
from simguard.simulation.backends.tool_call_simulator import ToolCallSimulator

_SHELL_LANGS = {"bash", "sh", "shell", "cmd", "powershell", "zsh"}


class SimulationManager:
    def __init__(self):
        self._llm     = BehaviorSimulator()
        self._microvm = CodeSimulator()
        self._tool    = ToolCallSimulator()

    def _classify(self, action: Action) -> ActionType:
        if action.action_type:
            return action.action_type
        if action.language and action.language.lower() in _SHELL_LANGS:
            return ActionType.SHELL
        if action.code:
            return ActionType.CODE_EXECUTION
        if action.tool_name:
            return ActionType.TOOL_CALL
        return ActionType.CONVERSATIONAL

    async def simulate(self, action: Action) -> SimulationTrace:
        action_type = self._classify(action)

        if action_type in (ActionType.CODE_EXECUTION, ActionType.SHELL):
            return await self._microvm.simulate(action)

        if action_type == ActionType.TOOL_CALL:
            return await self._tool.simulate(action)

        return await self._llm.simulate(action)
