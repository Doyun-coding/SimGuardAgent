"""
실제 도구를 shadow 환경에서 실행하는 실행기.

ToolCallSimulator는 행동에 실행기가 붙어 있으면 도구 결과를 상상하지 않고 실제 도구를
복제한 환경에서 실행해 결과와 상태 변화를 본다. 실제 환경은 바꾸지 않는다.

제공하는 실행기
  RuntimeShadowToolExecutor   메모리 안의 환경 객체를 복사해서 실행 (AgentDojo 등)
  DatabaseShadowExecutor      DB 도구를 트랜잭션 안에서 실행하고 되돌림 (simguard.shadow_db)
  MCPShadowExecutor           MCP 도구를 테스트용 MCP 서버 인스턴스에서 실행 (simguard.shadow_mcp)
직접 만들 때는 shadow_execute(tool_name, tool_arguments) -> ShadowExecutionResult 하나만 구현한다.
async 함수여도 된다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class ShadowExecutionResult:
    """Observed result of running a proposed tool call in a shadow environment."""

    result: str
    error: str | None = None
    state_diff: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ShadowToolExecutor(Protocol):
    """실제 도구를 shadow 환경에서 실행한다. 실제 환경은 바꾸지 않아야 한다."""

    def shadow_execute(self, tool_name: str, tool_arguments: dict) -> ShadowExecutionResult | None:
        """동기 함수이거나, ShadowExecutionResult를 돌려주는 async 함수다.
        이 호출을 shadow에서 안전하게 실행할 수 없으면 None을 돌려준다. 그러면 내부 모델이 결과를 예상한다."""
        ...


class RuntimeShadowToolExecutor:
    """Generic executor for Pydantic-like env + runtime.run_function systems.

    This matches AgentDojo directly, but is intentionally not AgentDojo-specific:
    any agent runtime with ``run_function(env, tool_name, args)`` and an env that
    supports ``model_copy(deep=True)``/``model_dump(mode="json")`` can use it.
    """

    def __init__(
        self,
        runtime: Any,
        env: Any,
        result_formatter: Callable[[Any], str] | None = None,
    ) -> None:
        self.runtime = runtime
        self.env = env
        self.result_formatter = result_formatter or str

    def _format(self, result: Any) -> str:
        try:
            return self.result_formatter(result)
        except Exception:
            return str(result)

    def _copy_env(self) -> Any:
        if hasattr(self.env, "model_copy"):
            return self.env.model_copy(deep=True)
        if hasattr(self.env, "copy"):
            try:
                return self.env.copy(deep=True)
            except TypeError:
                return self.env.copy()
        raise TypeError("Shadow execution requires an environment copy method.")

    @staticmethod
    def _snapshot(env: Any) -> dict:
        if hasattr(env, "model_dump"):
            return env.model_dump(mode="json")
        if hasattr(env, "dict"):
            return env.dict()
        return {}

    @staticmethod
    def _state_diff(before: dict, after: dict) -> str:
        try:
            from deepdiff import DeepDiff

            diff = DeepDiff(before, after, ignore_order=True, verbose_level=2)
            return diff.to_json(default_mapping={object: str})[:4000]
        except Exception as exc:
            return json.dumps({"diff_error": str(exc)}, ensure_ascii=False)

    def shadow_execute(self, tool_name: str, tool_arguments: dict) -> ShadowExecutionResult:
        shadow_env = self._copy_env()
        before = self._snapshot(self.env)
        result, error = self.runtime.run_function(shadow_env, tool_name, tool_arguments)
        after = self._snapshot(shadow_env)
        return ShadowExecutionResult(
            result=self._format(result),
            error=error,
            state_diff=self._state_diff(before, after),
        )
