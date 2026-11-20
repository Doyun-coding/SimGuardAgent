"""
판정 전에 실제 도구를 실행하는 실행기의 공통 형식.

ToolCallSimulator는 행동에 실행기가 붙어 있으면 도구 결과를 상상하지 않고 실행기가 돌려준 실제
결과로 판정한다. 실행기가 None을 돌려주면 그 호출은 실행할 수 없다는 뜻이고, 내부 모델이 결과를 예상한다.

제공하는 실행기
  LiveReadExecutor   에이전트의 실제 도구를 GET 전용 모드로 실행 (simguard.live_http)
  DatabaseExecutor   에이전트 DB에서 SQL을 실행하고 되돌림 (simguard.live_db)
  InternalDatabase   에이전트 DB가 없을 때 SimGuardAgent 내부 SQLite에서 실행 (simguard.internal_database)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolRunResult:
    """판정 전에 도구 호출을 실제로 실행해 본 결과."""

    result: str
    error: str | None = None
    state_diff: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class LiveToolExecutor(Protocol):
    def live_execute(self, tool_name: str, tool_arguments: dict) -> ToolRunResult | None:
        """동기 함수이거나 ToolRunResult를 돌려주는 async 함수다.
        이 호출을 안전하게 실행할 수 없으면 None을 돌려준다. 그러면 내부 모델이 결과를 예상한다."""
        ...
