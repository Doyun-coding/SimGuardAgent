"""
AgentAdapter 기본 클래스

새 에이전트 추가 방법:
  1. attacks-bench/jailbreak/agents/<agent_name>.py 생성
  2. AgentAdapter 상속
  3. query() 구현
  4. run_benchmark.py의 AGENT_REGISTRY에 등록
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class AgentAdapter(ABC):
    """에이전트와 벤치마크 러너 사이의 인터페이스"""

    name: str = ""

    @abstractmethod
    async def query(self, task: str, simguard: bool) -> dict:
        """
        에이전트에 task를 입력하고 결과를 반환한다.

        반환 dict 필수 키:
          report          : str | None   — 생성된 리포트 (없으면 None)
          simguard_decision: str | None  — "BLOCK" / "ALLOW" / "REWRITE" (simguard=False면 None)
          risk_score      : float | None
          error           : str | None   — 오류 메시지
        """

    def is_blocked(self, response: dict) -> bool | None:
        """simguard=True일 때 BLOCK 여부. simguard=False면 None."""
        decision = response.get("simguard_decision")
        if decision is None:
            return None
        return decision == "BLOCK"
