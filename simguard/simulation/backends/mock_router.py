"""MockRouter — B(스펙 기반) 우선, 없으면 C(LLM)로 폴백.

  요청 → OpenAPIMock.handle()  (스펙에 있으면 충실한 실측)
        └ None이면 → LLMMock.handle()  (임의 API 커버리지)

상태(state)는 라우터 인스턴스가 보유해 호출 간 공유된다 → 한 세션 안에서 부작용이
누적된다(DELETE 후 GET하면 사라짐). judge는 (status, body, side_effects)를 근거로 본다.
"""
from __future__ import annotations

from typing import Any

from simguard.simulation.backends.llm_mock import LLMMock
from simguard.simulation.backends.openapi_mock import OpenAPIMock


class MockRouter:
    def __init__(self):
        self._b = OpenAPIMock()
        self._c = LLMMock()
        self.state: dict[str, dict] = self._b.seed_state()

    async def route(self, method: str, path: str, body: dict | None
                    ) -> tuple[int, Any, list[str], str]:
        """(status, body, side_effects, backend) 반환. backend는 'openapi' 또는 'llm'."""
        hit = self._b.handle(method, path, body, self.state)
        if hit is not None:
            status, rbody, se = hit
            return status, rbody, se, "openapi"
        status, rbody, se = await self._c.handle(method, path, body, self.state)
        return status, rbody, se, "llm"
