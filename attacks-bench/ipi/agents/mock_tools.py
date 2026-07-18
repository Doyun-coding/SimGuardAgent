"""
Mock Tools — IPI 벤치마크용 가짜 툴

실제 DuckDuckGo를 호출하지 않고, 미리 정의된 injected_content를 반환한다.
에이전트 입장에서는 진짜 검색 결과처럼 보임.
"""
from __future__ import annotations

from smolagents import Tool


class MockSearchTool(Tool):
    """
    항상 injected_content를 반환하는 가짜 검색 툴.
    baseline: 에이전트가 주입 내용을 그대로 읽음
    simguard: SimGuard가 pre-scan 후 BLOCK/ALLOW
    """
    name        = "web_search"
    description = "Search the web for information about a given query."
    inputs      = {"query": {"type": "string", "description": "The search query"}}
    output_type = "string"

    def __init__(self, injected_content: str):
        self._content = injected_content
        super().__init__()

    def forward(self, query: str) -> str:
        return self._content
