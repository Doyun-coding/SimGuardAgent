"""
MCPShadowExecutor

MCP 도구 호출을 테스트용 MCP 서버 인스턴스에서 실행해 결과를 본다. 에이전트가 쓰는 실제 MCP 서버와는
따로 연결하므로 실제 서버에는 호출이 가지 않는다. ToolCallSimulator는 이 결과를 LLM 예상 대신
도구 결과로 쓴다.

  # 같은 서버 프로그램을 테스트 설정(테스트 DB, 샌드박스 키 등)으로 따로 띄운다
  MCPShadowExecutor(StdioServerParameters(command="python", args=["server.py"], env={"DB": "test.db"}))
  # 이미 떠 있는 테스트·스테이징 MCP 서버 (Streamable HTTP)
  MCPShadowExecutor("https://staging.example.com/mcp")

shadow의 안전성은 테스트 인스턴스에 달려 있다
  stdio 서버는 호출마다 새로 띄우므로 서버 프로세스 안의 상태는 남지 않는다. 그러나 서버가 외부
  서비스(메일, 결제, 실제 DB)에 연결하면 그 호출은 실제로 일어난다. 테스트 인스턴스는 반드시
  테스트용 외부 자원에 연결되도록 설정해야 한다. 실제 서버를 그대로 넘기면 shadow가 아니다.

상태 변화
  snapshot(session)을 넘기면 호출 전후에 실행해 차이를 state_diff로 돌려준다. 예를 들어 서버의 읽기
  도구를 불러 목록을 가져오는 함수를 넘긴다.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from simguard.shadow_api import ShadowExecutionResult


class MCPShadowExecutor:
    def __init__(
        self,
        server: Any,
        snapshot: Callable[[Any], Awaitable[Any]] | None = None,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        """
        server:   mcp.StdioServerParameters (테스트 인스턴스를 띄울 명령), 또는 테스트 서버의 HTTP URL
        snapshot: ClientSession을 받아 비교할 상태를 돌려주는 async 함수 (선택)
        timeout:  호출 하나의 최대 시간(초)
        headers:  HTTP 서버에 보낼 헤더 (테스트용 인증 등)
        """
        self._server = server
        self._snapshot = snapshot
        self._timeout = timeout
        self._headers = headers

    @asynccontextmanager
    async def _session(self):
        from mcp import ClientSession

        if isinstance(self._server, str):
            from mcp.client.streamable_http import streamablehttp_client

            transport = streamablehttp_client(self._server, headers=self._headers)
        else:
            from mcp.client.stdio import stdio_client

            transport = stdio_client(self._server)
        async with transport as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                yield session

    @staticmethod
    def _text(result: Any) -> str:
        parts = []
        for item in getattr(result, "content", None) or []:
            text = getattr(item, "text", None)
            parts.append(text if isinstance(text, str) else json.dumps(item.model_dump(mode="json"), ensure_ascii=False))
        structured = getattr(result, "structuredContent", None)
        if structured and not parts:
            parts.append(json.dumps(structured, ensure_ascii=False, default=str))
        return "\n".join(parts)

    async def _call(self, tool_name: str, tool_arguments: dict) -> ShadowExecutionResult:
        async with self._session() as session:
            before = await self._snapshot(session) if self._snapshot else None
            result = await session.call_tool(tool_name, tool_arguments)
            after = await self._snapshot(session) if self._snapshot else None
        diff = "" if self._snapshot is None else json.dumps(
            {} if before == after else {"before": before, "after": after}, ensure_ascii=False, default=str
        )[:4000]
        text = self._text(result)
        return ShadowExecutionResult(
            result="" if result.isError else text,
            error=(text or "tool returned an error") if result.isError else None,
            state_diff=diff,
            metadata={"shadow_mode": "mcp_test_instance"},
        )

    async def shadow_execute(self, tool_name: str, tool_arguments: dict) -> ShadowExecutionResult:
        try:
            return await asyncio.wait_for(self._call(tool_name, dict(tool_arguments or {})), self._timeout)
        except Exception as exc:
            # 테스트 서버에 연결하지 못한 것은 도구의 결과가 아니다. 그 사실을 오류로 넘겨 판정에 맡긴다.
            return ShadowExecutionResult(
                result="",
                error=f"shadow MCP server failed: {type(exc).__name__}: {str(exc)[:300]}",
                metadata={"shadow_mode": "mcp_test_instance", "executed": False},
            )
