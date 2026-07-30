"""API mock 실행 — ToolCallSimulator의 api 카테고리 처리에 쓰인다.

에이전트의 API 호출을 mock으로 실제 실행해 응답+부작용을 관찰한다. 판정은 하지
않는다(ToolCallSimulator가 ToolCallJudge로 판정). 여기는 "LLM 상상" 대신 "실측 응답"을
만들어 주는 역할만 한다.

  B: OpenAPI 스펙이 있는 API → 스펙대로 충실히 (상태코드·스키마·CRUD 부작용)
  C: 스펙 없는 임의 API      → 상태 유지 LLM mock

mock은 안전 필터를 걸지 않는다(결과를 보여주는 게 목적). 방어 판정은 judge가 한다.
"""
from __future__ import annotations

import json
from typing import Any

from simguard.simulation.backends.mock_router import MockRouter

_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}


async def run_api_mock(tool_arguments: dict) -> str:
    """API 호출을 mock으로 실행하고 관찰 결과(응답+부작용)를 문자열로 반환한다."""
    method, path, body = _parse_call(tool_arguments)
    router = MockRouter()   # 호출마다 시드 상태에서 시작(세션 격리)
    status, rbody, side_effects, backend = await router.route(method, path, body)

    out = f"HTTP {status}\n{_as_text(rbody)}"
    if side_effects:
        out += "\n\n[Observed side effects]\n" + "\n".join(f"- {e}" for e in side_effects)
    out += f"\n\n[mock backend: {backend}]"
    return out


def _parse_call(tool_arguments: dict) -> tuple[str, str, dict | None]:
    """tool_arguments에서 (method, path, body)를 뽑는다. 흔한 키 별칭을 폭넓게 받는다."""
    args = tool_arguments or {}
    method = str(args.get("method") or args.get("http_method") or "").upper()
    url = (args.get("url") or args.get("endpoint") or args.get("path")
           or args.get("uri") or "")
    body = (args.get("body") or args.get("data") or args.get("json")
            or args.get("payload"))

    if method not in _METHODS:
        method = "POST" if body else "GET"

    # 절대 URL(http://host/…) fetch는 SSRF 표면 → 호스트 보존을 위해 /fetch로 보낸다.
    # 상대 경로(/users)는 알려진 API 호출이라 직접 라우팅한다.
    url_str = str(url)
    if url_str.lower().startswith(("http://", "https://")):
        from urllib.parse import quote
        path = f"/fetch?url={quote(url_str, safe='')}"
    else:
        path = _to_path(url_str)

    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            body = {"_raw": body}
    if body is not None and not isinstance(body, dict):
        body = {"_raw": body}

    return method, path, body


def _to_path(url: str) -> str:
    u = url.strip()
    for scheme in ("http://", "https://"):
        if u.lower().startswith(scheme):
            rest = u[len(scheme):]
            slash = rest.find("/")
            return rest[slash:] if slash != -1 else "/"
    if not u.startswith("/"):
        u = "/" + u
    return u


def _as_text(body: Any) -> str:
    return body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
