"""
내부 시뮬레이터 LLM 호출 공통부.

thinking 엔드포인트의 빈 응답(content=None)과 JSON 파싱 실패를 한곳에서 처리한다.
"""
from __future__ import annotations

import asyncio
import json
import os
import re

from simguard.config import SHADOW_MODEL


async def shadow_json_completion(client, prompt: str) -> str:
    """Return JSON text without passing OpenRouter ``content=None`` onward."""
    max_attempts = max(1, int(os.getenv("SHADOW_MAX_ATTEMPTS", "3")))
    max_tokens = max(1024, int(os.getenv("SHADOW_MAX_TOKENS", "4096")))
    # Thinking endpoints can spend the whole completion budget on reasoning
    # and return empty content, forcing a retry on every call. JUDGE_NO_THINK=1
    # disables reasoning from the first attempt (same knob as ExecutionJudge).
    force_no_think = os.getenv("JUDGE_NO_THINK", "0").lower() in {"1", "true", "yes", "on"}
    last_error: Exception | None = None
    last_reasoning = ""
    for attempt in range(max_attempts):
        try:
            request = {
                "model": SHADOW_MODEL,
                "temperature": 0,
                "max_tokens": max_tokens,
                "messages": [{
                    "role": "user",
                    "content": f"{prompt}\n\n/no_think\nReturn the JSON object immediately.",
                }],
                "response_format": {"type": "json_object"},
            }
            if force_no_think or attempt:
                request["extra_body"] = {
                    "reasoning": {"effort": "none", "exclude": True}
                }
            resp = await client.chat.completions.create(**request)
            message = resp.choices[0].message
            content = message.content or ""
            extra = getattr(message, "model_extra", None) or {}
            reasoning = (
                getattr(message, "reasoning_content", None)
                or extra.get("reasoning_content")
                or ""
            )
            if content.strip():
                return content
            if reasoning.strip():
                last_reasoning = reasoning
            raise ValueError(
                "shadow model returned empty content "
                f"(finish_reason={resp.choices[0].finish_reason})"
            )
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max_attempts:
                await asyncio.sleep(min(2 ** attempt, 4))
    if last_reasoning:
        return last_reasoning
    assert last_error is not None
    raise RuntimeError(
        f"Shadow JSON completion failed after {max_attempts} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def parse_json_object(content: str) -> dict:
    text = (content or "").strip()
    if not text:
        raise ValueError("empty shadow response")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                value, _ = decoder.raw_decode(text[match.start():])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise
    if not isinstance(value, dict):
        raise ValueError(f"shadow output was {type(value).__name__}, expected object")
    return value
