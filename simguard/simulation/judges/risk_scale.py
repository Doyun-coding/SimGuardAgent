"""Judge 응답(JSON 객체)을 읽는다. 코드 블록이나 앞뒤 글을 떼고, 쉼표 누락 같은 문법 오류만 고친다."""
from __future__ import annotations

import json
import re

def parse_json_object(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as original_error:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise original_error
        candidate = text[start:end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            # 쉼표 누락 같은 문법 오류만 고친다. 필드 검사는 호출한 쪽에서 한다.
            try:
                from json_repair import repair_json
            except ImportError:
                raise original_error
            parsed = repair_json(candidate, return_objects=True)
    if not isinstance(parsed, dict):
        raise ValueError("Judge response must be a JSON object")
    return parsed
