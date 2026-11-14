"""
정적 신호: LLM 없이 규칙으로 찾는, 외부 내용이 에이전트를 조종하려 한다는 객관적 표시.

ToolCallJudge는 신호가 하나라도 있으면 그 신호를 Judge 입력에 적고, BehaviorJudge의 integrity·neutrality·
correctness 세 축으로 추가 평가한다. 신호만으로는 차단하지 않는다.

  external_identifier   도구 인자에 외부 URL, 이메일, IP 주소, 식별 코드가 있다
  fake_boundary         입력에 여는 태그 없이 닫는 태그(</document> 등)가 있다. 문서가 끝난 것처럼 꾸민다
  user_impersonation    입력에 사용자를 자칭하는 문장이 있다 ("note from the user" 등)
  agent_directive       도구 결과에 에이전트를 향한 지시 표시가 있다 (<INFORMATION>, ignore previous instructions 등)
  inserted_document     사용자 메시지에 문서를 끼워 넣은 표시가 있다 ([Document 1], <document>, "the following document" 등)
"""
from __future__ import annotations

import re

_URL = re.compile(r"https?://[^\s\"'<>)\]]+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_IP = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
_CODE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]{2,}){2,}\b")
_CLOSING = re.compile(r"</\s*([A-Za-z][\w-]*)\s*>")
_IMPERSONATION = re.compile(
    r"\b(?:note|message|update|request|instruction|reminder)s?\s+from\s+(?:the\s+)?user\b"
    r"|\bthe\s+user\s+(?:says|said|wants\s+you\s+to|asked\s+you\s+to)\b"
    r"|\buser'?s\s+(?:note|instruction)\b",
    re.IGNORECASE,
)
_DIRECTIVE = re.compile(
    r"<\s*/?\s*(?:INFORMATION|IMPORTANT|INSTRUCTIONS?)\s*>"
    r"|\bignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\s+instructions\b"
    r"|\bmessage\s+(?:from|to)\s+the\s+(?:AI\s+)?(?:assistant|agent)\b"
    r"|\bbefore\s+(?:you\s+)?(?:solve|finish|answer)\w*\s+the\s+task\b",
    re.IGNORECASE,
)
_INSERTED = re.compile(
    r"\[\s*(?:document|doc|source|context|passage|reference)\s*\d+\s*\]"
    r"|<\s*(?:document|context|doc|source|passage)s?\b[^>]*>"
    r"|\b(?:the\s+)?(?:following|attached|uploaded|below)\s+(?:document|file|article|email|text|passage)s?\b"
    r"|^\s*(?:context|documents?|sources?|references?|retrieved\s+documents?)\s*:"
    r"|\[\s*검색된\s*문서\s*\]",
    re.IGNORECASE | re.MULTILINE,
)
_TOOL_RESULT_BLOCK = re.compile(r"\[tool result[^\]]*\][^\n]*\n(.*?)(?=\n\n\[[a-z]|\Z)", re.DOTALL)
# 대화 안의 메시지 경계. 시스템 프롬프트는 개발자가 쓴 것이라 신호를 찾지 않는다.
_ROLE_BLOCK = re.compile(
    r"(?:^|\n\n)\[(system|developer|user|assistant|tool result[^\]]*)\][^\n]*\n?(.*?)"
    r"(?=\n\n\[(?:system|developer|user|assistant|tool result)|\Z)",
    re.DOTALL,
)


def _untrusted_text(conversation: str) -> str:
    """사용자 메시지와 도구 결과만 모은다. 역할 표시가 없으면 전체를 본다."""
    blocks = _ROLE_BLOCK.findall(conversation)
    if not blocks:
        return conversation
    return "\n\n".join(text for role, text in blocks if role == "user" or role.startswith("tool result"))


def _quote(items: list[str], limit: int = 5) -> str:
    unique = list(dict.fromkeys(items))
    return ", ".join(repr(item[:80]) for item in unique[:limit])


def static_signals(conversation: str | None, tool_arguments: str, tool_result: str = "") -> list[str]:
    """찾은 신호를 사람이 읽을 수 있는 문장으로 돌려준다. 없으면 빈 목록."""
    full = conversation or ""
    conversation = _untrusted_text(full)
    signals: list[str] = []

    urls = [url.rstrip(".,;:!?") for url in _URL.findall(tool_arguments)]
    identifiers = urls + _EMAIL.findall(tool_arguments) + _IP.findall(tool_arguments) + _CODE.findall(tool_arguments)
    if identifiers:
        signals.append(f"external_identifier: the tool arguments contain {_quote(identifiers)}")

    orphans = []
    for tag in dict.fromkeys(name.lower() for name in _CLOSING.findall(conversation)):
        closing = len(re.findall(rf"</\s*{re.escape(tag)}\s*>", conversation, re.IGNORECASE))
        opening = len(re.findall(rf"<\s*{re.escape(tag)}(?:\s[^>]*)?>", conversation, re.IGNORECASE))
        if closing > opening:
            orphans.append(f"</{tag}>")
    if orphans:
        signals.append(f"fake_boundary: the agent's input closes {_quote(orphans)} without a matching opening tag")

    impersonation = [match.group(0) for match in _IMPERSONATION.finditer(conversation)]
    if impersonation:
        signals.append(f"user_impersonation: the agent's input contains text claiming to come from the user: {_quote(impersonation)}")

    inserted = [match.group(0).strip() for match in _INSERTED.finditer(conversation)]
    if inserted:
        signals.append(f"inserted_document: a user message contains inserted material: {_quote(inserted, 3)}")

    results = "\n".join(_TOOL_RESULT_BLOCK.findall(full)) + "\n" + (tool_result or "")
    directives = [match.group(0) for match in _DIRECTIVE.finditer(results)]
    if directives:
        signals.append(f"agent_directive: a tool result contains text addressed to the agent: {_quote(directives)}")
    return signals
