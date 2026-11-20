"""
명령이 실행하는 파일을 찾아 ExecutionJudge에 내용을 보여 준다.

`npm run lint`, `pip install -e .`, `make`, `bash scripts/x.sh`는 명령 문자열만 보면 평범하지만, 실제로
실행되는 것은 package.json의 스크립트, setup.py, Makefile, 스크립트 파일 안의 명령이다. 샌드박스는
네트워크가 막혀 있어 그 명령이 실패해도 출력에 흔적이 거의 남지 않는다. 그래서 실행 전 작업 파일에서
그 파일들을 읽어 Judge에 준다. 에이전트의 작업 폴더에 실제로 있는 파일만 읽는다.

  run_files_text(code, read)   read(path) -> bytes | None  (상대 경로는 작업 폴더 기준)
"""
from __future__ import annotations

import re
from collections.abc import Callable

_FILE_CHARS = 3_000
_TOTAL_CHARS = 9_000
_MAX_FILES = 8

# 빌드·설치·테스트 도구 -> 그 도구가 실행하는 정의 파일
_TOOL_FILES = [
    (r"\b(?:npm|yarn|pnpm|npx)\b", ["package.json"]),
    (r"\bmake\b", ["Makefile", "makefile", "GNUmakefile"]),
    (r"\bpip3?\s+install\b[^\n;&|]*(?:\s-e\b|\s\.(?:\s|$)|\s\.\[)|\bsetup\.py\b|\bpoetry\s+install\b|\b-m\s+build\b",
     ["setup.py", "pyproject.toml", "setup.cfg"]),
    (r"\b(?:pytest|tox)\b", ["conftest.py", "tox.ini", "pytest.ini"]),
    (r"\bpre-commit\b", [".pre-commit-config.yaml"]),
    (r"\bgit\s+commit\b", [".git/hooks/pre-commit", ".git/hooks/commit-msg"]),
    (r"\bgit\s+push\b", [".git/hooks/pre-push"]),
    (r"\bdocker(?:-compose|\s+compose|\s+build)\b", ["Dockerfile", "docker-compose.yml", "compose.yaml"]),
    (r"\bcargo\s+(?:build|run|test)\b", ["build.rs"]),
]
# 인터프리터에 넘긴 파일, ./로 실행한 파일, pip -r로 읽는 파일
_ARG_FILES = [
    r"(?:^|[\s;&|(`])(?:bash|sh|zsh|python3?|node|ruby|perl|source|\.)\s+(?:-[A-Za-z]\S*\s+)*([^\s;&|)`'\"-][^\s;&|)`'\"]*)",
    r"(?:^|[\s;&|(`])(\./[^\s;&|)`'\"]+)",
    r"\bpip3?\s+install\b[^\n;&|]*?\s-[rc]\s+([^\s;&|]+)",
]
# 처음 찾은 파일이 부르는 스크립트와 따르는 설정 파일
_SCRIPT = re.compile(r"[\w./-]+\.(?:sh|bash|py|js|mjs|cjs|ts|rb|pl|json|ya?ml|toml|ini|conf|cfg)\b")


def files_run_by(code: str) -> list[str]:
    """명령이 실행하거나 설치 중에 읽는 파일 경로 후보 (상대 경로는 작업 폴더 기준)."""
    found: list[str] = []
    for pattern, names in _TOOL_FILES:
        if re.search(pattern, code):
            found.extend(names)
    for pattern in _ARG_FILES:
        found.extend(match.group(1) for match in re.finditer(pattern, code, flags=re.MULTILINE))
    return list(dict.fromkeys(path for path in found if "://" not in path and path not in {".", "-"}))


def run_files_text(code: str, read: Callable[[str], bytes | None]) -> str:
    """실행되는 파일들의 내용. 그 파일이 다시 부르는 스크립트와 설정 파일도 한 단계 더 읽는다. 없으면 빈 문자열."""
    queue, seen, parts, total = files_run_by(code), set(), [], 0
    hop = len(queue)
    index = 0
    while index < len(queue) and len(parts) < _MAX_FILES and total < _TOTAL_CHARS:
        path = queue[index]
        index += 1
        if path in seen:
            continue
        seen.add(path)
        try:
            data = read(path)
        except Exception:
            data = None
        if data is None:
            continue
        text = data.decode("utf-8", errors="replace")[:_FILE_CHARS]
        parts.append(f"=== {path}\n{text}")
        total += len(text)
        if index <= hop:  # 처음 찾은 파일 안에서 부르는 스크립트까지만 따라간다
            queue.extend(p for p in _SCRIPT.findall(text) if p not in seen and "://" not in p)
    return "\n\n".join(parts)[:_TOTAL_CHARS]
