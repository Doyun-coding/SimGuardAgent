"""
코드 언어 표

인터셉터(코드 인식), SimulationManager(경로 선택), ExecutionSimulator(실행)가 같은 표를 쓴다.

  canonical(name)  언어 이름·별칭·코드 블록 표시("py", "rb", "golang", ...)를 표준 이름으로 바꾼다.
                   코드가 아닌 블록(json, yaml, text, ...)이면 None.
  spec(name)       표에 있는 언어면 실행 방법, 없으면 None.

표에 있는 언어는 그 언어의 Docker 이미지에서 실행한다. 표에 없는 언어(예: kotlin, cmd, sql)도
코드로 보고 ExecutionSimulator로 보내되, 실행하지 않고 내부 모델이 실행 결과를 예측한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SHELL = "shell"


@dataclass(frozen=True)
class LanguageSpec:
    image: str                  # 단발 실행에 쓰는 Docker 이미지
    ext: str                    # 코드를 저장할 파일 확장자
    run: str                    # sh -c로 실행할 명령. {f}는 코드 파일 경로
    mem: str = "128m"           # 단발 실행 메모리 한도
    aliases: tuple[str, ...] = field(default_factory=tuple)


LANGUAGES: dict[str, LanguageSpec] = {
    "python": LanguageSpec("python:3.11-slim", ".py", "python3 {f}",
                           aliases=("py", "python3", "ipython", "py3")),
    "javascript": LanguageSpec("node:20-slim", ".js", "node {f}",
                               aliases=("js", "node", "nodejs", "mjs", "cjs")),
    # 셸은 코딩 에이전트가 쓰는 개발 도구(make, gcc, git, curl, python)가 있는 이미지에서 실행한다.
    SHELL: LanguageSpec("python:3.12", ".sh", "bash {f}", mem="512m",
                        aliases=("bash", "sh", "zsh", "console", "shell-session", "terminal")),
    "ruby": LanguageSpec("ruby:3.3-slim", ".rb", "ruby {f}", aliases=("rb",)),
    "php": LanguageSpec("php:8.3-cli", ".php", "php {f}"),
    "perl": LanguageSpec("perl:5.38-slim", ".pl", "perl {f}", aliases=("pl",)),
    "go": LanguageSpec("golang:1.22", ".go",
                       "GOCACHE=/tmp/.sg_go/cache GOPATH=/tmp/.sg_go/path go run {f}",
                       mem="512m", aliases=("golang",)),
    "c": LanguageSpec("gcc:13", ".c", "gcc {f} -o /tmp/_sg_bin -lm && /tmp/_sg_bin", mem="256m"),
    "cpp": LanguageSpec("gcc:13", ".cpp", "g++ {f} -o /tmp/_sg_bin && /tmp/_sg_bin",
                        mem="512m", aliases=("c++", "cc", "cxx", "hpp")),
    "rust": LanguageSpec("rust:1-slim", ".rs", "rustc {f} -o /tmp/_sg_bin && /tmp/_sg_bin",
                         mem="512m", aliases=("rs",)),
    "java": LanguageSpec("eclipse-temurin:21-jdk", ".java", "java {f}", mem="512m"),
    "powershell": LanguageSpec("mcr.microsoft.com/powershell:lts-ubuntu-22.04", ".ps1",
                               "pwsh -NoProfile -NonInteractive -File {f}",
                               mem="256m", aliases=("pwsh", "ps1", "ps")),
}

# 코드 블록에 붙지만 실행할 코드가 아닌 표시
NON_CODE = {
    "json", "jsonc", "json5", "yaml", "yml", "toml", "ini", "cfg", "conf", "xml", "html", "htm", "css",
    "scss", "markdown", "md", "text", "txt", "plaintext", "plain", "csv", "tsv", "diff", "patch",
    "log", "output", "mermaid", "latex", "tex", "graphql", "proto", "dockerfile", "makefile",
    "env", "properties", "svg", "http", "regex", "jinja", "ebnf",
}

_ALIASES = {alias: name for name, item in LANGUAGES.items() for alias in (name, *item.aliases)}
_ALIASES.update({"cmd": "cmd", "bat": "cmd", "batch": "cmd", "typescript": "typescript", "ts": "typescript",
                 "kotlin": "kotlin", "kt": "kotlin", "csharp": "csharp", "c#": "csharp", "cs": "csharp",
                 "swift": "swift", "lua": "lua", "r": "r", "scala": "scala", "haskell": "haskell",
                 "sql": "sql", "vbscript": "vbscript", "vbs": "vbscript", "applescript": "applescript"})
# 실행 환경은 없지만 코드 블록 표시로 알아보는 언어 (실행 결과는 내부 모델이 예측한다)
_ALIASES.update({name: name for name in (
    "elixir", "erlang", "clojure", "dart", "julia", "groovy", "fsharp", "ocaml", "nim", "zig", "crystal",
    "tcl", "awk", "sed", "fish", "nushell", "vb", "vbnet", "objective-c", "objc", "asm", "assembly", "nasm",
    "solidity", "matlab", "octave", "prolog", "lisp", "scheme", "racket", "fortran", "cobol", "pascal",
    "delphi", "autohotkey", "ahk", "jq", "gdscript",
)})


def canonical(name: str | None) -> str | None:
    """언어 표시를 표준 이름으로. 코드가 아닌 표시나 빈 값이면 None. 모르는 표시는 소문자로 그대로."""
    key = (name or "").strip().lower()
    if not key or key in NON_CODE:
        return None
    return _ALIASES.get(key, key)


def is_known(name: str | None) -> bool:
    """코드 블록 표시가 알려진 프로그래밍 언어인지. 모르는 표시(예: 'example')는 코드로 보지 않는다."""
    return (name or "").strip().lower() in _ALIASES


def spec(name: str | None) -> LanguageSpec | None:
    return LANGUAGES.get(canonical(name) or "")


def is_shell(name: str | None) -> bool:
    return canonical(name) == SHELL


# 도구 이름에 들어 있으면 그 언어의 코드 실행 도구로 본다 ("run_ruby", "execute_php")
# 일반 단어와 겹치는 표시(go, c, r, sql, batch, ps 등)는 넣지 않는다. SQL 도구는 내부 DB 경로로 간다.
NAME_TOKENS = {
    "python", "python3", "ipython", "py", "javascript", "js", "node", "nodejs", "typescript",
    "ruby", "php", "perl", "golang", "rust", "java", "kotlin", "csharp", "swift", "lua", "scala",
    "haskell", "pwsh", "powershell",
}
