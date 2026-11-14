"""
에이전트의 코드 실행 환경을 ExecutionSimulator에 맞춰 준다.

코드가 하는 일은 실행되는 환경의 상태(작업 파일, 설치된 도구)에 달려 있다. 빈 컨테이너에서
`bash scripts/x.sh`를 돌리면 "파일 없음"으로 끝나 시뮬레이션이 의미가 없다. 그래서 코드를
시뮬레이션할 때만 컨테이너를 띄우고, 처음 띄울 때 에이전트의 작업 파일을 넣는다. 에이전트는
어디서 돌든 바꿀 필요가 없다.

  env = ExecutionEnvironment(workdir="/home/me/project")                   # 에이전트가 이 폴더에서 실행
  env = ExecutionEnvironment(files={"/app/x.sh": "..."}, cwd="/app")       # 파일 내용을 직접 줌

컨테이너는 대화마다 하나이고 대화가 이어지는 동안 재사용한다. 시뮬레이션에서 실행한 명령은
그 컨테이너에 그대로 남으므로, ALLOW되면 에이전트의 실제 환경과 같은 상태가 된다. BLOCK되면
실제 환경에는 일어나지 않은 변화가 남으므로 컨테이너를 버리고, 다음 코드 행동 때 다시 만든다.
workdir 방식은 그때 실제 폴더를 다시 복사하고, files 방식은 처음 파일에 허용된 행동을 다시 적용한다.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import tarfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from simguard.models import Action, ActionType, Decision, InterceptionResult

logger = logging.getLogger(__name__)

# 개발 도구(gcc, make, git, curl)가 들어 있는 이미지. 가벼운 이미지는 image로 바꾼다.
DEFAULT_IMAGE = "python:3.12"
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache"}
_MAX_FILE_BYTES = 5 * 1024 * 1024
_MAX_TOTAL_BYTES = 200 * 1024 * 1024
_SHELL_LANGUAGES = {"bash", "sh", "shell", "zsh"}


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mode = 0o755 if name.endswith((".sh", ".py")) else 0o644
    info.mtime = int(time.time())
    tar.addfile(info, io.BytesIO(data))


def _files_archive(files: Mapping[str, Any], cwd: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for raw_path, content in files.items():
            path = PurePosixPath(str(raw_path).replace("\\", "/"))
            if not path.is_absolute():
                path = PurePosixPath(cwd) / path
            data = content.encode("utf-8") if isinstance(content, str) else bytes(content or b"")
            _add_bytes(tar, str(path).lstrip("/"), data)
    return buffer.getvalue()


def _workdir_archive(workdir: Path, cwd: str) -> bytes:
    buffer, total = io.BytesIO(), 0
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path in sorted(workdir.rglob("*")):
            relative = path.relative_to(workdir)
            if any(part in _SKIP_DIRS for part in relative.parts) or not path.is_file():
                continue
            size = path.stat().st_size
            if size > _MAX_FILE_BYTES or total + size > _MAX_TOTAL_BYTES:
                continue
            total += size
            _add_bytes(tar, str(PurePosixPath(cwd) / relative.as_posix()).lstrip("/"), path.read_bytes())
    return buffer.getvalue()


@dataclass
class _Conversation:
    session: Any = None
    allowed: list[Action] = field(default_factory=list)
    last_used: float = field(default_factory=time.monotonic)


class ExecutionEnvironment:
    """ExecutionSimulator가 에이전트와 같은 파일 위에서 코드를 실행하게 한다."""

    def __init__(
        self,
        workdir: str | Path | None = None,
        files: Mapping[str, Any] | None = None,
        cwd: str | None = None,
        image: str = DEFAULT_IMAGE,
        max_sessions: int = 8,
        ttl_seconds: float = 900,
    ) -> None:
        if workdir is None and files is None:
            raise ValueError("ExecutionEnvironment needs workdir or files")
        self.workdir = Path(workdir).resolve() if workdir is not None else None
        if self.workdir is not None and not self.workdir.is_dir():
            raise ValueError(f"workdir does not exist: {self.workdir}")
        self.files = dict(files or {})
        self.cwd = cwd or "/workspace"
        self.image = image
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._conversations: dict[str, _Conversation] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- keys

    @staticmethod
    def conversation_key(messages: list[dict]) -> str:
        """대화를 시작한 시스템 메시지와 첫 사용자 메시지로 대화를 구분한다."""
        head = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            head.append({"role": message.get("role"), "content": message.get("content")})
            if message.get("role") == "user":
                break
        return hashlib.sha256(json.dumps(head, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()

    @staticmethod
    def is_new_conversation(messages: list[dict]) -> bool:
        return not any(isinstance(message, dict) and message.get("role") == "assistant" for message in messages)

    # ---------------------------------------------------------------- sessions

    def _archive(self) -> bytes:
        if self.workdir is not None:
            return _workdir_archive(self.workdir, self.cwd)
        return _files_archive(self.files, self.cwd)

    def _create(self, conversation: _Conversation) -> Any:
        from simguard.simulation.backends.execution_simulator import MicroVMSession

        session = MicroVMSession.create(self.image, self.cwd)
        if session is None:
            return None
        session.seed_archive(self._archive())
        if self.workdir is None:
            # 파일 내용은 처음 상태이므로 지금까지 허용된 행동을 다시 적용해 실제 상태와 맞춘다.
            for action in conversation.allowed:
                if action.action_type == ActionType.SHELL or action.language.lower() in _SHELL_LANGUAGES:
                    session.exec_shell(action.code)
                else:
                    session.exec_code(action.code, action.language)
        return session

    @staticmethod
    def _stop(conversation: _Conversation) -> None:
        session, conversation.session = conversation.session, None
        if session is not None:
            try:
                session.stop()
            except Exception as exc:
                logger.warning("failed to stop simulation container: %s", type(exc).__name__)

    def _evict(self) -> None:
        now = time.monotonic()
        stale = [key for key, item in self._conversations.items() if now - item.last_used > self.ttl_seconds]
        overflow = sorted(self._conversations, key=lambda key: self._conversations[key].last_used)
        stale += overflow[: max(0, len(self._conversations) - self.max_sessions)]
        for key in dict.fromkeys(stale):
            conversation = self._conversations.pop(key, None)
            if conversation is not None:
                self._stop(conversation)

    def _conversation(self, key: str, is_new: bool) -> _Conversation:
        with self._lock:
            if is_new and key in self._conversations:
                self._stop(self._conversations.pop(key))
            conversation = self._conversations.setdefault(key, _Conversation())
            conversation.last_used = time.monotonic()
            self._evict()
            return conversation

    def session_sync(self, key: str, is_new: bool = False) -> Any:
        conversation = self._conversation(key, is_new)
        with self._lock:
            if conversation.session is None:
                conversation.session = self._create(conversation)
            return conversation.session

    async def session(self, key: str, is_new: bool = False) -> Any:
        return await asyncio.to_thread(self.session_sync, key, is_new)

    def factory(self, messages: list[dict]):
        """intercept_output의 execution_session_factory로 넘길 함수. 코드 행동이 나올 때만 불린다."""
        key, is_new = self.conversation_key(messages), self.is_new_conversation(messages)
        return lambda: self.session(key, is_new)

    def record(self, messages: list[dict], result: InterceptionResult) -> None:
        """판정 결과를 반영한다. BLOCK이면 컨테이너를 버리고, ALLOW면 허용된 코드 행동을 기억한다."""
        key = self.conversation_key(messages)
        with self._lock:
            conversation = self._conversations.get(key)
            if conversation is None:
                return
            if result.decision == Decision.BLOCK:
                self._stop(conversation)
            else:
                conversation.allowed.extend(
                    trace.action.model_copy(update={"execution_session": None})
                    for trace in result.traces
                    if trace.action.code
                )

    def close(self) -> None:
        with self._lock:
            conversations, self._conversations = list(self._conversations.values()), {}
        for conversation in conversations:
            self._stop(conversation)

    def __enter__(self) -> "ExecutionEnvironment":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
