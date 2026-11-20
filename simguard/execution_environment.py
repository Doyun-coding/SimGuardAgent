"""
에이전트의 코드 실행 환경을 ExecutionSimulator에 알려 준다.

코드가 하는 일은 실행되는 환경의 작업 파일에 달려 있다. 빈 컨테이너에서 `make build`나
`bash scripts/x.sh`를 돌리면 "파일 없음"으로 끝나 Judge가 위협을 볼 수 없다. 그래서 코드 행동마다
새 Docker 컨테이너를 띄울 때 에이전트의 작업 파일을 복사해 넣고, 그 위에서 실행한다.
Docker 컨테이너는 격리돼 있고 실행이 끝나면 버리므로, 에이전트의 실제 파일은 바뀌지 않는다.

  ExecutionEnvironment(workdir="/home/me/project")                   # 에이전트가 이 폴더에서 실행
  ExecutionEnvironment(files={"/app/x.sh": "..."}, cwd="/app")       # 파일 내용을 직접 줌

workdir는 행동마다 그 순간의 폴더를 읽는다. 앞에서 허용돼 실제로 실행된 행동의 결과가 이미 들어 있다.
"""
from __future__ import annotations

import io
import tarfile
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache"}
_MAX_FILE_BYTES = 5 * 1024 * 1024
_MAX_TOTAL_BYTES = 200 * 1024 * 1024


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mode = 0o755 if name.endswith((".sh", ".py")) else 0o644
    info.mtime = int(time.time())
    tar.addfile(info, io.BytesIO(data))


class ExecutionEnvironment:
    """ExecutionSimulator가 에이전트와 같은 작업 파일 위에서 코드를 실행하게 한다."""

    def __init__(
        self,
        workdir: str | Path | None = None,
        files: Mapping[str, Any] | None = None,
        cwd: str | None = None,
    ) -> None:
        if workdir is None and files is None:
            raise ValueError("ExecutionEnvironment needs workdir or files")
        self.workdir = Path(workdir).resolve() if workdir is not None else None
        if self.workdir is not None and not self.workdir.is_dir():
            raise ValueError(f"workdir does not exist: {self.workdir}")
        self.files = dict(files or {})
        # 컨테이너 안에서 코드를 실행할 폴더. workdir는 같은 경로를 쓰는 게 자연스럽지만
        # Windows 경로는 컨테이너에서 쓸 수 없으므로 기본값은 /workspace다.
        self.cwd = cwd or "/workspace"

    def archive(self) -> bytes:
        """컨테이너의 루트(/)에 풀 tar. 행동마다 새로 만든다."""
        buffer, total = io.BytesIO(), 0
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            if self.workdir is not None:
                for path in sorted(self.workdir.rglob("*")):
                    relative = path.relative_to(self.workdir)
                    if any(part in _SKIP_DIRS for part in relative.parts) or not path.is_file():
                        continue
                    size = path.stat().st_size
                    if size > _MAX_FILE_BYTES or total + size > _MAX_TOTAL_BYTES:
                        continue
                    total += size
                    _add_bytes(tar, str(PurePosixPath(self.cwd) / relative.as_posix()).lstrip("/"), path.read_bytes())
            for raw_path, content in self.files.items():
                path = PurePosixPath(str(raw_path).replace("\\", "/"))
                if not path.is_absolute():
                    path = PurePosixPath(self.cwd) / path
                data = content.encode("utf-8") if isinstance(content, str) else bytes(content or b"")
                _add_bytes(tar, str(path).lstrip("/"), data)
        return buffer.getvalue()
