"""
CodeSimulator

Shell·Code를 격리된 환경(Docker/subprocess)에서 실행한다.
평가는 CodeJudge에 위임한다.

격리 전략 (우선순위):
  1. Docker container — 네트워크 차단 / read-only fs / non-root / 메모리·CPU 제한
  2. subprocess + venv — Docker 미설치 시 fallback

사용 대상:
  ActionType.CODE_EXECUTION — Python·JS·Bash 코드 스니펫
  ActionType.SHELL          — Shell 명령 (bash -c "...")
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import venv
from pathlib import Path

from simguard.models import Action, ActionType, SimulationBackend, SimulationTrace
from simguard.simulation.judges.code_judge import CodeJudge

EXEC_TIMEOUT    = 15   # 초, 단일 exec
SESSION_TIMEOUT = 300  # 초, 세션 전체 수명

# ─────────────────────────────────────────────────────────────────────────────
# Docker 이미지 맵
# ─────────────────────────────────────────────────────────────────────────────
_IMAGES = {
    "python": "python:3.11-slim",
    "py":     "python:3.11-slim",
    "javascript": "node:20-slim",
    "js":     "node:20-slim",
    "node":   "node:20-slim",
    "bash":   "ubuntu:22.04",
    "sh":     "ubuntu:22.04",
    "shell":  "ubuntu:22.04",
}


# ─────────────────────────────────────────────────────────────────────────────
# MicroVMSession — 프롬프트 하나의 수명 동안 유지되는 격리 컨테이너
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


class MicroVMSession:
    """
    프롬프트 입력 시 생성 → 에이전트 종료 시 삭제되는 격리 컨테이너 세션.

    하나의 컨테이너 안에서 모든 코드/Shell 실행이 누적되므로:
      - step 1에서 설치한 패키지를 step 3에서 사용 가능
      - 파일시스템 변경이 session 전체에 걸쳐 추적됨 (cross-layer 분석 기반)
      - 컨테이너는 network_mode=none 이므로 외부 통신 불가

    사용법:
      session = MicroVMSession.create()   # 프롬프트 시작 시
      result  = session.exec_code(code)   # 각 tool call 시
      diff    = session.stop()            # 에이전트 종료 시 → 전체 fs diff 반환
    """

    _DEFAULT_IMAGE = "python:3.11-slim"

    def __init__(self, client, image: str):
        self._client    = client
        self._image     = image
        self._container = None
        self._exec_log: list[dict] = []   # 모든 실행 기록 (cross-layer 분석용)

    # ── 생성 / 종료 ────────────────────────────────────────────────────────────

    _warned = False

    @classmethod
    def create(cls, image: str = _DEFAULT_IMAGE) -> "MicroVMSession | None":
        """Docker 가용 시 세션 시작. 없으면 None (subprocess fallback 사용).

        실패를 조용히 삼키면 격리 없이 돌고 있다는 사실이 드러나지 않으므로,
        프로세스당 한 번은 반드시 경고를 남긴다.
        """
        try:
            import docker
            client = docker.from_env()
            client.ping()
            session = cls(client, image)
            session._start()
            return session
        except Exception as e:
            if not cls._warned:
                cls._warned = True
                logger.warning(
                    "MicroVM 사용 불가 — 격리 없이 subprocess로 실행됩니다. 원인: %s: %s",
                    type(e).__name__, str(e)[:200],
                )
            return None

    def _start(self):
        import docker.errors
        try:
            self._client.images.get(self._image)
        except docker.errors.ImageNotFound:
            self._client.images.pull(self._image)

        self._container = self._client.containers.run(
            self._image,
            command=["tail", "-f", "/dev/null"],   # 세션 유지용 슬립 프로세스
            network_mode="none",
            mem_limit="256m",
            cpu_quota=50000, cpu_period=100000,
            security_opt=["no-new-privileges:true"],
            detach=True,
            remove=False,
        )

    def stop(self) -> list[dict]:
        """세션 종료. 세션 전체 파일시스템 diff를 반환하고 컨테이너 삭제."""
        fs_changes = []
        if self._container:
            try:
                fs_changes = self._container.diff() or []
                self._container.stop(timeout=5)
                self._container.remove(force=True)
            except Exception:
                pass
            self._container = None
        return fs_changes

    # ── 실행 ──────────────────────────────────────────────────────────────────

    def exec_code(self, code: str, language: str = "python") -> dict:
        """코드를 컨테이너에서 실행하고 결과 반환."""
        self._put_file("/tmp/_sg_code.py", code)
        lang = language.lower()
        if lang in ("python", "py"):
            cmd = ["python3", "/tmp/_sg_code.py"]
        elif lang in ("javascript", "js", "node"):
            cmd = ["node", "/tmp/_sg_code.py"]
        else:
            cmd = ["bash", "/tmp/_sg_code.py"]
        return self._exec(cmd)

    def exec_shell(self, command: str) -> dict:
        """Shell 명령을 컨테이너에서 실행하고 결과 반환."""
        return self._exec(["bash", "-c", command])

    def snapshot_diff(self) -> list[dict]:
        """현재까지의 파일시스템 변경 목록 (세션 종료 없이 중간 확인)."""
        if self._container:
            return self._container.diff() or []
        return []

    @property
    def exec_log(self) -> list[dict]:
        """세션 내 모든 실행 기록 (cross-layer 분석용)."""
        return self._exec_log

    # ── 내부 ──────────────────────────────────────────────────────────────────

    def _exec(self, cmd: list) -> dict:
        start = time.time()
        try:
            exit_code, output = self._container.exec_run(
                cmd, demux=True, tty=False
            )
            stdout = (output[0] or b"").decode("utf-8", errors="replace")[:12000]
            stderr = (output[1] or b"").decode("utf-8", errors="replace")[:8000]
            timed_out = False
        except Exception as e:
            exit_code, stdout, stderr, timed_out = -1, "", str(e), True

        result = {
            "stdout":     stdout,
            "stderr":     stderr,
            "returncode": exit_code,
            "fs_changes": [],          # 세션 종료 시 전체 diff로 대체
            "network_status": "blocked (network_mode=none)",
            "timed_out":  timed_out,
            "elapsed":    round(time.time() - start, 2),
            "backend":    "docker-session",
        }
        self._exec_log.append(result)
        return result

    def _put_file(self, path: str, content: str):
        import io, tarfile
        from pathlib import Path
        data = content.encode("utf-8")
        buf  = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=Path(path).name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        buf.seek(0)
        self._container.put_archive(str(Path(path).parent), buf)


class CodeSimulator:
    """
    Docker container를 microVM으로 사용하는 격리 실행 백엔드.
    Docker 미설치 시 subprocess + venv로 자동 fallback.
    """

    def __init__(self):
        self._docker = self._init_docker()
        self._judge  = CodeJudge()

    # ──────────────────────────────────────────────────────────────────────────
    # 공개 인터페이스
    # ──────────────────────────────────────────────────────────────────────────

    async def simulate(
        self,
        action: Action,
        session: "MicroVMSession | None" = None,
    ) -> SimulationTrace:
        code     = action.code or ""
        language = action.language or "python"
        is_shell = action.action_type == ActionType.SHELL or language.lower() in ("bash", "sh", "shell")

        if session:
            # 세션 모드: 기존 컨테이너에서 실행 (상태 누적)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                session.exec_shell if is_shell else session.exec_code,
                code,
                *(() if is_shell else (language,)),
            )
        else:
            # 단발 모드: 요청마다 새 컨테이너/venv
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._run, code, language, is_shell
            )

        events, risk_score = await self._judge.judge(action, result)

        return SimulationTrace(
            action=action,
            backend_used=SimulationBackend.ISOLATED,
            simulated_output=_fmt(result),
            security_events=events,
            risk_score=risk_score,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 실행 (동기, executor thread에서 호출)
    # ──────────────────────────────────────────────────────────────────────────

    def _run(self, code: str, language: str, is_shell: bool) -> dict:
        if self._docker:
            return self._docker_run(code, language, is_shell)
        return self._subprocess_run(code, language)

    # ── Docker backend ────────────────────────────────────────────────────────

    def _docker_run(self, code: str, language: str, is_shell: bool) -> dict:
        import docker.errors

        image = _IMAGES.get(language.lower(), "python:3.11-slim")

        if is_shell:
            image = "ubuntu:22.04"
            cmd   = ["bash", "-c", code]
        elif language.lower() in ("python", "py"):
            cmd = ["python", "-c", code]
        elif language.lower() in ("javascript", "js", "node"):
            cmd = ["node", "-e", code]
        else:
            cmd = ["bash", "-c", code]

        # 이미지 pull (없을 때만)
        try:
            self._docker.images.get(image)
        except docker.errors.ImageNotFound:
            self._docker.images.pull(image)

        start = time.time()
        container = None
        try:
            container = self._docker.containers.run(
                image=image,
                command=cmd,
                network_mode="none",                     # 네트워크 완전 차단
                mem_limit="128m",                        # 메모리 제한
                cpu_quota=50000, cpu_period=100000,      # CPU 50%
                read_only=True,                          # 루트 fs 읽기 전용
                tmpfs={"/tmp": "size=32m,exec"},         # 실행 가능한 임시 공간
                security_opt=["no-new-privileges:true"], # 권한 상승 차단
                remove=False,
                detach=True,
                stdout=True,
                stderr=True,
            )

            try:
                exit_info = container.wait(timeout=EXEC_TIMEOUT)
                timed_out = False
            except Exception:
                container.kill()
                timed_out = True
                exit_info = {"StatusCode": -1}

            stdout     = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")[:12000]
            stderr_log = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")[:8000]
            fs_changes = container.diff() or []

        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

        return {
            "stdout":         stdout,
            "stderr":         stderr_log,
            "returncode":     exit_info.get("StatusCode", -1),
            "fs_changes":     fs_changes,        # [{"Kind": 0|1|2, "Path": "..."}, ...]
            "network_status": "blocked (network_mode=none)",
            "timed_out":      timed_out,
            "elapsed":        round(time.time() - start, 2),
            "backend":        "docker",
        }

    # ── subprocess fallback ───────────────────────────────────────────────────

    def _subprocess_run(self, code: str, language: str) -> dict:
        workspace = Path(tempfile.mkdtemp(prefix="simguard_vm_"))
        start     = time.time()
        try:
            venv_dir = workspace / "venv"
            venv.create(str(venv_dir), with_pip=False, clear=True)
            python_exe = (
                venv_dir / "Scripts" / "python.exe"
                if sys.platform == "win32"
                else venv_dir / "bin" / "python"
            )
            code_file = workspace / "code.py"
            code_file.write_text(code, encoding="utf-8")

            try:
                proc = subprocess.run(
                    [str(python_exe), str(code_file)],
                    capture_output=True, text=True,
                    timeout=EXEC_TIMEOUT,
                    cwd=str(workspace), env={},
                )
                timed_out  = False
                stdout     = proc.stdout[:3000]
                stderr_log = proc.stderr[:3000]
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out  = True
                stdout     = ""
                stderr_log = f"[timeout after {EXEC_TIMEOUT}s]"
                returncode = -1

            fs_changes = [
                {"Kind": 1, "Path": str(p.relative_to(workspace))}
                for p in workspace.rglob("*")
                if p.is_file()
                and not str(p).startswith(str(venv_dir))
                and p.name != "code.py"
            ]

            return {
                "stdout":         stdout,
                "stderr":         stderr_log,
                "returncode":     returncode,
                "fs_changes":     fs_changes,
                "network_status": "not isolated (no Docker)",
                "timed_out":      timed_out,
                "elapsed":        round(time.time() - start, 2),
                "backend":        "subprocess",
            }
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Docker 초기화
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _init_docker():
        try:
            import docker
            client = docker.from_env()
            client.ping()
            return client
        except Exception:
            return None



def _fmt(result: dict) -> str:
    parts = [f"[backend: {result.get('backend', '?')}  elapsed: {result.get('elapsed', '?')}s]"]
    if result["stdout"]:
        parts.append(f"[stdout]\n{result['stdout']}")
    if result["stderr"]:
        parts.append(f"[stderr]\n{result['stderr']}")
    if result["fs_changes"]:
        paths = [c.get("Path", str(c)) for c in result["fs_changes"]]
        parts.append(f"[fs_changes]\n" + "\n".join(paths[:20]))
    parts.append(f"[network] {result.get('network_status', '?')}")
    return "\n\n".join(parts)
