"""
Isolated Execution Backend

요청마다 새 Python venv를 생성하여 코드를 격리 실행한다.
실제 환경에 영향을 주지 않고 코드의 동작과 부작용을 시뮬레이션한다.

흐름:
  1. 임시 작업 디렉터리 생성
  2. 그 안에 신선한 venv 생성 (clear=True → 항상 깨끗한 상태)
  3. venv Python으로 코드 실행 (TIMEOUT 이내)
  4. stdout / stderr / 파일시스템 변경 캡처
  5. 결과 분석 → SecurityEvent + risk_score
  6. 임시 디렉터리 삭제 (예외 발생해도 항상)
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

from simguard.models import Action, SecurityEvent, SimulationBackend, SimulationTrace

TIMEOUT = 10  # 초

_DANGEROUS_PATTERNS = [
    # 시스템 파괴
    "rm -rf", "del /f /s", "format c:", "shutdown",
    # 코드 실행 확장
    "exec(", "eval(", "compile(",
    # 네트워크
    "urllib", "requests", "httpx", "socket", "subprocess",
    # 민감 경로 접근
    "/etc/passwd", "/etc/shadow", "c:\\windows\\system32",
    # 환경 변수 탈취
    "os.environ", "getenv",
]


class IsolatedExecutor:
    async def simulate(self, action: Action) -> SimulationTrace:
        code     = action.code or action.user_request
        language = action.language

        result = await asyncio.get_event_loop().run_in_executor(
            None, self._run_isolated, code, language
        )

        events, risk_score = self._analyze(result)

        return SimulationTrace(
            action=action,
            backend_used=SimulationBackend.ISOLATED,
            simulated_output=self._format_output(result),
            security_events=events,
            risk_score=risk_score,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 격리 실행 (동기, executor 내에서 실행)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_isolated(self, code: str, language: str) -> dict:
        workspace = Path(tempfile.mkdtemp(prefix="simguard_iso_"))
        try:
            return self._execute(workspace, code, language)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def _execute(self, workspace: Path, code: str, language: str) -> dict:
        # 1. 신선한 venv 생성
        venv_dir = workspace / "venv"
        venv.create(str(venv_dir), with_pip=False, clear=True)

        python_exe = (
            venv_dir / "Scripts" / "python.exe"
            if sys.platform == "win32"
            else venv_dir / "bin" / "python"
        )

        # 2. 코드 파일 저장
        code_file = workspace / f"code.{self._ext(language)}"
        code_file.write_text(code, encoding="utf-8")

        # 3. 실행
        try:
            proc = subprocess.run(
                [str(python_exe), str(code_file)],
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                cwd=str(workspace),
                env={},   # 환경 변수 격리
            )
            timed_out = False
            stdout    = proc.stdout[:2000]
            stderr    = proc.stderr[:2000]
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out  = True
            stdout     = ""
            stderr     = f"[timeout after {TIMEOUT}s]"
            returncode = -1

        # 4. 파일시스템 변경 감지 (venv 제외)
        fs_changes = [
            str(p.relative_to(workspace))
            for p in workspace.rglob("*")
            if p.is_file()
            and not str(p).startswith(str(venv_dir))
            and p.name != code_file.name
        ]

        return {
            "stdout":     stdout,
            "stderr":     stderr,
            "returncode": returncode,
            "fs_changes": fs_changes,
            "timed_out":  timed_out,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 결과 분석
    # ──────────────────────────────────────────────────────────────────────────

    def _analyze(self, result: dict) -> tuple[list[SecurityEvent], float]:
        events:     list[SecurityEvent] = []
        risk_score: float               = 0.0

        combined = (result["stdout"] + result["stderr"]).lower()

        # 위험 패턴 검사
        for pattern in _DANGEROUS_PATTERNS:
            if pattern.lower() in combined:
                events.append(SecurityEvent(
                    type="dangerous_pattern",
                    location="code",
                    severity="high",
                    evidence=f"Pattern '{pattern}' detected in execution output",
                ))
                risk_score = max(risk_score, 0.85)

        # 파일시스템 변경
        if result["fs_changes"]:
            events.append(SecurityEvent(
                type="filesystem_change",
                location="code",
                severity="medium",
                evidence=f"Files created/modified: {result['fs_changes']}",
            ))
            risk_score = max(risk_score, 0.5)

        # 타임아웃 (무한루프 / 리소스 고갈 시도)
        if result["timed_out"]:
            events.append(SecurityEvent(
                type="execution_timeout",
                location="code",
                severity="medium",
                evidence=f"Code execution exceeded {TIMEOUT}s timeout",
            ))
            risk_score = max(risk_score, 0.4)

        # 비정상 종료
        if result["returncode"] not in (0, -1):
            events.append(SecurityEvent(
                type="runtime_error",
                location="code",
                severity="low",
                evidence=f"Exit code {result['returncode']}: {result['stderr'][:200]}",
            ))

        return events, risk_score

    def _format_output(self, result: dict) -> str:
        parts = []
        if result["stdout"]:
            parts.append(f"[stdout]\n{result['stdout']}")
        if result["stderr"]:
            parts.append(f"[stderr]\n{result['stderr']}")
        if result["fs_changes"]:
            parts.append(f"[fs_changes]\n{result['fs_changes']}")
        return "\n\n".join(parts) or "(no output)"

    @staticmethod
    def _ext(language: str) -> str:
        return {"python": "py", "bash": "sh", "javascript": "js"}.get(language, "txt")
