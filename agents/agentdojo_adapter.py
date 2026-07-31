#!/usr/bin/env python3
"""Launch the AgentDojo/SABER adapter with its local vLLM dependency.

Usage:
    python ./agents/agentdojo_adapter.py --benchmark SABER ...

The launcher:
1. re-executes itself with the project .venv when necessary,
2. starts the Docker Compose model-server and waits for its health check,
3. injects the local OpenAI-compatible endpoint defaults, and
4. runs the real benchmark adapter in the current process.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
ADAPTER = (
    REPO_ROOT
    / "attacks-bench"
    / "agentdojo_adapter"
    / "agentdojo_adapter.py"
)


def _using_project_venv() -> bool:
    return Path(sys.executable).resolve() == VENV_PYTHON.resolve()


def _ensure_project_venv() -> None:
    if _using_project_venv():
        return
    if not VENV_PYTHON.is_file():
        raise SystemExit(f"Project Python not found: {VENV_PYTHON}")
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])


def _start_model_server() -> None:
    command = [
        "docker",
        "compose",
        "up",
        "-d",
        "--wait",
        "model-server",
    ]
    try:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    except FileNotFoundError:
        raise SystemExit("Docker CLI was not found.") from None
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Could not start the Docker model-server (exit={exc.returncode})."
        ) from None


def _with_local_model_defaults(arguments: list[str]) -> list[str]:
    result = list(arguments)
    defaults = (
        ("--openai-base-url", "http://127.0.0.1:8000/v1"),
        ("--openai-api-key", "dummy"),
        ("--model-id", "qwen-local"),
    )
    for option, value in defaults:
        if option not in result:
            result.extend((option, value))
    return result


def main() -> None:
    _ensure_project_venv()
    _start_model_server()
    arguments = _with_local_model_defaults(sys.argv[1:])
    os.chdir(REPO_ROOT)
    os.execv(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(ADAPTER), *arguments],
    )


if __name__ == "__main__":
    main()
