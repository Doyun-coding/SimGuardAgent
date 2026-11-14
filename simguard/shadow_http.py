"""
LiveReadExecutor: 도구를 GET 전용 모드로 실제 실행한다.

어떤 도구가 어떤 요청을 보낼지 미리 알 수 없으므로, 도구를 분류하지 않고 실제로 실행하되 나가는 요청을
하나씩 검사한다.

  GET·HEAD·OPTIONS HTTP 요청          실제로 보낸다. 실제 응답이 도구 결과가 된다.
  POST·PUT·PATCH·DELETE 등           보내기 전에 실행을 멈춘다.
  HTTP가 아닌 부작용                   멈춘다: 파일 쓰기·삭제, 프로세스 실행, 80·443 외 포트 연결(DB, SMTP 등),
                                     SQLite 연결, 환경 변수 변경
  허용 도메인 밖 (allowed_hosts를 준 경우)  멈춘다.

멈추면 shadow_execute가 None을 돌려주고, ToolCallSimulator는 내부 모델로 결과를 예상한다. 멈추기 전에
보낸 GET은 이미 나간 상태다.

검사하는 곳
  http.client.HTTPConnection.putrequest   requests, urllib3, urllib.request가 모두 지난다
  httpx.Client.send, httpx.AsyncClient.send
  aiohttp.ClientSession._request
  sys.addaudithook                       파일, 프로세스, 소켓 연결 등 파이썬 수준의 부작용
검사는 SimGuardAgent가 판정용으로 실행하는 호출(contextvar)에만 걸린다. 에이전트가 실제로 실행할 때는
걸리지 않는다. 도구가 만든 스레드와 ThreadPoolExecutor 작업에도 이어진다.

보지 못하는 것: C 확장이 직접 하는 네트워크·파일 작업(psycopg 같은 DB 드라이버, gRPC), 이미 열려 있던
연결로 보내는 데이터, 서버 쪽에서 상태를 바꾸는 GET(잘못 만든 API). MCP 도구는 서버 프로세스가 요청을
보내므로 여기서 실행하지 않는다.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import functools
import http.client
import inspect
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from simguard.shadow_api import ShadowExecutionResult

_READ_METHODS = {"GET", "HEAD", "OPTIONS"}
_MAX_RESULT = 8000
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_STOP_EVENTS = {
    "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.truncate", "os.link",
    "os.symlink", "os.utime", "os.chflags", "os.lchflags", "os.lchmod", "os.kill", "os.killpg", "os.fork",
    "os.forkpty", "os.system", "os.startfile", "os.putenv", "os.unsetenv", "subprocess.Popen",
    "_winapi.CreateProcess", "shutil.rmtree", "shutil.copyfile", "shutil.copymode", "shutil.copystat",
    "shutil.copytree", "shutil.move", "shutil.chown", "shutil.make_archive", "shutil.unpack_archive",
    "sqlite3.connect", "ftplib.connect", "smtplib.connect", "smtplib.send", "poplib.connect", "imaplib.open",
    "nntplib.connect", "telnetlib.Telnet.open", "webbrowser.open", "socket.bind", "socket.sendto",
    "socket.sendmsg", "winreg.CreateKey", "winreg.DeleteKey", "winreg.DeleteValue", "winreg.SetValue",
    "winreg.SetValueEx", "winreg.SaveKey",
}
_STOP_PREFIXES = ("os.exec", "os.spawn", "os.posix_spawn")

_config: dict[str, Any] = {"enabled": True, "allowed_hosts": None, "ports": frozenset({80, 443}), "timeout": 30.0}


class LiveReadStopped(BaseException):
    """GET 전용 실행 중 허용되지 않은 동작이 나와 멈춘다. 도구의 except Exception에 잡히지 않게 BaseException이다."""


@dataclass
class _Guard:
    allowed_hosts: frozenset | None
    ports: frozenset
    requests: list[str] = field(default_factory=list)
    stopped: str | None = None


_GUARD: contextvars.ContextVar[_Guard | None] = contextvars.ContextVar("simguard_live_read", default=None)


def _stop(guard: _Guard, reason: str) -> None:
    if guard.stopped is None:
        guard.stopped = reason
    raise LiveReadStopped(guard.stopped)


def _host_allowed(host: str, allowed: frozenset) -> bool:
    return any(host == item or host.endswith("." + item) for item in allowed)


def _check_request(method: Any, url: str) -> None:
    guard = _GUARD.get()
    if guard is None:
        return
    if guard.stopped:
        _stop(guard, guard.stopped)
    verb = str(method or "GET").upper()
    if verb not in _READ_METHODS:
        _stop(guard, f"{verb} {url}")
    host = (urlsplit(url).hostname or "").lower()
    if guard.allowed_hosts is not None and not _host_allowed(host, guard.allowed_hosts):
        _stop(guard, f"host not allowed: {verb} {url}")
    guard.requests.append(f"{verb} {url}")


def _audit(event: str, args: tuple) -> None:
    guard = _GUARD.get()
    if guard is None:
        return
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        if isinstance(path, int):
            return
        writing = any(c in mode for c in "wax+") if isinstance(mode, str) else bool((flags or 0) & _WRITE_FLAGS)
        name = str(path)
        if writing and not name.endswith(".pyc") and "__pycache__" not in name:
            _stop(guard, f"file write: {name}")
    elif event == "socket.connect":
        address = args[1] if len(args) > 1 else None
        if isinstance(address, tuple) and len(address) >= 2:
            if address[1] not in guard.ports:
                _stop(guard, f"connection to {address[0]}:{address[1]} (not an HTTP port)")
        else:
            _stop(guard, f"connection to {address!r}")
    elif event in _STOP_EVENTS or event.startswith(_STOP_PREFIXES):
        _stop(guard, f"side effect: {event}")


# ── 가로채기 설치 ─────────────────────────────────────────────────────────

_installed: set[str] = set()
_install_lock = threading.Lock()


def _install() -> None:
    """한 번만 설치한다. 나중에 import된 httpx·aiohttp는 다음 실행 때 설치한다."""
    with _install_lock:
        if "core" not in _installed:
            sys.addaudithook(_audit)

            original_putrequest = http.client.HTTPConnection.putrequest

            def putrequest(self, method, url, *args, **kwargs):
                if _GUARD.get() is not None:
                    full = url if "://" in url else (
                        f"{'https' if isinstance(self, http.client.HTTPSConnection) else 'http'}://{self.host}:{self.port}{url}")
                    _check_request(method, full)
                return original_putrequest(self, method, url, *args, **kwargs)

            http.client.HTTPConnection.putrequest = putrequest

            original_start = threading.Thread.start

            def start(self, *args, **kwargs):
                if _GUARD.get() is not None:
                    context, run = contextvars.copy_context(), self.run

                    def guarded_run():
                        try:
                            context.run(run)
                        except LiveReadStopped:
                            pass  # 멈춘 사실은 guard에 남아 있다. 스레드 예외 로그만 막는다.

                    self.run = guarded_run
                return original_start(self, *args, **kwargs)

            threading.Thread.start = start

            original_submit = concurrent.futures.ThreadPoolExecutor.submit

            def submit(self, fn, /, *args, **kwargs):
                if _GUARD.get() is not None:
                    fn = functools.partial(contextvars.copy_context().run, fn)
                return original_submit(self, fn, *args, **kwargs)

            concurrent.futures.ThreadPoolExecutor.submit = submit
            _installed.add("core")

        httpx = sys.modules.get("httpx")
        if httpx is not None and "httpx" not in _installed:
            sync_send, async_send = httpx.Client.send, httpx.AsyncClient.send

            def send(self, request, *args, **kwargs):
                _check_request(request.method, str(request.url))
                return sync_send(self, request, *args, **kwargs)

            async def asend(self, request, *args, **kwargs):
                _check_request(request.method, str(request.url))
                return await async_send(self, request, *args, **kwargs)

            httpx.Client.send, httpx.AsyncClient.send = send, asend
            _installed.add("httpx")

        aiohttp = sys.modules.get("aiohttp")
        if aiohttp is not None and "aiohttp" not in _installed:
            original_request = aiohttp.ClientSession._request

            async def _request(self, method, str_or_url, *args, **kwargs):
                _check_request(method, str(str_or_url))
                return await original_request(self, method, str_or_url, *args, **kwargs)

            aiohttp.ClientSession._request = _request
            _installed.add("aiohttp")


# ── 실행기 ────────────────────────────────────────────────────────────────

def _format(value: Any) -> str:
    content = getattr(value, "content", None)  # LangChain ToolMessage
    if isinstance(content, str):
        value = content
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    return text[:_MAX_RESULT]


async def _as_coroutine(awaitable):
    return await awaitable


class LiveReadExecutor:
    """실제 도구(simguard.RealTool)를 GET 전용 모드로 실행한다."""

    def __init__(self, tool: Any, allowed_hosts: Any = None, ports: Any = (80, 443), timeout: float = 30.0) -> None:
        """
        tool:          실행할 실제 도구 (simguard.find_tool이 찾은 RealTool)
        allowed_hosts: 요청을 보내도 되는 도메인. 하위 도메인도 포함한다. None이면 모든 도메인.
        ports:         연결해도 되는 포트 (HTTP·HTTPS)
        timeout:       이 시간 안에 끝나지 않으면 멈추고 예상으로 넘어간다 (초)
        """
        self._tool = tool
        self._allowed_hosts = frozenset(h.lower() for h in allowed_hosts) if allowed_hosts is not None else None
        self._ports = frozenset(ports)
        self._timeout = timeout

    def _run(self, guard: _Guard, arguments: dict) -> Any:
        _install()
        token = _GUARD.set(guard)
        try:
            value = self._tool._invoke(arguments)
            if not inspect.isawaitable(value):
                return value
            # async 도구: 이 스레드의 새 이벤트 루프에서 실행한다. 루프 자체를 만드는 것은 검사하지 않는다
            # (Windows는 루프를 만들 때 내부 소켓을 연다).
            loop = contextvars.Context().run(asyncio.new_event_loop)
            try:
                task = loop.create_task(_as_coroutine(value), context=contextvars.copy_context())
                return loop.run_until_complete(task)
            finally:
                contextvars.Context().run(loop.close)
        finally:
            _GUARD.reset(token)

    async def shadow_execute(self, tool_name: str, tool_arguments: dict) -> ShadowExecutionResult | None:
        guard = _Guard(self._allowed_hosts, self._ports)
        error = None
        try:
            value = await asyncio.wait_for(asyncio.to_thread(self._run, guard, dict(tool_arguments or {})), self._timeout)
        except asyncio.TimeoutError:
            guard.stopped = guard.stopped or f"timeout after {self._timeout}s"
            return None
        except LiveReadStopped:
            return None
        except NotImplementedError:
            return None  # 예: async 전용 LangChain 도구를 동기로 부른 경우. 실제 도구 결과가 아니다.
        except Exception as exc:
            value, error = "", f"{type(exc).__name__}: {exc}"
        if guard.stopped:
            # 도구가 멈춤 예외를 잡아 삼켰어도, 허용되지 않은 동작을 시도했으므로 결과를 쓰지 않는다.
            return None
        return ShadowExecutionResult(
            result=_format(value),
            error=error,
            state_diff=json.dumps({"requests": guard.requests, "note": "GET-only live run: only the requests listed were sent"},
                                  ensure_ascii=False),
            metadata={"shadow_mode": "live_read", "requests": list(guard.requests)},
        )


def configure_live_reads(enabled: bool = True, allowed_hosts: Any = None, ports: Any = (80, 443),
                         timeout: float = 30.0) -> None:
    _config.update(enabled=enabled, allowed_hosts=allowed_hosts, ports=frozenset(ports), timeout=timeout)


def live_read_executor_for(tool_name: str, arguments: Any) -> LiveReadExecutor | None:
    """이 프로세스에 실제 도구가 있으면 GET 전용 실행기를 돌려준다. SQL 호출과 MCP 도구는 제외한다."""
    if not _config["enabled"]:
        return None
    from simguard.shadow_db import is_sql_call
    from simguard.tool_discovery import find_tool

    if is_sql_call(arguments):
        return None
    tool = find_tool(tool_name)
    if tool is None or tool.framework == "mcp":
        return None
    return LiveReadExecutor(tool, _config["allowed_hosts"], _config["ports"], _config["timeout"])
