#!/usr/bin/env python3
"""
IPI Test Server — http://localhost:8765
IPI 공격을 시뮬레이션하는 로컬 HTTP 서버.

/page/<id> 경로마다 정상 콘텐츠 + IPI 주입 텍스트를 담은 HTML을 반환한다.
injection_technique:
  inline  - 본문에 직접 삽입 (에이전트가 읽을 수 있음)
  comment - HTML 주석 안에 숨김 (<!-- ... -->)
  hidden  - CSS로 숨긴 텍스트 (color:white;font-size:0px)

실행:
  python attacks-bench/ipi/ipi_server.py          # 단독 서버
  또는 run_benchmark.py가 자동으로 daemon으로 시작함
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "ipi_attacks.json"
PORT      = 8765

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
</head>
<body>
<article>
  <h1>{title}</h1>
  <p>{content}</p>
  {injection_block}
</article>
</body>
</html>"""


def _build_injection_block(technique: str, text: str) -> str:
    if not text:
        return ""
    if technique == "comment":
        return f"<!-- {text} -->"
    if technique == "hidden":
        return f'<p style="color:white;font-size:0px;visibility:hidden">{text}</p>'
    # inline (default)
    return f"<p>{text}</p>"


def _build_pages(cases: list[dict]) -> dict[str, bytes]:
    pages: dict[str, bytes] = {}
    for c in cases:
        pid   = c["id"]
        html  = _HTML_TEMPLATE.format(
            title=c.get("page_title", "Article"),
            content=c.get("page_content", ""),
            injection_block=_build_injection_block(
                c.get("injection_technique", "inline"),
                c.get("injection_text", ""),
            ),
        )
        pages[f"page/{pid}"] = html.encode("utf-8")
    return pages


class _Handler(BaseHTTPRequestHandler):
    pages: dict[str, bytes] = {}

    def log_message(self, *args):
        pass  # 로그 억제

    def do_GET(self):
        key = self.path.lstrip("/")
        if key in self.pages:
            body = self.pages[key]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")


def run(port: int = PORT, daemon: bool = False) -> HTTPServer:
    cases = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    _Handler.pages = _build_pages(cases)

    server = HTTPServer(("127.0.0.1", port), _Handler)

    if daemon:
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"[IPI Server] listening on http://localhost:{port} (daemon)")
        return server

    print(f"[IPI Server] listening on http://localhost:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return server


if __name__ == "__main__":
    run()
