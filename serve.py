"""零依赖静态服务器：托管 web/ 页面并提供实验 JSON。

用法：python3 serve.py [--port 8000]
数据文件较大（3000 步 × 64 卡），启用 gzip 传输。
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")
OUT = os.path.join(ROOT, "output")

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s\n" % (fmt % args))

    def _send(self, path: str, ctype: str):
        if not os.path.exists(path):
            self.send_error(404)
            return
        with open(path, "rb") as f:
            raw = f.read()
        # JSON 大文件 gzip 后体积可降一个数量级
        if path.endswith(".json") and "gzip" in self.headers.get("Accept-Encoding", ""):
            body = gzip.compress(raw, compresslevel=6)
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        url = self.path.split("?")[0]
        if url in ("/", "/index.html"):
            return self._send(os.path.join(WEB, "index.html"), MIME[".html"])
        if url.startswith("/web/"):
            name = url[5:]
            ext = os.path.splitext(name)[1]
            return self._send(os.path.join(WEB, name), MIME.get(ext, "application/octet-stream"))
        if url == "/data/experiment.json":
            return self._send(os.path.join(OUT, "experiment.json"), MIME[".json"])
        if url == "/data/report.json":
            return self._send(os.path.join(OUT, "experiment_report.json"), MIME[".json"])
        self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if not os.path.exists(os.path.join(OUT, "experiment.json")):
        print("未发现 output/experiment.json，请先运行 python3 run_experiment.py")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"慢节点监控页面：http://127.0.0.1:{args.port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
