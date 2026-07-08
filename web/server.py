"""
Dashboard server (chi dung thu vien chuan cua Python - khong can pip install).
- Phuc vu trang dieu huong web/index.html
- API POST /api/launch?service=<ten> -> chay file .bat tuong ung trong __bat/
"""
import http.server
import json
import os
import socketserver
import subprocess
import threading
import urllib.parse
import webbrowser

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(WEB_DIR)
BAT_DIR = os.path.join(REPO_DIR, "__bat")
STATS_PATH = os.path.join(REPO_DIR, "services", "people_counter", "data", "stats.json")
PORT = 8000

# Map ten service -> file .bat khoi chay
SERVICES = {
    "object_detection": "start_object_detection.bat",
    "hand_gesture": "start_hand_gesture.bat",
    "people_counter": "start_people_counter.bat",
}

DEFAULT_STATS = {
    "updated": 0, "current": 0, "total_seen": 0, "in_count": 0,
    "out_count": 0, "inside": 0, "conversion": 0.0,
    "per_minute": {"labels": [], "in": [], "out": []},
}


def launch_bat(path):
    """Mo file .bat trong cua so rieng (uu tien Windows)."""
    if os.name == "nt":
        os.startfile(path)  # noqa: mo nhu double-click
    else:
        # Case default chung (Linux/macOS) - chay bang bash neu co
        subprocess.Popen(["bash", path], cwd=REPO_DIR)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def log_message(self, fmt, *args):
        pass  # yen lang

    def _json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/stats":
            data = DEFAULT_STATS
            try:
                with open(STATS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:  # noqa: file chua ton tai / dang ghi
                pass
            self._json(200, data)
            return
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/launch":
            self.send_error(404)
            return
        qs = urllib.parse.parse_qs(parsed.query)
        service = (qs.get("service") or [""])[0]
        bat = SERVICES.get(service)
        if not bat:
            self._json(400, {"ok": False, "message": f"Service khong hop le: {service}"})
            return
        path = os.path.join(BAT_DIR, bat)
        if not os.path.isfile(path):
            self._json(400, {"ok": False, "message": f"Khong tim thay: {path}"})
            return
        try:
            launch_bat(path)
            self._json(200, {"ok": True, "message": f"Da khoi chay: {service}"})
        except Exception as e:  # noqa
            self._json(500, {"ok": False, "message": str(e)})


def main():
    socketserver.TCPServer.allow_reuse_address = True
    url = f"http://localhost:{PORT}"
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"[i] Dashboard chay tai {url}")
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[i] Da dung server.")


if __name__ == "__main__":
    main()
