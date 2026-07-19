"""
Dashboard server (chi dung thu vien chuan cua Python - khong can pip install).
- Phuc vu trang dieu huong web/index.html
- API POST /api/launch?service=<ten> -> chay file .bat tuong ung trong __bat/
"""
import http.server
import json
import os
import socket
import socketserver
import subprocess
import sys
import threading
import urllib.parse
import webbrowser

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(WEB_DIR)
BAT_DIR = os.path.join(REPO_DIR, "__bat")
STATS_PATH = os.path.join(REPO_DIR, "services", "people_counter", "data", "stats.json")
PORT = 8000

# Cac trang playground chay trong trinh duyet (nam o thu muc khac, tu chua = CDN)
PLAYGROUND_DIR = os.path.join(REPO_DIR, "__MORE", "20260714")
PLAYGROUND_FILES = {"camera-ai-playground.html", "camera-ai-mediapipe.html"}

# Map ten service -> file .bat khoi chay
SERVICES = {
    "object_detection": "start_object_detection.bat",
    "hand_gesture": "start_hand_gesture.bat",
    "people_counter": "start_people_counter.bat",
    "movement_heatmap": "start_movement_heatmap.bat",
    "air_painter": "start_air_painter.bat",
    "fitness_coach": "start_fitness_coach.bat",
    "security_cam": "start_security_cam.bat",
    "fruit_catch": "start_fruit_catch.bat",
    "ninja_slash": "start_ninja_slash.bat",
    "space_dodge": "start_space_dodge.bat",
    "pose_echo": "start_pose_echo.bat",
    "face_emoji": "start_face_emoji.bat",
    "angel_wings": "start_angel_wings.bat",
    # Face Recognition co nhieu buoc -> moi buoc 1 key
    "face_setup": "setup_face_recognition.bat",
    "face_enroll": "start_face_enroll.bat",
    "face_label": "start_face_label.bat",
    "face_train": "start_face_train.bat",
    "face_camera": "start_face_camera.bat",
}

# Map ten -> thu muc mo nhanh trong Explorer (nut "Mo thu muc")
FOLDERS = {
    "face_data": os.path.join(REPO_DIR, "services", "face_recognition", "__data"),
    "wings_data": os.path.join(REPO_DIR, "services", "angel_wings", "__data", "wings"),
}

DEFAULT_STATS = {
    "updated": 0, "current": 0, "total_seen": 0, "in_count": 0,
    "out_count": 0, "inside": 0, "conversion": 0.0,
    "per_minute": {"labels": [], "in": [], "out": []},
}


def open_folder(path):
    """Mo thu muc bang trinh quan ly file cua he dieu hanh."""
    os.makedirs(path, exist_ok=True)
    if os.name == "nt":
        os.startfile(path)  # noqa
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


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

    def end_headers(self):
        # Chong cache: luon tai ban index.html moi nhat (tranh xem trang cu)
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/stats":
            data = DEFAULT_STATS
            try:
                with open(STATS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:  # noqa: file chua ton tai / dang ghi
                pass
            self._json(200, data)
            return
        # Phuc vu trang playground tu thu muc __MORE (khong nam trong web/)
        name = os.path.basename(path)
        if name in PLAYGROUND_FILES:
            self._serve_playground(name)
            return
        super().do_GET()

    def _serve_playground(self, name):
        fpath = os.path.join(PLAYGROUND_DIR, name)
        try:
            with open(fpath, "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(404, "File not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)

        # Mo nhanh thu muc trong Explorer
        if parsed.path == "/api/open":
            key = (qs.get("folder") or [""])[0]
            folder = FOLDERS.get(key)
            if not folder:
                self._json(400, {"ok": False, "message": f"Thu muc khong hop le: {key}"})
                return
            try:
                open_folder(folder)
                self._json(200, {"ok": True, "message": f"Da mo thu muc: {folder}"})
            except Exception as e:  # noqa
                self._json(500, {"ok": False, "message": str(e)})
            return

        if parsed.path != "/api/launch":
            self.send_error(404)
            return
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


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Server da luong: xu ly nhieu ket noi/keep-alive cung luc.

    TCPServer don luong se bi ket khi trinh duyet giu ket noi keep-alive
    -> trang khong tai duoc. ThreadingMixIn khac phuc dieu do.
    """
    allow_reuse_address = True
    daemon_threads = True


def _port_in_use(port):
    """True neu da co server khac dang lang nghe o cong nay."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    url = f"http://localhost:{PORT}"
    if _port_in_use(PORT):
        print("=" * 60)
        print(f"[!] Cong {PORT} DANG BI CHIEM boi mot server khac.")
        print("    Ban co the dang mo __MORE\\20260714\\start.bat hoac 1 dashboard cu.")
        print("    -> DONG cua so server do lai, roi chay lai run_dashboard.bat.")
        print(f"    (Neu trinh duyet van hien trang cu: bam Ctrl+Shift+R de bo cache.)")
        print("=" * 60)
        return
    with ThreadingServer(("", PORT), Handler) as httpd:
        print(f"[i] Dashboard chay tai {url}")
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[i] Da dung server.")


if __name__ == "__main__":
    main()
