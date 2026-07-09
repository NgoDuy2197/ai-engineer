"""
Service: People Counter (dem nguoi + dem nguoi vao quan qua vach).

- Theo doi camera, moi nguoi MOI xuat hien -> total_seen +1
- Ve 1 vach (click 2 diem chuot). Khi 1 nguoi di qua vach -> +1 vao/ra
- Ghi thong ke ra data/stats.json de Dashboard web doc va phan tich theo phut.

Toi uu Intel UHD 620 / Iris Xe qua OpenVINO; fallback CPU chung (case default).

Phim tat:
  Chuot trai (2 lan) : ve vach (2 diem)
  R : xoa vach de ve lai
  F : dao chieu "vao/ra"
  C : reset bo dem
  Q / ESC : thoat
"""
import argparse
import json
import os
import threading
import time
from collections import defaultdict

import cv2

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STATS_PATH = os.path.join(DATA_DIR, "stats.json")

# Diem phat hien qua vach duoc nhac len 1 chut so voi day khung (theo % chieu cao box).
FEET_OFFSET = 0.15

try:
    import winsound  # co san tren Windows
    _HAS_SOUND = os.name == "nt"
except ImportError:
    _HAS_SOUND = False


def _beep_seq(tones):
    for freq, dur in tones:
        winsound.Beep(freq, dur)


def play_event_sound(going_in):
    """Phat am thanh khac nhau: VAO = 2 not di len, RA = 2 not di xuong."""
    if not _HAS_SOUND:
        return
    tones = [(880, 120), (1320, 150)] if going_in else [(660, 120), (440, 170)]
    threading.Thread(target=_beep_seq, args=(tones,), daemon=True).start()


def load_model(weights, device, imgsz, reexport):
    from ultralytics import YOLO

    if device in ("auto", "intel:cpu", "intel:gpu"):
        try:
            import openvino  # noqa: F401
            ov_dir = os.path.splitext(weights)[0] + "_openvino_model"
            if reexport and os.path.isdir(ov_dir):
                import shutil
                shutil.rmtree(ov_dir)
            if not os.path.isdir(ov_dir):
                print("[i] Xuat model sang OpenVINO (chi lam 1 lan)...")
                YOLO(weights).export(format="openvino", imgsz=imgsz)
            model = YOLO(ov_dir, task="detect")
            ov_device = "intel:gpu" if device == "intel:gpu" else "intel:cpu"
            print(f"[i] Backend: OpenVINO ({ov_device})")
            return model, ov_device
        except Exception as e:  # noqa
            print(f"[!] OpenVINO khong dung duoc: {e} -> Fallback CPU thuong.")
    print("[i] Backend: PyTorch CPU")
    return YOLO(weights), "cpu"


def side(a, b, p):
    """Dau cua tich cheo (b-a) x (p-a): xac dinh diem p o phia nao cua vach a->b."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def seg_cross(p1, p2, a, b):
    """(p1->p2) co cat doan vach (a->b) khong."""
    d1, d2 = side(a, b, p1), side(a, b, p2)
    d3, d4 = side(p1, p2, a), side(p1, p2, b)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def main():
    ap = argparse.ArgumentParser(description="Dem nguoi + dem qua vach")
    ap.add_argument("--source", default="0", help="0=webcam hoac duong dan video")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "intel:cpu", "intel:gpu", "cpu"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--reexport", action="store_true")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    model, dev = load_model(args.weights, args.device, args.imgsz, args.reexport)

    src = int(args.source) if args.source.isdigit() else args.source
    use_dshow = os.name == "nt" and isinstance(src, int)
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if use_dshow else 0)
    if not cap.isOpened():
        print(f"[!] Khong mo duoc nguon: {args.source}")
        return

    win = "People Counter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    state = {"line": [], "flip": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(state["line"]) >= 2:
                state["line"] = []
            state["line"].append((x, y))

    cv2.setMouseCallback(win, on_mouse)

    seen_ids = set()
    prev_center = {}
    in_count = 0
    out_count = 0
    per_min_in = defaultdict(int)
    per_min_out = defaultdict(int)
    last_write = 0.0
    fps = 0.0
    prev_t = time.time()
    line_init = False

    def write_stats(current, now):
        cur_min = int(now // 60)
        mins = list(range(cur_min - 14, cur_min + 1))
        labels = [time.strftime("%H:%M", time.localtime(m * 60)) for m in mins]
        total_seen = len(seen_ids)
        stats = {
            "updated": now,
            "current": current,
            "total_seen": total_seen,
            "in_count": in_count,
            "out_count": out_count,
            "inside": in_count - out_count,
            "conversion": round(in_count / total_seen, 3) if total_seen else 0.0,
            "per_minute": {
                "labels": labels,
                "in": [per_min_in.get(m, 0) for m in mins],
                "out": [per_min_out.get(m, 0) for m in mins],
            },
        }
        tmp = STATS_PATH + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False)
            # Windows co the bao WinError 5 khi file dang bi tien trinh khac (dashboard,
            # antivirus) mo doc -> thu lai vai lan, roi ghi truc tiep neu van khong duoc.
            for attempt in range(5):
                try:
                    os.replace(tmp, STATS_PATH)
                    return
                except PermissionError:
                    time.sleep(0.05)
            with open(STATS_PATH, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False)
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError as e:
            print(f"[!] Khong ghi duoc stats.json (bo qua): {e}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if not line_init and not state["line"]:
            state["line"] = [(w // 2, 0), (w // 2, h)]  # vach doc mac dinh
        line_init = True

        results = model.track(frame, persist=True, classes=[0], conf=args.conf,
                              imgsz=args.imgsz, device=dev, verbose=False,
                              tracker="bytetrack.yaml")
        r = results[0]
        current = 0
        now = time.time()

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            xyxy = r.boxes.xyxy.cpu().numpy()
            current = len(ids)
            for tid, box in zip(ids, xyxy):
                x1, y1, x2, y2 = box
                foot_y = y2 - FEET_OFFSET * (y2 - y1)   # nhac len 1 chut so voi day khung
                center = ((x1 + x2) / 2.0, float(foot_y))
                if tid not in seen_ids:
                    seen_ids.add(tid)
                # ve box + id
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
                cv2.putText(frame, f"#{tid}", (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
                cv2.circle(frame, (int(center[0]), int(center[1])), 4, (0, 0, 255), -1)

                # kiem tra qua vach
                if tid in prev_center and len(state["line"]) == 2:
                    a, b = state["line"]
                    p0 = prev_center[tid]
                    if seg_cross(p0, center, a, b):
                        going_in = side(a, b, p0) < 0 and side(a, b, center) > 0
                        if state["flip"]:
                            going_in = not going_in
                        cur_min = int(now // 60)
                        if going_in:
                            in_count += 1
                            per_min_in[cur_min] += 1
                        else:
                            out_count += 1
                            per_min_out[cur_min] += 1
                        play_event_sound(going_in)
                prev_center[tid] = center

        # ==== overlay ====
        if len(state["line"]) == 2:
            cv2.line(frame, state["line"][0], state["line"][1], (0, 165, 255), 3)
        elif len(state["line"]) == 1:
            cv2.circle(frame, state["line"][0], 5, (0, 165, 255), -1)

        now2 = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now2 - prev_t, 1e-6))
        prev_t = now2

        panel = [
            f"Dang trong khung: {current}",
            f"Tong nguoi thay: {len(seen_ids)}",
            f"VAO quan: {in_count}   RA: {out_count}   Trong quan: {in_count - out_count}",
            f"FPS: {fps:4.1f} [{dev}]  |  R:ve lai vach  F:dao chieu  C:reset  Q:thoat",
        ]
        y = 26
        cv2.rectangle(frame, (0, 0), (w, 118), (0, 0, 0), -1)
        for i, txt in enumerate(panel):
            col = (0, 255, 180) if i < 3 else (180, 180, 180)
            cv2.putText(frame, txt, (12, y + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        cv2.imshow(win, frame)

        if now - last_write > 1.0:
            write_stats(current, now)
            last_write = now

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        elif k == ord("r"):
            state["line"] = []
            line_init = False
        elif k == ord("f"):
            state["flip"] = not state["flip"]
        elif k == ord("c"):
            seen_ids.clear()
            prev_center.clear()
            in_count = out_count = 0
            per_min_in.clear()
            per_min_out.clear()

    write_stats(0, time.time())
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
