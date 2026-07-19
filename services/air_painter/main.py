"""
Service: Air Painter (ve trong khong khi bang ngon tay).

Cu chi (tay phai/trai deu duoc, camera da lat guong nhu selfie):
  - Gio 1 ngon TRO           -> VE theo dau ngon tro
  - Gio 2 ngon (tro + giua)  -> CHON mau/cong cu tren thanh mau (khong ve)
  - Nam tay / khong gio ngon -> nghi (nhac but len)

Thanh mau tren cung: cham 2 ngon vao o mau de doi mau; o "TAY" = cuc tay (xoa).

Phim tat:
  C : xoa toan bo tranh
  S : luu tranh vao data/
  + / - : tang / giam co but
  Q / ESC : thoat

Chay tren CPU (MediaPipe model_complexity=0) -> nhe cho Intel UHD 620 / Iris Xe.
"""
import argparse
import math
import os
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_PATH = os.path.join(HERE, "models", "hand_landmarker.task")

# (ten, mau BGR). O cuoi la cuc tay (eraser).
PALETTE = [
    ("XANH", (0, 200, 0)),
    ("DUONG", (255, 90, 0)),
    ("DO", (0, 0, 255)),
    ("VANG", (0, 220, 255)),
    ("TIM", (255, 0, 200)),
    ("TRANG", (255, 255, 255)),
    ("TAY", (30, 30, 30)),
]


def finger_up(pts, tip, pip):
    """Ngon duoi neu tip xa co tay hon pip (khong phu thuoc huong tay)."""
    w = pts[0]
    d = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])  # noqa: E731
    return d(pts[tip], w) > d(pts[pip], w) * 1.05


def main():
    ap = argparse.ArgumentParser(description="Ve trong khong khi bang ngon tay")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--brush", type=int, default=8, help="Co net ve ban dau")
    ap.add_argument("--max-hands", type=int, default=2,
                    help="So ban tay toi da ve cung luc")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    use_dshow = os.name == "nt"
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if use_dshow else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("[!] Khong mo duoc camera")
        return

    if not os.path.isfile(MODEL_PATH):
        print(f"[!] Khong tim thay model: {MODEL_PATH}")
        print("    Tai tai: https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        return
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=max(1, args.max_hands),
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5)
    landmarker = mp_vision.HandLandmarker.create_from_options(options)
    t_start = time.time()
    last_ts = -1

    canvas = None
    prev_pts = {}              # diem ve truoc do theo tung tay {chi_so_tay: (x, y)}
    color = PALETTE[0][1]
    color_name = PALETTE[0][0]
    brush = max(2, args.brush)
    bar_h = 60                 # chieu cao thanh mau

    win = "Air Painter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)      # guong selfie
        h, w = frame.shape[:2]
        if canvas is None:
            canvas = np.zeros((h, w, 3), np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.time() - t_start) * 1000)   # phai tang dan cho mode VIDEO
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts
        res = landmarker.detect_for_video(mp_img, ts)
        mode = "NGHI"

        # prev cho frame nay: chi giu lai cac chi so tay dang co
        cur_prev = {}
        for hi, lm in enumerate(res.hand_landmarks):
            pts = [(p.x * w, p.y * h) for p in lm]
            index_up = finger_up(pts, 8, 6)
            middle_up = finger_up(pts, 12, 10)
            ix, iy = int(pts[8][0]), int(pts[8][1])   # dau ngon tro

            if index_up and middle_up:
                # ==== che do CHON (dung chung mau) ====
                mode = "CHON"
                cv2.circle(frame, (ix, iy), 14, (200, 200, 200), 2)
                if iy < bar_h:
                    slot = int(ix / (w / len(PALETTE)))
                    slot = max(0, min(len(PALETTE) - 1, slot))
                    color_name, color = PALETTE[slot][0], PALETTE[slot][1]
            elif index_up:
                # ==== che do VE (moi tay ve net rieng) ====
                if mode != "CHON":
                    mode = "VE"
                is_eraser = color_name == "TAY"
                size = brush * 5 if is_eraser else brush
                draw_col = (0, 0, 0) if is_eraser else color
                cv2.circle(frame, (ix, iy), size, draw_col, 2)
                prev = prev_pts.get(hi)
                if prev is not None:
                    cv2.line(canvas, prev, (ix, iy), draw_col, size * 2)
                cur_prev[hi] = (ix, iy)
            # con lai (nam tay) -> khong luu prev cho tay nay -> nhac but

        prev_pts = cur_prev

        # ==== ghep tranh len khung hinh ====
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        inv = cv2.bitwise_not(mask)
        frame = cv2.bitwise_and(frame, frame, mask=inv)
        frame = cv2.bitwise_or(frame, canvas)

        # ==== thanh mau tren cung ====
        slot_w = w / len(PALETTE)
        for i, (name, col) in enumerate(PALETTE):
            x0 = int(i * slot_w)
            x1 = int((i + 1) * slot_w)
            cv2.rectangle(frame, (x0, 0), (x1, bar_h), col, -1)
            sel = (name == color_name)
            cv2.rectangle(frame, (x0, 0), (x1, bar_h),
                          (255, 255, 255) if sel else (60, 60, 60), 3 if sel else 1)
            txt_col = (0, 0, 0) if name in ("VANG", "TRANG") else (255, 255, 255)
            cv2.putText(frame, name, (x0 + 8, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_col, 2)

        # ==== thanh trang thai ====
        cv2.rectangle(frame, (0, h - 34), (w, h), (15, 15, 15), -1)
        cv2.putText(frame,
                    f"Che do: {mode}  |  Mau: {color_name}  |  Co but: {brush}  "
                    f"|  1 ngon: ve  2 ngon: chon  C:xoa  S:luu  +/-:but  Q:thoat",
                    (12, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        elif k == ord("c"):
            canvas[:] = 0
        elif k in (ord("+"), ord("=")):
            brush = min(40, brush + 2)
        elif k in (ord("-"), ord("_")):
            brush = max(2, brush - 2)
        elif k == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            path = os.path.join(DATA_DIR, f"paint_{ts}.png")
            cv2.imwrite(path, frame)
            print(f"[i] Da luu: {path}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
