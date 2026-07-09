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

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

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
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    use_dshow = os.name == "nt"
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if use_dshow else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("[!] Khong mo duoc camera")
        return

    hands = mp.solutions.hands.Hands(
        model_complexity=0, max_num_hands=1,
        min_detection_confidence=0.6, min_tracking_confidence=0.5)

    canvas = None
    prev = None                # diem ve truoc do
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

        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mode = "NGHI"

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            pts = [(p.x * w, p.y * h) for p in lm.landmark]
            index_up = finger_up(pts, 8, 6)
            middle_up = finger_up(pts, 12, 10)
            ix, iy = int(pts[8][0]), int(pts[8][1])   # dau ngon tro

            if index_up and middle_up:
                # ==== che do CHON ====
                mode = "CHON"
                prev = None
                cv2.circle(frame, (ix, iy), 14, (200, 200, 200), 2)
                if iy < bar_h:
                    slot = int(ix / (w / len(PALETTE)))
                    slot = max(0, min(len(PALETTE) - 1, slot))
                    color_name, color = PALETTE[slot][0], PALETTE[slot][1]
            elif index_up:
                # ==== che do VE ====
                mode = "VE"
                is_eraser = color_name == "TAY"
                size = brush * 5 if is_eraser else brush
                draw_col = (0, 0, 0) if is_eraser else color
                cv2.circle(frame, (ix, iy), size, draw_col, 2)
                if prev is not None:
                    cv2.line(canvas, prev, (ix, iy), draw_col, size * 2)
                prev = (ix, iy)
            else:
                prev = None
        else:
            prev = None

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
    hands.close()


if __name__ == "__main__":
    main()
