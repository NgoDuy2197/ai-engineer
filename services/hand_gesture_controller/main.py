"""
Service: Hand Gesture Controller (dieu khien anh bang cu chi tay).

Cu chi:
  - Quet trai        -> anh truoc
  - Quet phai        -> anh sau
  - Gio 5 ngon (giu ~0.8s) -> vao che do XOAY anh
  - Trong che do XOAY: nghieng ban tay de xoay anh
  - Nam tay (giu ~0.8s)    -> thoat che do XOAY
  - Q hoac ESC -> thoat

Chay tren CPU (MediaPipe model_complexity=0) -> nhe cho Intel UHD 620 / Iris Xe.
Copy anh vao thu muc images/ ben canh file nay.
"""
import argparse
import math
import os
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_images(folder):
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(IMG_EXT)]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def count_fingers(pts):
    """Dem so ngon dang duoi (khong phu thuoc huong tay)."""
    wrist = pts[0]
    fingers = 0
    # 4 ngon: tip xa co tay hon pip -> ngon duoi
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if _dist(pts[tip], wrist) > _dist(pts[pip], wrist) * 1.05:
            fingers += 1
    # ngon cai: so voi goc long ban tay phia ngon ut (17)
    if _dist(pts[4], pts[17]) > _dist(pts[2], pts[17]) * 1.05:
        fingers += 1
    return fingers


def hand_angle(pts):
    """Goc (do) cua vector co tay(0) -> middle_mcp(9)."""
    dx = pts[9][0] - pts[0][0]
    dy = pts[9][1] - pts[0][1]
    return math.degrees(math.atan2(dy, dx))


def rotate_image(img, angle):
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - cx
    M[1, 2] += nh / 2 - cy
    return cv2.warpAffine(img, M, (nw, nh), borderValue=(25, 25, 25))


def fit(img, box_w, box_h):
    h, w = img.shape[:2]
    s = min(box_w / w, box_h / h)
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))


def main():
    ap = argparse.ArgumentParser(description="Dieu khien anh bang cu chi tay")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--images", default=os.path.join(here, "images"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--height", type=int, default=700)
    args = ap.parse_args()

    images = load_images(args.images)
    idx = 0
    cur = cv2.imread(images[idx]) if images else None

    use_dshow = os.name == "nt"
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if use_dshow else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("[!] Khong mo duoc camera")
        return

    hands = mp.solutions.hands.Hands(
        model_complexity=0, max_num_hands=1,
        min_detection_confidence=0.6, min_tracking_confidence=0.5)
    draw = mp.solutions.drawing_utils

    xs = deque(maxlen=12)            # lich su (thoi_gian, x) de phat hien quet
    last_swipe = 0.0
    mode = "NORMAL"                  # NORMAL | ROTATE
    open_since = None                # moc thoi gian bat dau gio 5 ngon
    fist_since = None                # moc thoi gian bat dau nam tay
    rotate = 0.0
    base_angle = None

    W, H = args.width, args.height
    win = "Hand Gesture Controller"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)   # guong nhu selfie -> di chuyen tu nhien
        fh, fw = frame.shape[:2]
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        now = time.time()
        gesture = "-"

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            pts = [(p.x * fw, p.y * fh) for p in lm.landmark]
            nf = count_fingers(pts)
            gesture = f"{nf} ngon"
            draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)

            if mode == "NORMAL":
                px = pts[9][0] / fw            # x long ban tay (chuan hoa 0..1)
                xs.append((now, px))
                # phat hien quet: dich chuyen ngang nhanh
                if now - last_swipe > 0.8 and len(xs) >= 5:
                    t0, x0 = xs[0]
                    t1, x1 = xs[-1]
                    if t1 - t0 < 0.5 and abs(x1 - x0) > 0.28 and images:
                        if x1 - x0 > 0:
                            idx = (idx + 1) % len(images)
                            gesture = "QUET PHAI ->"
                        else:
                            idx = (idx - 1) % len(images)
                            gesture = "<- QUET TRAI"
                        cur = cv2.imread(images[idx])
                        rotate = 0.0
                        last_swipe = now
                        xs.clear()
                # gio 5 ngon giu ~0.8s -> vao che do xoay
                if nf >= 5:
                    open_since = open_since or now
                    if now - open_since > 0.8:
                        mode = "ROTATE"
                        base_angle = None
                        open_since = None
                else:
                    open_since = None
                fist_since = None
            else:  # ROTATE
                a = hand_angle(pts)
                if base_angle is None:
                    base_angle = a - rotate   # giu anh khong nhay khi vao mode
                rotate = a - base_angle
                # nam tay giu ~0.8s -> thoat
                if nf <= 1:
                    fist_since = fist_since or now
                    if now - fist_since > 0.8:
                        mode = "NORMAL"
                        fist_since = None
                else:
                    fist_since = None
        else:
            xs.clear()
            open_since = fist_since = None

        # ==== ve canvas hien thi ====
        canvas = np.full((H, W, 3), 25, np.uint8)
        top, region_h = 40, (H - 45) - 40
        if cur is not None:
            disp = rotate_image(cur, rotate) if abs(rotate) > 0.5 else cur
            disp = fit(disp, W - 40, region_h)
            y = top + (region_h - disp.shape[0]) // 2
            x = (W - disp.shape[1]) // 2
            canvas[y:y + disp.shape[0], x:x + disp.shape[1]] = disp
        else:
            cv2.putText(canvas, "Copy anh (.jpg/.png) vao thu muc images/",
                        (40, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        # camera thumbnail goc phai tren
        thumb = cv2.resize(frame, (200, 150))
        canvas[10:160, W - 210:W - 10] = thumb
        cv2.rectangle(canvas, (W - 210, 10), (W - 10, 160), (80, 80, 80), 1)

        # thanh trang thai
        color = (60, 200, 90) if mode == "NORMAL" else (60, 160, 255)
        cv2.rectangle(canvas, (0, H - 40), (W, H), (15, 15, 15), -1)
        info = f"Che do: {mode}   |   Cu chi: {gesture}"
        if images:
            info += f"   |   Anh {idx + 1}/{len(images)}"
        if mode == "ROTATE":
            info += f"   |   Goc: {rotate:6.1f}"
        cv2.putText(canvas, info, (15, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(canvas,
                    "Quet trai/phai: doi anh | 5 ngon: xoay | Nam tay: thoat xoay | Q/ESC: thoat",
                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow(win, canvas)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
