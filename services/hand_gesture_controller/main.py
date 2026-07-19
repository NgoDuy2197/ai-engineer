"""
Service: Hand Gesture Controller (dieu khien anh bang cu chi tay).

Cu chi:
  - Gio 1 ngon (ngon tro) roi luot trai/phai -> doi anh (nhay)
  - Gio 2 ngon             -> reset anh ve trang thai ban dau (het xoay)
  - Gio 3 hoac 4 ngon      -> hien so ngon to giua man hinh roi tu bien mat
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
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Ket noi cac diem landmark ban tay (thay cho mp.solutions.hands.HAND_CONNECTIONS,
# vi ban mediapipe cho Python 3.13 chi con Tasks API, khong con "solutions").
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # ngon cai
    (0, 5), (5, 6), (6, 7), (7, 8),          # ngon tro
    (5, 9), (9, 10), (10, 11), (11, 12),     # ngon giua
    (9, 13), (13, 14), (14, 15), (15, 16),   # ngon ap ut
    (13, 17), (17, 18), (18, 19), (19, 20),  # ngon ut
    (0, 17),                                 # long ban tay
]


def draw_hand(frame, pts):
    """Ve khung xuong ban tay (thay cho mp.solutions.drawing_utils)."""
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, (int(pts[a][0]), int(pts[a][1])),
                 (int(pts[b][0]), int(pts[b][1])), (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)


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
    ap.add_argument("--max-hands", type=int, default=2,
                    help="So ban tay toi da nhan dien cung luc")
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

    model_path = os.path.join(here, "models", "hand_landmarker.task")
    if not os.path.isfile(model_path):
        print(f"[!] Khong tim thay model: {model_path}")
        print("    Tai tai: https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        return
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=max(1, args.max_hands),
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5)
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    # --- state RIENG theo tung tay (dict theo chi so tay i) de khong lan cu chi ---
    xs = {}                          # i -> deque (thoi_gian, x) phat hien quet 1 ngon
    last_swipe = {}                  # i -> moc thoi gian quet gan nhat cua tay do
    open_since = {}                  # i -> moc thoi gian bat dau gio 5 ngon
    fist_since = {}                  # i -> moc thoi gian bat dau nam tay
    prev_nf = {}                     # i -> so ngon frame truoc (bat "canh len")

    # --- state CHUNG cho anh dang hien (single anh, single goc xoay) ---
    mode = "NORMAL"                  # NORMAL | ROTATE
    rotate_hand = None               # chi so tay dang "cam quyen" xoay anh
    rotate = 0.0
    base_angle = None
    big_num = 0                      # so ngon dang hien to giua man hinh (3/4)
    big_until = 0.0                  # thoi diem se an so to di

    t_start = time.time()            # moc de tinh timestamp cho detect_for_video
    last_ts = -1

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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.time() - t_start) * 1000)   # phai tang dan cho mode VIDEO
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts
        res = landmarker.detect_for_video(mp_img, ts)
        now = time.time()
        gesture = "-"

        hands = res.hand_landmarks or []
        present = set(range(len(hands)))

        # ---- don dep state cua nhung tay khong con xuat hien ----
        for d in (xs, last_swipe, open_since, fist_since, prev_nf):
            for k in [kk for kk in d if kk not in present]:
                del d[k]

        # ---- tinh truoc landmark + so ngon + ve khung xuong cho MOI tay ----
        hands_info = []              # list (i, pts, nf)
        for i, lm in enumerate(hands):
            pts = [(p.x * fw, p.y * fh) for p in lm]
            nf = count_fingers(pts)
            draw_hand(frame, pts)
            hands_info.append((i, pts, nf))

        if hands_info:
            gesture = " ".join(f"T{i}:{nf}" for i, _, nf in hands_info)

        # ---- neu tay dang cam quyen xoay da bien mat -> chuyen quyen cho tay khac ----
        if mode == "ROTATE" and rotate_hand not in present:
            rotate_hand = min(present) if present else None
            base_angle = None         # gan lai moc de anh khong nhay khi doi tay

        # ---- xu ly cu chi cho TUNG tay (state rieng, khong xung dot) ----
        for i, pts, nf in hands_info:
            if mode == "NORMAL":
                # ---- QUET doi anh bang 1 ngon (theo dau ngon tro, nhay hon) ----
                if nf == 1:
                    dq = xs.setdefault(i, deque(maxlen=12))
                    tipx = pts[8][0] / fw          # x dau ngon tro (chuan hoa 0..1)
                    dq.append((now, tipx))
                    if now - last_swipe.get(i, 0.0) > 0.5 and len(dq) >= 3:
                        t0, x0 = dq[0]
                        t1, x1 = dq[-1]
                        if t1 - t0 < 0.6 and abs(x1 - x0) > 0.15 and images:
                            if x1 - x0 > 0:
                                idx = (idx + 1) % len(images)
                                gesture = "QUET PHAI ->"
                            else:
                                idx = (idx - 1) % len(images)
                                gesture = "<- QUET TRAI"
                            cur = cv2.imread(images[idx])
                            rotate = 0.0
                            last_swipe[i] = now
                            dq.clear()
                elif i in xs:
                    xs[i].clear()

                # ---- 2 ngon: reset anh ve trang thai ban dau (het xoay) ----
                if nf == 2 and prev_nf.get(i) != 2:
                    rotate = 0.0
                    base_angle = None
                    gesture = "RESET ANH"

                # ---- 3 / 4 ngon: hien so to giua man hinh roi tu bien mat ----
                if nf in (3, 4) and prev_nf.get(i) != nf:
                    big_num = nf
                    big_until = now + 1.0

                # ---- gio 5 ngon giu ~0.8s -> vao che do XOAY ----
                if nf >= 5:
                    open_since[i] = open_since.get(i) or now
                    if now - open_since[i] > 0.8:
                        mode = "ROTATE"
                        rotate_hand = i           # tay nay cam quyen xoay
                        base_angle = None
                        open_since.pop(i, None)
                else:
                    open_since.pop(i, None)
                fist_since.pop(i, None)
            else:  # ROTATE - chi tay dang cam quyen moi dieu khien goc xoay
                if i != rotate_hand:
                    continue
                a = hand_angle(pts)
                if base_angle is None:
                    base_angle = a - rotate   # giu anh khong nhay khi vao mode
                rotate = a - base_angle
                gesture = "XOAY ANH"
                # nam tay giu ~0.8s -> thoat
                if nf <= 1:
                    fist_since[i] = fist_since.get(i) or now
                    if now - fist_since[i] > 0.8:
                        mode = "NORMAL"
                        rotate_hand = None
                        fist_since.pop(i, None)
                else:
                    fist_since.pop(i, None)

        # ---- cap nhat so ngon frame truoc cho tung tay (bat "canh len") ----
        for i, _, nf in hands_info:
            prev_nf[i] = nf

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

        # ==== so ngon to giua man hinh (3/4 ngon), mo dan roi bien mat ====
        if now < big_until:
            alpha = min(1.0, (big_until - now) / 0.4)   # mo dan trong 0.4s cuoi
            txt = str(big_num)
            scale, thick = 12.0, 20
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            tx, ty = (W - tw) // 2, (H + th) // 2
            overlay = canvas.copy()
            cv2.putText(overlay, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (80, 220, 255), thick, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

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
                    "1 ngon: quet doi anh | 2 ngon: reset | 3/4 ngon: hien so | "
                    "5 ngon: xoay | Nam tay: thoat xoay | Q/ESC: thoat",
                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow(win, canvas)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
