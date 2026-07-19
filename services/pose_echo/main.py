"""
Service: Pose Echo / Bong Que Tre (ghi lai chuyen dong, ve nguoi que tre ~3 giay).

- Camera ghi lai tu the (pose) cua ban lien tuc vao bo nho dem.
- Sau khoang 3 giay dau (dang ghi), mot "nguoi que" mau xanh hien ra va
  LAM LAI dung nhung gi ban da lam ~3 giay truoc -> nhu co mot cai bong
  di sau lung ban dung 3 giay.
- Co the chinh do tre bang phim + / - (0.5s den 6s).

Dung MediaPipe PoseLandmarker (Tasks API) - chay CPU, hop Intel UHD 620 / Iris Xe.

Phim tat:
  + / - : tang / giam do tre
  L     : bat/tat ve khung xuong hien tai (mau xam mo)
  C     : xoa bo nho ghi (ghi lai tu dau)
  Q/ESC : thoat
"""
import argparse
import os
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "pose_landmarker_lite.task")

# Ket noi khung xuong Pose 33 diem.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]
VIS_MIN = 0.3


def draw_skeleton(frame, pts, w, h, color, thick, pt_r, alpha):
    """Ve nguoi que tu danh sach diem chuan hoa [(x,y,visibility)]. alpha<1 = mo."""
    if not pts:
        return
    pix = [(int(x * w), int(y * h), v) for (x, y, v) in pts]
    layer = frame if alpha >= 1.0 else frame.copy()
    for a, b in POSE_CONNECTIONS:
        if pix[a][2] > VIS_MIN and pix[b][2] > VIS_MIN:
            cv2.line(layer, pix[a][:2], pix[b][:2], color, thick, cv2.LINE_AA)
    for x, y, v in pix:
        if v > VIS_MIN:
            cv2.circle(layer, (x, y), pt_r, color, -1)
    # dau to hon 1 chut (landmark 0 = mui)
    if pix[0][2] > VIS_MIN:
        cv2.circle(layer, pix[0][:2], pt_r * 3, color, 2)
    if alpha < 1.0:
        cv2.addWeighted(layer, alpha, frame, 1 - alpha, 0, frame)


def main():
    ap = argparse.ArgumentParser(description="Ve nguoi que tre 3 giay")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--delay", type=float, default=3.0, help="Do tre (giay)")
    ap.add_argument("--max-poses", type=int, default=2,
                    help="So nguoi toi da nhan dien (num_poses)")
    args = ap.parse_args()

    max_poses = max(1, args.max_poses)

    use_dshow = os.name == "nt"
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if use_dshow else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("[!] Khong mo duoc camera")
        return

    if not os.path.isfile(MODEL_PATH):
        print(f"[!] Khong tim thay model: {MODEL_PATH}")
        print("    Tai: https://storage.googleapis.com/mediapipe-models/"
              "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
        return
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=max_poses,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    pose = mp_vision.PoseLandmarker.create_from_options(options)
    t_start = time.time()
    last_ts = -1

    delay = max(0.5, args.delay)
    # PER-PERSON: moi nguoi (theo chi so pose) co bo dem lich su rieng.
    # bufs[i] = deque cac (thoi_gian, pts | None) cua nguoi thu i.
    bufs = {}                    # {chi_so_pose: deque((thoi_gian, pts | None))}
    record_start = time.time()   # moc bat dau ghi (de tinh khi nao bong hien)
    show_live = True

    win = "Pose Echo - Bong Que Tre"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()

        # ==== nhan dien pose hien tai ====
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((now - t_start) * 1000)
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts
        res = pose.detect_for_video(mp_img, ts)

        # ==== gom pose cua tung nguoi (theo chi so) ====
        # LAP qua tat ca pose trong ket qua, moi nguoi mot danh sach diem.
        cur_by_person = {}
        if res.pose_landmarks:
            for i, lm in enumerate(res.pose_landmarks):
                cur_by_person[i] = [(p.x, p.y, p.visibility) for p in lm]

        # ==== luu vao bo dem tung nguoi + cat bo phan cu hon do tre ====
        # Ghi cho tat ca chi so da tung xuat hien (nguoi vang mat -> luu None)
        # de bong van chay lien tuc theo thoi gian.
        active_idx = set(bufs.keys()) | set(cur_by_person.keys())
        for i in active_idx:
            b = bufs.setdefault(i, deque())
            b.append((now, cur_by_person.get(i)))
            while len(b) >= 2 and now - b[0][0] > delay:
                b.popleft()

        # ve khung xuong hien tai (mo, de tham chieu) cho TUNG nguoi
        if show_live:
            for cur_pts in cur_by_person.values():
                draw_skeleton(frame, cur_pts, w, h, (170, 170, 170), 2, 3, 0.35)

        # ==== ve "bong" tre ~delay giay cho TUNG nguoi ====
        elapsed = now - record_start
        if elapsed >= delay:
            any_ghost = False
            for b in bufs.values():
                if not b:
                    continue
                ghost_pts = b[0][1]                   # frame cu nhat trong cua so ~ delay
                if ghost_pts is not None:
                    any_ghost = True
                draw_skeleton(frame, ghost_pts, w, h, (80, 230, 90), 5, 6, 0.9)
            if not any_ghost:
                cv2.putText(frame, "(3 giay truoc chua thay nguoi)",
                            (w // 2 - 200, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 230, 90), 2)
        else:
            remain = delay - elapsed
            cv2.putText(frame, f"Dang ghi... bong hien sau {remain:0.1f}s",
                        (w // 2 - 220, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)

        # ==== HUD ====
        cv2.rectangle(frame, (0, 0), (w, 44), (0, 0, 0), -1)
        cv2.putText(frame, f"Do tre: {delay:0.1f}s   (+/- de chinh)", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 230, 90), 2)
        cv2.putText(frame,
                    "Nguoi que XANH = ban cua 3s truoc  |  L:khung hien tai  C:ghi lai  Q:thoat",
                    (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        elif k in (ord("+"), ord("=")):
            delay = min(6.0, delay + 0.5)
            bufs.clear()          # xoa buffer tat ca nguoi
            record_start = now
        elif k in (ord("-"), ord("_")):
            delay = max(0.5, delay - 0.5)
            bufs.clear()          # xoa buffer tat ca nguoi
            record_start = now
        elif k == ord("l"):
            show_live = not show_live
        elif k == ord("c"):
            bufs.clear()          # xoa buffer tat ca nguoi
            record_start = now

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()
