"""
Service: Fitness Coach (dem so lan tap qua goc khop - MediaPipe Pose).

Ho tro:
  --exercise squat  : dem SQUAT   (goc dau goi: hong-goi-co chan)
  --exercise curl   : dem GAP TAY (goc khuyu: vai-khuyu-co tay)

Cach dem: khi goc khop di tu "duoi thang" (goc lon) -> "gap sau" (goc nho)
roi tro lai -> tinh 1 lan (rep). Co thanh do sau (%) truc quan.

Chay tren CPU (MediaPipe Pose model_complexity=1) -> hop Intel UHD 620 / Iris Xe.

Phim tat:
  E : doi bai tap (squat <-> curl)
  C : reset bo dem
  Q / ESC : thoat
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
MODEL_PATH = os.path.join(HERE, "models", "pose_landmarker_lite.task")

# Ket noi khung xuong Pose 33 diem (thay cho mp_pose.POSE_CONNECTIONS,
# vi ban mediapipe cho Python 3.13 chi con Tasks API, khong con "solutions").
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


def draw_pose(frame, lms, w, h):
    """Ve khung xuong Pose (thay cho mp_draw.draw_landmarks)."""
    pix = [(int(p.x * w), int(p.y * h)) for p in lms]
    for a, b in POSE_CONNECTIONS:
        cv2.line(frame, pix[a], pix[b], (245, 117, 66), 2)
    for x, y in pix:
        cv2.circle(frame, (x, y), 4, (245, 66, 230), -1)

# Cau hinh moi bai: (bo 3 diem trai, bo 3 diem phai, goc_sau, goc_thang, ten)
# Landmark Pose: vai 11/12, khuyu 13/14, co tay 15/16, hong 23/24, goi 25/26, co chan 27/28
EXERCISES = {
    "squat": {"left": (23, 25, 27), "right": (24, 26, 28),
              "deep": 95, "straight": 165, "label": "SQUAT (dung len - ngoi xuong)"},
    "curl": {"left": (11, 13, 15), "right": (12, 14, 16),
             "deep": 55, "straight": 150, "label": "GAP TAY (bicep curl)"},
}


def angle(a, b, c):
    """Goc (do) tai dinh b tao boi 3 diem a-b-c."""
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    ang = abs(ang)
    return 360 - ang if ang > 180 else ang


def pick_side(lms, cfg, w, h):
    """Chon ben (trai/phai) co do tin cay landmark cao hon; tra goc + toa do khuyu/goi."""
    best = None
    for key in ("left", "right"):
        i, j, k = cfg[key]
        vis = min(lms[i].visibility, lms[j].visibility, lms[k].visibility)
        a = (lms[i].x * w, lms[i].y * h)
        b = (lms[j].x * w, lms[j].y * h)
        c = (lms[k].x * w, lms[k].y * h)
        if best is None or vis > best[0]:
            best = (vis, angle(a, b, c), b)
    return best[1], best[2]  # goc, diem khop giua


def main():
    ap = argparse.ArgumentParser(description="Dem so lan tap qua goc khop")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--exercise", default="squat", choices=list(EXERCISES.keys()))
    args = ap.parse_args()

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
              "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
        return
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    pose = mp_vision.PoseLandmarker.create_from_options(options)
    t_start = time.time()
    last_ts = -1

    ex = args.exercise
    count = 0
    stage = "up"     # up = duoi thang, down = gap sau
    win = "Fitness Coach"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.time() - t_start) * 1000)   # phai tang dan cho mode VIDEO
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts
        res = pose.detect_for_video(mp_img, ts)
        cfg = EXERCISES[ex]
        ang = None

        if res.pose_landmarks:
            lms = res.pose_landmarks[0]
            draw_pose(frame, lms, w, h)
            ang, joint = pick_side(lms, cfg, w, h)

            # may bay trang thai dem rep
            if ang > cfg["straight"]:
                stage = "up"
            if ang < cfg["deep"] and stage == "up":
                stage = "down"
                count += 1

            cv2.putText(frame, f"{int(ang)}", (int(joint[0]) + 10, int(joint[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # ==== phan tram do sau (0% duoi thang -> 100% gap sau) ====
        if ang is not None:
            pct = (cfg["straight"] - ang) / max(1, cfg["straight"] - cfg["deep"])
            pct = int(np.clip(pct, 0, 1) * 100)
        else:
            pct = 0
        bar_y = int(np.interp(pct, [0, 100], [h - 60, 120]))
        cv2.rectangle(frame, (w - 70, 120), (w - 30, h - 60), (80, 80, 80), 2)
        cv2.rectangle(frame, (w - 70, bar_y), (w - 30, h - 60), (0, 200, 0), -1)
        cv2.putText(frame, f"{pct}%", (w - 78, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        # ==== panel ====
        cv2.rectangle(frame, (0, 0), (w, 100), (15, 15, 15), -1)
        cv2.putText(frame, f"{cfg['label']}", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 200, 255), 2)
        cv2.putText(frame, f"So lan: {count}    Trang thai: {stage.upper()}", (14, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2)
        cv2.putText(frame, "E:doi bai  C:reset  Q:thoat", (14, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        elif k == ord("c"):
            count = 0
            stage = "up"
        elif k == ord("e"):
            keys = list(EXERCISES.keys())
            ex = keys[(keys.index(ex) + 1) % len(keys)]
            count = 0
            stage = "up"

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()
