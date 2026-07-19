"""
Service: Fitness Coach (dem so lan tap qua goc khop - MediaPipe Pose).

Ho tro NHIEU nguoi tap cung luc (--max-poses).

  --exercise squat  : dem SQUAT   (goc dau goi: hong-goi-co chan)
  --exercise curl   : dem GAP TAY (goc khuyu: vai-khuyu-co tay)

Cach dem: khi goc khop di tu "duoi thang" (goc lon) -> "gap sau" (goc nho)
roi tro lai -> tinh 1 lan (rep). Co thanh do sau (%) truc quan.
Moi nguoi (theo chi so pose) co bo dem + trang thai RIENG.

Chay tren CPU (MediaPipe Pose Tasks API) -> hop Intel UHD 620 / Iris Xe.

Phim tat:
  E : doi bai tap (squat <-> curl)
  C : reset bo dem (tat ca nguoi)
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

# Bang mau BGR de phan biet tung nguoi (lap vong khi vuot qua)
PERSON_COLORS = [
    (66, 245, 117),   # xanh la
    (245, 117, 66),   # cam
    (117, 66, 245),   # tim
    (66, 200, 245),   # vang
    (245, 66, 200),   # hong
    (200, 245, 66),   # xanh ngoc
]


def draw_pose(frame, lms, w, h, color):
    """Ve khung xuong Pose (thay cho mp_draw.draw_landmarks)."""
    pix = [(int(p.x * w), int(p.y * h)) for p in lms]
    for a, b in POSE_CONNECTIONS:
        cv2.line(frame, pix[a], pix[b], color, 2)
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


def anchor_point(lms, w, h):
    """Diem neo de dat thong tin gan nguoi (uu tien vai, roi hong)."""
    for idx in (11, 12, 23, 24, 0):
        p = lms[idx]
        if getattr(p, "visibility", 1.0) > 0.3:
            return int(p.x * w), int(p.y * h)
    p = lms[0]
    return int(p.x * w), int(p.y * h)


def main():
    ap = argparse.ArgumentParser(description="Dem so lan tap qua goc khop")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--exercise", default="squat", choices=list(EXERCISES.keys()))
    ap.add_argument("--max-poses", type=int, default=2,
                    help="So nguoi toi da nhan dien cung luc")
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
        num_poses=max(1, args.max_poses),
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    pose = mp_vision.PoseLandmarker.create_from_options(options)
    t_start = time.time()
    last_ts = -1

    ex = args.exercise
    # Trang thai PER-PERSON theo chi so pose: {idx: {"count": int, "stage": str}}
    # Luu y: chi so pose giua cac frame khong on dinh tuyet doi -> chap nhan don gian.
    people = {}

    def state_for(idx):
        if idx not in people:
            people[idx] = {"count": 0, "stage": "up"}
        return people[idx]

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

        poses = res.pose_landmarks if res.pose_landmarks else []
        # LAP qua tung nguoi, dem rep doc lap
        for idx, lms in enumerate(poses):
            color = PERSON_COLORS[idx % len(PERSON_COLORS)]
            draw_pose(frame, lms, w, h, color)
            ang, joint = pick_side(lms, cfg, w, h)
            st = state_for(idx)

            # may bay trang thai dem rep cho RIENG nguoi nay
            if ang > cfg["straight"]:
                st["stage"] = "up"
            if ang < cfg["deep"] and st["stage"] == "up":
                st["stage"] = "down"
                st["count"] += 1

            # goc tai khop giua
            cv2.putText(frame, f"{int(ang)}", (int(joint[0]) + 10, int(joint[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # ==== phan tram do sau (0% duoi thang -> 100% gap sau) cua nguoi nay ====
            pct = (cfg["straight"] - ang) / max(1, cfg["straight"] - cfg["deep"])
            pct = int(np.clip(pct, 0, 1) * 100)

            # ==== ve thong tin gan nguoi (theo diem neo vai/hong) ====
            ax, ay = anchor_point(lms, w, h)
            # nhan so rep + ten nguoi (nen mo de de doc)
            tag = f"#{idx + 1}  Reps:{st['count']}  {st['stage'].upper()}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            tx = int(np.clip(ax - tw // 2, 4, w - tw - 4))
            ty = int(np.clip(ay - 40, th + 6, h - 60))
            cv2.rectangle(frame, (tx - 4, ty - th - 6), (tx + tw + 4, ty + 6),
                          (15, 15, 15), -1)
            cv2.putText(frame, tag, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # thanh % nho dat ben canh nguoi nay
            bx = int(np.clip(ax + tw // 2 + 16, 20, w - 20))
            bar_top, bar_bot = ty + 16, ty + 116
            fill_y = int(np.interp(pct, [0, 100], [bar_bot, bar_top]))
            cv2.rectangle(frame, (bx, bar_top), (bx + 16, bar_bot), (80, 80, 80), 2)
            cv2.rectangle(frame, (bx, fill_y), (bx + 16, bar_bot), (0, 200, 0), -1)
            cv2.putText(frame, f"{pct}%", (bx - 6, bar_top - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

        # ==== panel tong ====
        cv2.rectangle(frame, (0, 0), (w, 100), (15, 15, 15), -1)
        cv2.putText(frame, f"{cfg['label']}", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 200, 255), 2)
        total = sum(p["count"] for p in people.values())
        cv2.putText(frame,
                    f"So nguoi: {len(poses)}    Tong reps: {total}", (14, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2)
        cv2.putText(frame, "E:doi bai  C:reset  Q:thoat", (14, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        elif k == ord("c"):
            people.clear()   # reset dem cua TAT CA nguoi
        elif k == ord("e"):
            keys = list(EXERCISES.keys())
            ex = keys[(keys.index(ex) + 1) % len(keys)]
            people.clear()   # doi bai -> reset tat ca

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()
