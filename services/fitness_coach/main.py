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

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

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

    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

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
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cfg = EXERCISES[ex]
        ang = None

        if res.pose_landmarks:
            mp_draw.draw_landmarks(frame, res.pose_landmarks,
                                   mp_pose.POSE_CONNECTIONS)
            ang, joint = pick_side(res.pose_landmarks.landmark, cfg, w, h)

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
