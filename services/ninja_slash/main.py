"""
Service: Ninja Slash (chem hoa qua bay bang dau ngon tay).

- Hoa qua bay len tu duoi (co trong luc roi roi xuong). Dua dau ngon TRO
  "chem" that nhanh qua chung de cat -> +diem, van toe nuoc.
- Chem lien tiep nhieu qua -> COMBO nhan diem.
- Thinh thoang co BOM (den, ngoi lua): chem trung bom -> THUA ngay.
- De qua roi khoi man (khong chem) -> mat 1 mang. Het 3 mang -> thua.
- Luoi kiem la vet sang chay theo dau ngon tro.

Dung MediaPipe HandLandmarker (Tasks API) - chay CPU, hop Intel UHD 620 / Iris Xe.

Phim tat:
  C : choi lai
  Q / ESC : thoat
"""
import argparse
import math
import os
import random
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "hand_landmarker.task")

FRUIT_COLORS = [
    (0, 0, 220), (0, 140, 255), (0, 220, 255), (60, 200, 60),
    (200, 80, 200), (200, 160, 0), (60, 60, 230),
]


def pt_seg_dist(p, a, b):
    """Khoang cach tu diem p den doan thang a-b (de kiem tra luoi kiem cat qua)."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def spawn_object(w, h, bomb_chance):
    kind = "bomb" if random.random() < bomb_chance else "fruit"
    r = random.randint(34, 46)
    color = (35, 35, 35) if kind == "bomb" else random.choice(FRUIT_COLORS)
    return {"x": random.uniform(w * 0.15, w * 0.85), "y": h + r,
            "vx": random.uniform(-2.6, 2.6), "vy": random.uniform(-20, -15.5),
            "r": r, "kind": kind, "color": color,
            "angle": random.uniform(0, 360), "spin": random.uniform(-9, 9)}


def spawn_particle(lst, x, y, color):
    ang = random.uniform(0, 2 * math.pi)
    spd = random.uniform(2, 8)
    lst.append({"x": x, "y": y, "vx": math.cos(ang) * spd,
                "vy": math.sin(ang) * spd, "life": 1.0, "color": color})


def draw_object(frame, o):
    x, y, r = int(o["x"]), int(o["y"]), o["r"]
    if o["kind"] == "bomb":
        cv2.circle(frame, (x, y), r, (35, 35, 35), -1)
        cv2.circle(frame, (x, y), r, (0, 0, 210), 3)
        cv2.putText(frame, "!", (x - 7, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        cv2.circle(frame, (x + int(r * 0.4), y - r), 4, (0, 210, 255), -1)  # ngoi lua
    else:
        cv2.circle(frame, (x, y), r, o["color"], -1)
        hi = tuple(min(255, c + 80) for c in o["color"])
        cv2.circle(frame, (x - int(r * 0.3), y - int(r * 0.3)), int(r * 0.35), hi, -1)
        cv2.circle(frame, (x, y), r, (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser(description="Chem hoa qua bay bang ngon tay")
    ap.add_argument("--camera", type=int, default=0)
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
        print("    Tai: https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        return
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5)
    landmarker = mp_vision.HandLandmarker.create_from_options(options)
    t_start = time.time()
    last_ts = -1

    win = "Ninja Slash"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def new_game():
        return {"objs": [], "particles": [], "score": 0, "best": 0,
                "combo": 0, "combo_t": 0.0, "lives": 3, "over": False, "frame": 0}

    G = new_game()
    trail = deque(maxlen=7)
    spawn_every = 22

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        G["frame"] += 1
        now = time.time()

        # ==== tay ====
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((now - t_start) * 1000)
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts
        res = landmarker.detect_for_video(mp_img, ts)

        tip = None
        if res.hand_landmarks:
            lm = res.hand_landmarks[0]
            tip = (lm[8].x * w, lm[8].y * h)     # dau ngon tro
            trail.append(tip)
        else:
            trail.clear()

        blade_fast = False
        if len(trail) >= 2:
            spd = math.hypot(trail[-1][0] - trail[-2][0], trail[-1][1] - trail[-2][1])
            blade_fast = spd > 14                # phai vung nhanh moi cat duoc

        # ==== logic game ====
        if not G["over"]:
            if G["frame"] % spawn_every == 0:
                bomb_chance = min(0.22, 0.05 + G["score"] * 0.002)
                G["objs"].append(spawn_object(w, h, bomb_chance))
                if random.random() < 0.35:
                    G["objs"].append(spawn_object(w, h, bomb_chance))

            alive = []
            for o in G["objs"]:
                o["vy"] += 0.42                  # trong luc
                o["x"] += o["vx"]
                o["y"] += o["vy"]
                o["angle"] += o["spin"]

                if (tip is not None and blade_fast and len(trail) >= 2
                        and pt_seg_dist((o["x"], o["y"]), trail[-2], trail[-1]) < o["r"] + 6):
                    if o["kind"] == "bomb":
                        G["over"] = True
                        for _ in range(60):
                            spawn_particle(G["particles"], o["x"], o["y"], (40, 40, 230))
                    else:
                        G["combo"] = G["combo"] + 1 if now - G["combo_t"] < 0.8 else 1
                        G["combo_t"] = now
                        G["score"] += G["combo"]
                        for _ in range(18):
                            spawn_particle(G["particles"], o["x"], o["y"], o["color"])
                    continue

                if o["y"] - o["r"] > h + 40:     # roi khoi man
                    if o["kind"] == "fruit":
                        G["lives"] -= 1
                        if G["lives"] <= 0:
                            G["over"] = True
                    continue
                alive.append(o)
            G["objs"] = alive

        for o in G["objs"]:
            draw_object(frame, o)

        # ==== van nuoc / manh vo ====
        alive_p = []
        for p in G["particles"]:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.3
            p["life"] -= 0.04
            if p["life"] > 0:
                c = tuple(int(v * p["life"]) for v in p["color"])
                cv2.circle(frame, (int(p["x"]), int(p["y"])),
                           max(1, int(5 * p["life"])), c, -1)
                alive_p.append(p)
        G["particles"] = alive_p

        # ==== luoi kiem ====
        if tip is not None and len(trail) >= 2:
            pts_ = list(trail)
            for i in range(1, len(pts_)):
                col = (255, 255, 200) if blade_fast else (180, 180, 120)
                cv2.line(frame, (int(pts_[i - 1][0]), int(pts_[i - 1][1])),
                         (int(pts_[i][0]), int(pts_[i][1])),
                         col, int(2 + i * 1.6), cv2.LINE_AA)
            cv2.circle(frame, (int(tip[0]), int(tip[1])), 10, (255, 255, 255), 2)

        # ==== HUD ====
        cv2.rectangle(frame, (0, 0), (w, 54), (0, 0, 0), -1)
        cv2.putText(frame, f"DIEM: {G['score']}", (16, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 180), 2)
        if G["combo"] >= 2 and now - G["combo_t"] < 0.8:
            cv2.putText(frame, f"COMBO x{G['combo']}!", (w // 2 - 90, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 220, 255), 2)
        for i in range(3):                       # mang = trai tim
            c = (0, 0, 230) if i < G["lives"] else (70, 70, 70)
            cx = w - 40 - i * 40
            cv2.circle(frame, (cx - 6, 24), 8, c, -1)
            cv2.circle(frame, (cx + 6, 24), 8, c, -1)
            cv2.drawContours(frame, [np.array([[cx - 14, 27], [cx + 14, 27],
                                                [cx, 44]], np.int32)], 0, c, -1)

        if G["over"]:
            G["best"] = max(G["best"], G["score"])
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, "THUA!", (w // 2 - 110, h // 2 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
            cv2.putText(frame, f"Diem: {G['score']}   (Ky luc: {G['best']})",
                        (w // 2 - 190, h // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 180), 2)
            cv2.putText(frame, "Nhan C de choi lai  |  Q de thoat",
                        (w // 2 - 200, h // 2 + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        else:
            cv2.putText(frame, "Vung nhanh ngon tro de chem  |  Tranh BOM (den)!",
                        (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        elif k == ord("c"):
            best = max(G["best"], G["score"])
            G = new_game()
            G["best"] = best
            trail.clear()

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
