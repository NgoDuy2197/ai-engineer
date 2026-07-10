"""
Service: Space Dodge (lai phi thuyen ne thien thach bang khuon mat).

- Di chuyen dau trai-phai de lai phi thuyen o day man hinh.
- Thien thach roi tu tren xuong, cang ve sau cang nhanh & nhieu.
- Va vao thien thach -> mat 1 mang (3 mang). Song cang lau diem cang cao.
- Ngoi sao vang: bay vao huong no -> +10 diem thuong.

Dung MediaPipe FaceDetector (Tasks API) - chay CPU, hop Intel UHD 620 / Iris Xe.

Phim tat:
  C : choi lai
  Q / ESC : thoat
"""
import argparse
import math
import os
import random
import threading

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "blaze_face_short_range.tflite")

SHIP_R = 34            # ban kinh va cham cua phi thuyen
STAR_BG = 70           # so ngoi sao nen


def beep(freq, dur):
    def _b():
        try:
            import winsound
            winsound.Beep(freq, dur)
        except Exception:  # noqa
            pass
    threading.Thread(target=_b, daemon=True).start()


def draw_ship(frame, x, y):
    pts = np.array([[x, y - 26], [x - 22, y + 22], [x + 22, y + 22]], np.int32)
    cv2.fillPoly(frame, [pts], (230, 180, 40))
    cv2.polylines(frame, [pts], True, (255, 240, 180), 2)
    cv2.circle(frame, (x, y), 7, (255, 255, 255), -1)              # buong lai
    # lua phut phia sau
    fl = random.randint(10, 22)
    cv2.line(frame, (x, y + 22), (x, y + 22 + fl), (0, 140, 255), 4)


def main():
    ap = argparse.ArgumentParser(description="Ne thien thach bang khuon mat")
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
              "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
        return
    options = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_detection_confidence=0.5)
    detector = mp_vision.FaceDetector.create_from_options(options)

    win = "Space Dodge"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # sao nen (toa do ti le 0..1 + toc do)
    stars = [[random.random(), random.random(), random.uniform(1.5, 4.0)]
             for _ in range(STAR_BG)]

    def new_game():
        return {"rocks": [], "bonus": [], "score": 0, "best": 0,
                "lives": 3, "over": False, "frame": 0, "hit_flash": 0}

    G = new_game()
    ship_x = None
    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_i += 1
        G["frame"] += 1
        if ship_x is None:
            ship_x = w / 2
        ship_y = h - 70

        # ==== lam toi nen + ve sao chay ====
        frame = (frame * 0.35).astype(np.uint8)
        for s in stars:
            s[1] += s[2] / h
            if s[1] > 1.0:
                s[0], s[1] = random.random(), 0.0
            cv2.circle(frame, (int(s[0] * w), int(s[1] * h)),
                       1 if s[2] < 3 else 2, (200, 200, 200), -1)

        # ==== khuon mat -> vi tri phi thuyen ====
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect_for_video(mp_img, frame_i * 33)
        face_found = False
        if res.detections:
            box = res.detections[0].bounding_box     # toa do PIXEL
            fx = box.origin_x + box.width / 2.0
            ship_x = 0.7 * ship_x + 0.3 * fx         # lam muot
            face_found = True
        ship_x = float(np.clip(ship_x, SHIP_R, w - SHIP_R))
        sx, sy = int(ship_x), int(ship_y)

        # ==== logic game ====
        if not G["over"]:
            level = G["frame"] / 600.0               # do kho tang dan
            spawn_every = max(9, int(20 - level * 3))
            if G["frame"] % spawn_every == 0:
                r = random.randint(20, 40)
                G["rocks"].append({"x": random.uniform(r, w - r), "y": -r, "r": r,
                                   "vy": random.uniform(4.5, 6.5) + level * 1.2,
                                   "angle": 0.0, "spin": random.uniform(-6, 6)})
            if G["frame"] % 150 == 0:                # sao thuong
                G["bonus"].append({"x": random.uniform(40, w - 40), "y": -20,
                                   "vy": random.uniform(4, 5.5), "a": 0.0})

            alive = []
            for o in G["rocks"]:
                o["y"] += o["vy"]
                o["angle"] += o["spin"]
                if math.hypot(o["x"] - ship_x, o["y"] - ship_y) < o["r"] + SHIP_R:
                    G["lives"] -= 1
                    G["hit_flash"] = 8
                    beep(300, 120)
                    if G["lives"] <= 0:
                        G["over"] = True
                    continue
                if o["y"] - o["r"] > h:
                    G["score"] += 1                  # ne thanh cong
                    continue
                alive.append(o)
            G["rocks"] = alive

            alive_b = []
            for b in G["bonus"]:
                b["y"] += b["vy"]
                b["a"] += 9
                if math.hypot(b["x"] - ship_x, b["y"] - ship_y) < 18 + SHIP_R:
                    G["score"] += 10
                    beep(1500, 90)
                    continue
                if b["y"] > h + 20:
                    continue
                alive_b.append(b)
            G["bonus"] = alive_b
            G["score"] += 0                          # (diem chinh tu ne da)

        # ==== ve thien thach ====
        for o in G["rocks"]:
            x, y, r = int(o["x"]), int(o["y"]), o["r"]
            cv2.circle(frame, (x, y), r, (110, 110, 120), -1)
            cv2.circle(frame, (x, y), r, (60, 60, 70), 3)
            cv2.circle(frame, (x - r // 3, y - r // 3), max(2, r // 5),
                       (150, 150, 160), -1)

        # ==== ve sao thuong ====
        for b in G["bonus"]:
            x, y = int(b["x"]), int(b["y"])
            m = cv2.getRotationMatrix2D((0, 0), b["a"], 1.0)
            spikes = []
            for k in range(10):
                rad = 16 if k % 2 == 0 else 7
                th = math.radians(k * 36)
                spikes.append([x + int(rad * math.cos(th)), y + int(rad * math.sin(th))])
            cv2.fillPoly(frame, [np.array(spikes, np.int32)], (0, 220, 255))

        # ==== phi thuyen + hieu ung trung ====
        if G["hit_flash"] > 0:
            G["hit_flash"] -= 1
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 12)
        if not G["over"]:
            draw_ship(frame, sx, sy)

        # ==== HUD ====
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"DIEM: {G['score']}", (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 180), 2)
        for i in range(3):
            c = (0, 0, 230) if i < G["lives"] else (70, 70, 70)
            cv2.circle(frame, (w - 30 - i * 34, 24), 10, c, -1)

        if not face_found and not G["over"]:
            cv2.putText(frame, "Khong thay mat - dua mat vao khung",
                        (w // 2 - 220, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 200, 255), 2)

        if G["over"]:
            G["best"] = max(G["best"], G["score"])
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, "NO TUNG!", (w // 2 - 150, h // 2 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 80, 255), 5)
            cv2.putText(frame, f"Diem: {G['score']}   (Ky luc: {G['best']})",
                        (w // 2 - 190, h // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 180), 2)
            cv2.putText(frame, "Nhan C de choi lai  |  Q de thoat",
                        (w // 2 - 200, h // 2 + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        else:
            cv2.putText(frame, "Nghieng dau trai/phai de ne  |  Sao vang: +10",
                        (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        elif k == ord("c"):
            best = max(G["best"], G["score"])
            G = new_game()
            G["best"] = best

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
