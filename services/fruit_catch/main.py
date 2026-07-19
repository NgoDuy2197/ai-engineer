"""
Service: Fruit Catch (game hung hoa qua bang khuon mat).

- MediaPipe Face Detection dinh vi mat -> cai RO o day man hinh chay theo mat
  (nghieng/di chuyen dau sang trai-phai de dua ro).
- Hoa qua (emoji) roi tu tren xuong, lon vong (xoay).
- Hung trung -> +1 diem + tieng "tinh" nhe.
- Cu hung du 10 qua -> ban PHAO HOA dep ~3 giay.

Emoji duoc ve bang Pillow + font Segoe UI Emoji (OpenCV khong ve duoc emoji mau);
neu khong co font se tu dong ve qua bang hinh tron mau (fallback).

Chay tren CPU (MediaPipe) -> nhe cho Intel UHD 620 / Iris Xe.

Phim tat:
  C : choi lai (reset diem)
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

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except Exception:  # noqa
    HAS_PIL = False

FRUITS = ["🍎", "🍊", "🍌", "🍇", "🍓", "🍉", "🍑", "🍒", "🥝", "🍍"]
# mau fallback (BGR) tuong ung tung qua khi khong co font emoji
FRUIT_COLORS = [
    (0, 0, 220), (0, 140, 255), (0, 220, 255), (150, 0, 120), (60, 60, 230),
    (60, 200, 60), (120, 160, 255), (60, 0, 200), (60, 200, 160), (0, 200, 230),
]
EMOJI_FONTS = [
    r"C:\Windows\Fonts\seguiemj.ttf",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
]


def load_emoji_font(size):
    if not HAS_PIL:
        return None
    for path in EMOJI_FONTS:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # noqa
                continue
    return None


def render_emoji(ch, font, size, fallback_color):
    """Render 1 emoji -> anh BGRA (numpy). Fallback: hinh tron mau."""
    d = int(size * 1.5)  # canvas vuong rong hon de xoay khong bi cat goc
    if font is not None:
        img = Image.new("RGBA", (d, d), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            draw.text((d // 2, d // 2), ch, font=font, anchor="mm",
                      embedded_color=True)
            rgba = np.array(img)                 # R,G,B,A
            if rgba[..., 3].max() > 0:
                return rgba[..., [2, 1, 0, 3]].copy()  # -> B,G,R,A
        except Exception:  # noqa
            pass
    # fallback: qua tron
    bgra = np.zeros((d, d, 4), np.uint8)
    cv2.circle(bgra, (d // 2, d // 2), int(size * 0.45), (*fallback_color, 255), -1)
    cv2.circle(bgra, (d // 2, d // 2), int(size * 0.45), (255, 255, 255, 255), 2)
    return bgra


def rotate_bgra(bgra, angle):
    h, w = bgra.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(bgra, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderValue=(0, 0, 0, 0))


def overlay_bgra(frame, bgra, cx, cy):
    """Dan anh BGRA vao frame tai tam (cx, cy), tu cat o vien."""
    bh, bw = bgra.shape[:2]
    x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
    x2, y2 = x1 + bw, y1 + bh
    fh, fw = frame.shape[:2]
    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(fw, x2), min(fh, y2)
    if fx1 >= fx2 or fy1 >= fy2:
        return
    rx1, ry1 = fx1 - x1, fy1 - y1
    rx2, ry2 = rx1 + (fx2 - fx1), ry1 + (fy2 - fy1)
    crop = bgra[ry1:ry2, rx1:rx2]
    a = crop[..., 3:4].astype(np.float32) / 255.0
    roi = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    frame[fy1:fy2, fx1:fx2] = (roi * (1 - a) + crop[..., :3] * a).astype(np.uint8)


def beep(freq, dur):
    """Tieng ngan, khong chan luong chinh (chi Windows)."""
    def _b():
        try:
            import winsound
            winsound.Beep(freq, dur)
        except Exception:  # noqa
            pass
    threading.Thread(target=_b, daemon=True).start()


def draw_basket(frame, basket_x, bw2, basket_top, h):
    """Ve 1 cai ro tai vi tri basket_x."""
    bl, br = int(basket_x - bw2), int(basket_x + bw2)
    by, bb = basket_top, h - 24
    poly = np.array([[bl, by], [br, by],
                     [int(basket_x + bw2 * 0.6), bb],
                     [int(basket_x - bw2 * 0.6), bb]], np.int32)
    cv2.fillPoly(frame, [poly], (40, 90, 160))
    cv2.polylines(frame, [poly], True, (30, 60, 110), 3)
    cv2.line(frame, (bl, by), (br, by), (60, 140, 220), 6)   # mieng ro
    for gx in range(bl + 20, br, 26):                        # nan doc
        cv2.line(frame, (gx, by), (int(gx * 0.9 + basket_x * 0.1), bb),
                 (30, 60, 110), 1)


def spawn_fireworks(particles, w, h):
    """Tao 1 chum phao hoa tai vi tri ngau nhien."""
    cx = random.randint(int(w * 0.2), int(w * 0.8))
    cy = random.randint(int(h * 0.15), int(h * 0.5))
    color = (random.randint(120, 255), random.randint(120, 255), random.randint(120, 255))
    for _ in range(46):
        ang = random.uniform(0, 2 * math.pi)
        spd = random.uniform(2.5, 7.0)
        particles.append({
            "x": cx, "y": cy,
            "vx": math.cos(ang) * spd, "vy": math.sin(ang) * spd,
            "life": 1.0, "color": color,
        })


def main():
    ap = argparse.ArgumentParser(description="Game hung hoa qua bang khuon mat")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--size", type=int, default=70, help="Kich thuoc emoji hoa qua")
    ap.add_argument("--speed", type=float, default=1.0, help="He so toc do roi")
    ap.add_argument("--max-faces", type=int, default=0,
                    help="Gioi han so mat/ro (0 = khong gioi han)")
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
              "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
        return
    options = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_detection_confidence=0.5)
    face = mp_vision.FaceDetector.create_from_options(options)

    font = load_emoji_font(args.size)
    if font is None:
        print("[!] Khong tim thay font emoji -> ve qua bang hinh tron mau.")
    # cache anh goc cho tung qua
    sprites = [render_emoji(ch, font, args.size, FRUIT_COLORS[i])
               for i, ch in enumerate(FRUITS)]

    score = 0
    missed = 0
    fruits = []              # {x, y, vy, kind, angle, spin}
    particles = []           # phao hoa
    fireworks_until = 0.0
    baskets = []             # danh sach vi tri x cua cac ro (moi mat 1 ro)
    frame_i = 0
    spawn_every = 26         # so frame giua 2 lan roi qua
    win = "Fruit Catch"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)      # guong selfie
        h, w = frame.shape[:2]
        frame_i += 1
        bw2 = 150                       # nua chieu rong mieng ro
        basket_top = h - 96

        # ==== dinh vi TAT CA mat -> moi mat 1 cai ro ====
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = face.detect_for_video(mp_img, frame_i * 33)  # timestamp ms tang dan
        dets = list(res.detections) if res.detections else []
        # gioi han so mat neu duoc yeu cau (lay mat co diem tin cay cao nhat)
        if args.max_faces > 0 and len(dets) > args.max_faces:
            dets.sort(key=lambda d: d.categories[0].score, reverse=True)
            dets = dets[:args.max_faces]
        # tam x cua tung mat (bounding_box la toa do PIXEL, khong phai ti le)
        face_centers = sorted(
            d.bounding_box.origin_x + d.bounding_box.width / 2.0 for d in dets)
        face_found = len(face_centers) > 0

        if face_found:
            # ghep tung ro cu voi mat theo thu tu trai->phai de lam muot
            new_baskets = []
            for i, fx in enumerate(face_centers):
                if i < len(baskets):
                    bx = 0.75 * baskets[i] + 0.25 * fx   # lam muot
                else:
                    bx = fx
                new_baskets.append(float(np.clip(bx, bw2, w - bw2)))
            baskets = new_baskets
        elif not baskets:
            # chua tung thay mat -> giu 1 ro o giua nhu truoc (khong crash)
            baskets = [w / 2.0]

        # ==== sinh qua moi ====
        if frame_i % spawn_every == 0:
            kind = random.randrange(len(FRUITS))
            fruits.append({
                "x": random.uniform(60, w - 60), "y": -40,
                "vy": random.uniform(2.2, 3.6) * args.speed,
                "kind": kind, "angle": random.uniform(0, 360),
                "spin": random.uniform(-7, 7),
            })

        # ==== cap nhat + ve qua ====
        alive = []
        for f in fruits:
            f["vy"] += 0.05 * args.speed        # trong luc nhe
            f["y"] += f["vy"]
            f["angle"] += f["spin"]
            # bat ky ro nao hung trung deu tinh (diem dung chung)
            in_zone = basket_top - 10 <= f["y"] <= basket_top + 46
            caught = in_zone and any(
                bx - bw2 <= f["x"] <= bx + bw2 for bx in baskets)
            if caught:
                score += 1
                beep(1300, 70)                  # tieng "tinh" nhe
                if score % 10 == 0:             # du 10 qua -> phao hoa
                    fireworks_until = frame_i + 90   # ~3s @ ~30fps
                    beep(1800, 220)
                continue
            if f["y"] - args.size > h:
                missed += 1
                continue
            spr = rotate_bgra(sprites[f["kind"]], f["angle"])
            overlay_bgra(frame, spr, int(f["x"]), int(f["y"]))
            alive.append(f)
        fruits = alive

        # ==== ve TAT CA cac ro ====
        for bx in baskets:
            draw_basket(frame, bx, bw2, basket_top, h)

        # ==== phao hoa ====
        if frame_i < fireworks_until and frame_i % 8 == 0:
            spawn_fireworks(particles, w, h)
        alive_p = []
        for p in particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.12
            p["life"] -= 0.03
            if p["life"] > 0:
                c = tuple(int(v * p["life"]) for v in p["color"])
                cv2.circle(frame, (int(p["x"]), int(p["y"])),
                           max(1, int(4 * p["life"])), c, -1)
                alive_p.append(p)
        particles = alive_p
        if frame_i < fireworks_until:
            cv2.putText(frame, "PHAO HOA!  +10 qua!", (w // 2 - 170, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (60, 220, 255), 3)

        # ==== panel ====
        cv2.rectangle(frame, (0, 0), (w, 86), (0, 0, 0), -1)
        cv2.putText(frame, f"DIEM: {score}", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 180), 2)
        n_players = len(baskets) if face_found else 0
        cv2.putText(frame,
                    f"Rot: {missed}    Toi phao hoa: {10 - score % 10} qua"
                    f"    Ro: {n_players}",
                    (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        hint = "Di chuyen dau de dua ro   |   C:choi lai  Q:thoat"
        if not face_found:
            hint = "Khong thay mat - dua mat vao khung   |   C:choi lai  Q:thoat"
        cv2.putText(frame, hint, (16, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        elif k == ord("c"):
            score = missed = 0
            fruits.clear()
            particles.clear()
            fireworks_until = 0.0

    cap.release()
    cv2.destroyAllWindows()
    face.close()


if __name__ == "__main__":
    main()
