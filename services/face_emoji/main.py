"""
Service: Face Emoji Party (dan emoji nhay nhot len mat).

- Dung MediaPipe FaceLandmarker (Tasks API) de tim mat, mui, tai, mom.
- Tu dong dan 1 emoji len tung bo phan (2 mat, mui, 2 tai, mom), emoji "nhay
  nhay ngoay ngoay" (bounce + lac).
- Moi 5-15 giay (ngau nhien) doi 1 emoji o 1 bo phan bat ky.
- Ha mom -> emoji ban ra lien tuc nhu voi phun, den khi ngam mom lai thi dung.
  (Phat hien ha mom bang blendshape 'jawOpen'.)
- Bo emoji da dang ~30 cai.

Emoji ve bang Pillow + font Segoe UI Emoji (OpenCV khong ve duoc emoji mau);
neu khong co font se fallback ve hinh tron mau.

Model face_landmarker.task tu tai ve models/ lan dau (can mang).
Toi uu chay tren may khong CUDA (Intel Iris Xe): chi 1 mat, emoji render san 1 lan.

Phim tat:
  R : doi ngay toan bo emoji
  M : bat/tat tieng
  + / - : chinh nguong "ha mom"
  Q / ESC : thoat
"""
import argparse
import math
import os
import random
import threading
import time
import urllib.request

import cv2
import numpy as np

try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
    HAS_PIL = True
except Exception:  # noqa
    HAS_PIL = False

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# ---- Bo emoji da dang (~30) ----
EMOJIS = [
    "😀", "😎", "🤪", "🥳", "😍", "🤡", "👽", "🤖", "👻", "💀",
    "🐱", "🐶", "🐸", "🦄", "🐵", "🐷", "🦊", "🐼", "🦁", "🐔",
    "🌟", "🔥", "⚡", "❤️", "💎", "🌈", "🎉", "🌸", "🦋", "🍀",
]
# emoji khi ban ra tu mom (bua tiec bay)
SHOOT_EMOJIS = ["💖", "✨", "🔥", "⭐", "💫", "🎉", "🌈", "💧", "🍬", "🫧"]

EMOJI_FONTS = [
    r"C:\Windows\Fonts\seguiemj.ttf",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
]
BASE = 140  # kich thuoc render goc cua moi sprite emoji

# ---- Chi so landmark FaceMesh/FaceLandmarker (468 diem) ----
REYE = [33, 133, 159, 145, 153, 158]     # mat phai (tren khung da lat guong)
LEYE = [362, 263, 386, 374, 380, 385]    # mat trai
NOSE = [4]                                # dau mui
REAR = [234]                              # ben phai mat (gan tai)
LEAR = [454]                              # ben trai mat (gan tai)


def download_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if os.path.exists(MODEL_PATH):
        return
    print("[i] Tai model face_landmarker.task (chi 1 lan)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"    -> luu {MODEL_PATH}")


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


def render_emoji(ch, font, size, fallback_color=(0, 200, 255)):
    """Render 1 emoji -> anh BGRA (numpy). Fallback: hinh tron mau."""
    d = int(size * 1.4)
    if font is not None:
        img = PILImage.new("RGBA", (d, d), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            draw.text((d // 2, d // 2), ch, font=font, anchor="mm",
                      embedded_color=True)
            rgba = np.array(img)
            if rgba[..., 3].max() > 0:
                return rgba[..., [2, 1, 0, 3]].copy()  # RGBA -> BGRA
        except Exception:  # noqa
            pass
    bgra = np.zeros((d, d, 4), np.uint8)
    cv2.circle(bgra, (d // 2, d // 2), int(size * 0.45), (*fallback_color, 255), -1)
    cv2.circle(bgra, (d // 2, d // 2), int(size * 0.45), (255, 255, 255, 255), 3)
    return bgra


def transform_sprite(bgra, size, angle):
    """Resize sprite ve canh 'size' roi xoay 'angle' do."""
    size = max(8, int(size))
    s = cv2.resize(bgra, (size, size), interpolation=cv2.INTER_LINEAR)
    if abs(angle) < 0.5:
        return s
    m = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    return cv2.warpAffine(s, m, (size, size), flags=cv2.INTER_LINEAR,
                          borderValue=(0, 0, 0, 0))


def overlay_bgra(frame, bgra, cx, cy, alpha=1.0):
    """Dan sprite BGRA vao frame tai tam (cx, cy), tu cat o vien."""
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
    a = crop[..., 3:4].astype(np.float32) / 255.0 * float(alpha)
    roi = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    frame[fy1:fy2, fx1:fx2] = (roi * (1 - a) + crop[..., :3] * a).astype(np.uint8)


def beep(on, freq, dur):
    if not on:
        return

    def _b():
        try:
            import winsound
            winsound.Beep(freq, dur)
        except Exception:  # noqa
            pass
    threading.Thread(target=_b, daemon=True).start()


def mean_point(lm, idxs, w, h):
    xs = [lm[i].x for i in idxs]
    ys = [lm[i].y for i in idxs]
    return np.array([sum(xs) / len(xs) * w, sum(ys) / len(ys) * h])


def jaw_open(res, i=0):
    """Diem 'jawOpen' (0..1) cua khuon mat thu i; 0 neu khong co."""
    if res.face_blendshapes and i < len(res.face_blendshapes):
        for c in res.face_blendshapes[i]:
            if c.category_name == "jawOpen":
                return c.score
    return 0.0


def main():
    ap = argparse.ArgumentParser(description="Face Emoji Party")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", default=None, help="Duong dan video (thay webcam)")
    ap.add_argument("--max-faces", type=int, default=4, help="So khuon mat toi da")
    ap.add_argument("--mouth-th", type=float, default=0.4,
                    help="Nguong ha mom theo blendshape jawOpen (0..1)")
    ap.add_argument("--min-gap", type=float, default=5.0, help="Giay it nhat truoc khi doi emoji")
    ap.add_argument("--max-gap", type=float, default=15.0, help="Giay nhieu nhat truoc khi doi emoji")
    args = ap.parse_args()

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except Exception as e:  # noqa
        print(f"[!] Thieu mediapipe: {e}\n    Chay setup truoc.")
        return

    try:
        download_model()
    except Exception as e:  # noqa
        print(f"[!] Khong tai duoc model: {e}")
        print(f"    Tai thu cong tai: {MODEL_URL}\n    Luu vao: {MODEL_PATH}")
        return

    font = load_emoji_font(BASE)
    if font is None:
        print("[!] Khong tim thay font emoji -> dung hinh tron mau thay the.")

    sprites = {ch: render_emoji(ch, font, BASE) for ch in EMOJIS}
    shoot_sprites = {ch: render_emoji(ch, font, BASE) for ch in SHOOT_EMOJIS}

    # 6 bo phan: mat P, mat T, mui, tai P, tai T, mom
    features = [
        {"key": "reye", "emoji": random.randrange(len(EMOJIS)), "scale": 0.22, "phase": 0.0},
        {"key": "leye", "emoji": random.randrange(len(EMOJIS)), "scale": 0.22, "phase": 1.1},
        {"key": "nose", "emoji": random.randrange(len(EMOJIS)), "scale": 0.30, "phase": 2.2},
        {"key": "rear", "emoji": random.randrange(len(EMOJIS)), "scale": 0.32, "phase": 3.3},
        {"key": "lear", "emoji": random.randrange(len(EMOJIS)), "scale": 0.32, "phase": 4.4},
        {"key": "mouth", "emoji": random.randrange(len(EMOJIS)), "scale": 0.34, "phase": 5.5},
    ]

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=max(1, args.max_faces),
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    src = args.source if args.source else args.camera
    use_dshow = os.name == "nt" and isinstance(src, int)
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if use_dshow else 0)
    if not cap.isOpened():
        print(f"[!] Khong mo duoc camera/nguon: {src}")
        return

    win = "Face Emoji Party"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    sound_on = True
    particles = []
    t0 = time.time()
    next_change = t0 + random.uniform(args.min_gap, args.max_gap)
    last_shot = 0.0
    fps, prev_t = 0.0, t0
    frame_i = 0

    def change_one():
        f = random.choice(features)
        old = f["emoji"]
        while len(EMOJIS) > 1 and f["emoji"] == old:
            f["emoji"] = random.randrange(len(EMOJIS))
        beep(sound_on, 1200, 70)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # guong nhu selfie
        h, w = frame.shape[:2]
        now = time.time()
        t = now - t0
        dt = max(now - prev_t, 1e-3)
        frame_i += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(mp_img, frame_i * 33)

        mouth_open_any = False
        if res.face_landmarks:
            open_mouths = []   # (tam mieng, be rong mat) cua cac mat dang ha mom
            # dung chung 1 bo emoji cho moi khuon mat (hieu ung bua tiec)
            for fi, lm in enumerate(res.face_landmarks):
                face_w = abs(lm[LEAR[0]].x - lm[REAR[0]].x) * w
                face_w = max(face_w, 40.0)

                anchors = {
                    "reye": mean_point(lm, REYE, w, h),
                    "leye": mean_point(lm, LEYE, w, h),
                    "nose": mean_point(lm, NOSE, w, h),
                    "rear": mean_point(lm, REAR, w, h),
                    "lear": mean_point(lm, LEAR, w, h),
                    "mouth": mean_point(lm, [13, 14], w, h),
                }
                this_open = jaw_open(res, fi) > args.mouth_th
                mouth_open_any = mouth_open_any or this_open

                # pha dao dong lech theo tung mat cho sinh dong
                fp = fi * 0.7
                for f in features:
                    p = anchors[f["key"]]
                    size = f["scale"] * face_w
                    bounce = math.sin(t * 6.0 + f["phase"] + fp) * size * 0.12
                    wiggle = math.sin(t * 8.0 + f["phase"] + fp) * 16.0
                    ch = EMOJIS[f["emoji"]]
                    spr = transform_sprite(sprites[ch], size, wiggle)
                    overlay_bgra(frame, spr, p[0], p[1] - bounce)

                if this_open:
                    open_mouths.append((anchors["mouth"], face_w))

            # moi mieng dang ha deu phun emoji
            if open_mouths and now - last_shot > 0.05:
                for mp_pt, face_w in open_mouths:
                    for _ in range(2):
                        ang = random.uniform(0.35, 0.65) * math.pi
                        spd = random.uniform(4.0, 9.0)
                        particles.append({
                            "x": mp_pt[0] + random.uniform(-6, 6),
                            "y": mp_pt[1] + random.uniform(-2, 6),
                            "vx": math.cos(ang) * spd * random.choice([-1, 1]),
                            "vy": -abs(math.sin(ang)) * spd * random.uniform(0.3, 1.0) + random.uniform(0, 3),
                            "spin": random.uniform(-8, 8),
                            "ang": random.uniform(0, 360),
                            "size": random.uniform(0.20, 0.34) * face_w,
                            "ch": random.choice(SHOOT_EMOJIS),
                            "life": 1.0,
                        })
                last_shot = now
        else:
            cv2.putText(frame, "Dua mat vao khung hinh...", (20, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)

        if now >= next_change:
            change_one()
            next_change = now + random.uniform(args.min_gap, args.max_gap)

        alive = []
        for pt in particles:
            pt["vy"] += 0.35
            pt["x"] += pt["vx"]
            pt["y"] += pt["vy"]
            pt["ang"] += pt["spin"]
            pt["life"] -= dt * 0.4
            if pt["life"] > 0 and -60 < pt["x"] < w + 60 and pt["y"] < h + 60:
                spr = transform_sprite(shoot_sprites[pt["ch"]], pt["size"], pt["ang"])
                overlay_bgra(frame, spr, pt["x"], pt["y"], alpha=min(1.0, pt["life"] * 1.5))
                alive.append(pt)
        particles = alive[-300:]

        now2 = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now2 - prev_t, 1e-6))
        prev_t = now2

        cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
        state = "HA MOM - dang phun!" if mouth_open else "ngam mom"
        cv2.putText(frame, f"FPS:{fps:4.1f}  {state}  |  R:doi het  M:tieng  +/-:nguong mom  Q:thoat",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)

        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        elif k == ord("r"):
            for f in features:
                f["emoji"] = random.randrange(len(EMOJIS))
            beep(sound_on, 1500, 90)
            next_change = now + random.uniform(args.min_gap, args.max_gap)
        elif k == ord("m"):
            sound_on = not sound_on
        elif k in (ord("+"), ord("=")):
            args.mouth_th = min(1.0, args.mouth_th + 0.05)
        elif k in (ord("-"), ord("_")):
            args.mouth_th = max(0.05, args.mouth_th - 0.05)

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
