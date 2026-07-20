"""
Service: Angel Wings (gan canh vao nguoi qua camera).

- Dung MediaPipe PoseLandmarker (Tasks API) bat NGUOI (1 hoac nhieu nguoi).
- Ban bo 1 anh PNG "canh trai" (nen trong suot) vao thu muc __data/wings/.
  Canh phai = lat guong tu canh trai. Chua co PNG thi tu ve 1 canh mac dinh.
- Gan canh vao vai/lung tung nguoi, tu dong scale theo be rong vai.
- Phim F: bat/tat che do VAY CANH (canh dap len xuong).
- Phim E: bat/tat che do HIEU UNG (canh phat ra hat lap lanh).

Toi uu cho may khong CUDA (Intel Iris Xe): dung pose_landmarker_lite, resize
canh theo tung nguoi, so nguoi gioi han qua --max-poses.

Phim tat:
  F : bat/tat vay canh
  E : bat/tat hieu ung phat sang
  + / - : phong to / thu nho canh
  Q / ESC : thoat
"""
import argparse
import math
import os
import random
import time
import traceback
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "__data")
WINGS_DIR = os.path.join(DATA_DIR, "wings")
MODEL_PATH = os.path.join(HERE, "models", "pose_landmarker_lite.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")

L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24


def ensure_dirs():
    for d in (os.path.dirname(MODEL_PATH), WINGS_DIR):
        os.makedirs(d, exist_ok=True)


def download_model():
    ensure_dirs()
    if os.path.exists(MODEL_PATH):
        return
    print("[i] Tai model pose_landmarker_lite.task (chi 1 lan)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"    -> luu {MODEL_PATH}")


def imread_unicode(path, flags=cv2.IMREAD_UNCHANGED):
    """Doc anh chiu duoc duong dan co dau (tieng Viet)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:  # noqa
        return None


def make_default_wing(h=420):
    """Tu ve 1 canh trai mac dinh (BGRA), goc gan vao than o canh PHAI-giua."""
    w = int(h * 1.05)
    img = np.zeros((h, w, 4), np.uint8)
    root = (int(w * 0.9), int(h * 0.5))     # goc canh (gan vao vai)
    n = 11
    for i in range(n):
        t = i / (n - 1)
        ang = math.radians(8 + t * 78)      # xoe len-trai
        length = w * (0.42 + 0.5 * math.sin(t * math.pi))
        tipx = int(root[0] - math.cos(ang) * length)
        tipy = int(root[1] - math.sin(ang) * length)
        thick = int(h * (0.11 - 0.05 * t))
        col = (255, 240 - int(40 * t), 210 - int(30 * t), 255)  # trang hoi xanh
        cv2.line(img, root, (tipx, tipy), col, max(3, thick), cv2.LINE_AA)
        cv2.circle(img, (tipx, tipy), max(2, thick // 2), col, -1, cv2.LINE_AA)
    # vien long vu mo
    cv2.circle(img, root, int(h * 0.09), (255, 255, 255, 255), -1, cv2.LINE_AA)
    return img


def _load_rgba(path):
    img = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 3:      # them alpha dac neu thieu
        a = np.full(img.shape[:2] + (1,), 255, np.uint8)
        img = np.concatenate([img, a], axis=2)
    return img.astype(np.uint8)


def load_arrow():
    """Nap mui ten: file bat ky co chu 'arrow' trong ten (arrow_here/arrow_down...).

    Tim trong __data/wings va __data, lay file moi nhat.
    """
    ensure_dirs()
    cands = []
    for d in (WINGS_DIR, DATA_DIR):
        if os.path.isdir(d):
            for f in os.listdir(d):
                if "arrow" in f.lower() and f.lower().endswith((".png", ".webp")):
                    cands.append(os.path.join(d, f))
    if not cands:
        return None
    p = max(cands, key=os.path.getmtime)
    img = _load_rgba(p)
    if img is not None:
        print(f"[i] Dung mui ten: {p}")
    return img


def load_all_wings():
    """Nap TAT CA anh canh trong __data/wings (tru arrow*), sap theo ten.

    Tra list [(ten, bgra)]. Trong thi dung 1 canh mac dinh.
    """
    ensure_dirs()
    files = sorted(f for f in os.listdir(WINGS_DIR)
                   if f.lower().endswith((".png", ".webp")) and "arrow" not in f.lower())
    out = []
    for f in files:
        img = _load_rgba(os.path.join(WINGS_DIR, f))
        if img is not None:
            out.append((os.path.splitext(f)[0], img))
    if not out:
        print("[i] Chua co PNG canh trong __data/wings -> dung canh mac dinh.")
        out.append(("mac dinh", make_default_wing()))
    else:
        print(f"[i] Co {len(out)} canh: {', '.join(n for n, _ in out)}  (phim N de doi)")
    return out


def overlay_bgra(frame, bgra, x, y, alpha=1.0):
    """Dan sprite BGRA vao frame, goc tren-trai (x, y), tu cat vien."""
    bh, bw = bgra.shape[:2]
    x, y = int(x), int(y)
    fh, fw = frame.shape[:2]
    fx1, fy1 = max(0, x), max(0, y)
    fx2, fy2 = min(fw, x + bw), min(fh, y + bh)
    if fx1 >= fx2 or fy1 >= fy2:
        return
    crop = bgra[fy1 - y:fy2 - y, fx1 - x:fx2 - x]
    a = crop[..., 3:4].astype(np.float32) / 255.0 * float(alpha)
    roi = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    frame[fy1:fy2, fx1:fx2] = (roi * (1 - a) + crop[..., :3] * a).astype(np.uint8)


def place_wing(frame, wing_bgra, target, target_w, flip, angle, anchor_ratio):
    """Scale + xoay quanh goc canh + dan sao cho goc canh trung 'target'.

    Tra ve toa do dau canh (tip) de phun hieu ung.
    """
    img = wing_bgra[:, ::-1].copy() if flip else wing_bgra
    h0, w0 = img.shape[:2]
    scale = max(0.05, target_w / w0)
    img = cv2.resize(img, (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                     interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    # dem quanh canh de khi xoay (vay) khong bi cat mep
    pad = int(max(h, w) * 0.5)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                             value=(0, 0, 0, 0))
    hh, ww = img.shape[:2]
    ax_r = (1 - anchor_ratio[0]) if flip else anchor_ratio[0]
    ax, ay = ax_r * w + pad, anchor_ratio[1] * h + pad
    m = cv2.getRotationMatrix2D((ax, ay), angle, 1.0)
    img = cv2.warpAffine(img, m, (ww, hh), flags=cv2.INTER_LINEAR,
                         borderValue=(0, 0, 0, 0))
    tx, ty = target
    overlay_bgra(frame, img, tx - ax, ty - ay)
    # tip = goc doi dien voi anchor (dau canh xoe ra)
    tip_x = pad if not flip else pad + w
    return (tx - ax + tip_x, ty - ay + pad + h * 0.1)


def main():
    ap = argparse.ArgumentParser(description="Gan canh vao nguoi qua camera")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", default=None, help="Duong dan video thay webcam")
    ap.add_argument("--max-poses", type=int, default=3, help="So nguoi toi da")
    ap.add_argument("--scale", type=float, default=2.4,
                    help="Be rong canh = scale * be rong vai")
    ap.add_argument("--front-th", type=float, default=0.5,
                    help="Nguong visibility mui de coi la dang quay mat truoc")
    ap.add_argument("--flap", action="store_true", help="Bat vay canh ngay tu dau")
    ap.add_argument("--effect", action="store_true", help="Bat hieu ung ngay tu dau")
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
        print(f"[!] Khong tai duoc model: {e}\n    Tai thu cong: {MODEL_URL}")
        return

    wings = load_all_wings()
    wing_idx = 0
    arrow = load_arrow()
    anchor_ratio = (0.88, 0.5)   # goc canh trai o canh phai-giua cua anh

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=max(1, args.max_poses),
        output_segmentation_masks=True,   # de canh moc tu SAU lung khi dung mat truoc
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    src = args.source if args.source else args.camera
    use_dshow = os.name == "nt" and isinstance(src, int)
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0] if use_dshow else [0]
    cap = None
    for be in backends:
        cap = cv2.VideoCapture(src, be)
        if cap.isOpened():
            break
        cap.release()
    if cap is None or not cap.isOpened():
        print(f"[!] Khong mo duoc camera/nguon: {src}")
        return

    win = "Angel Wings"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    flap_on, effect_on = args.flap, args.effect
    scale = args.scale
    behind_mode = 0          # 0=AUTO (theo huong mat)  1=luon sau lung  2=luon tren
    behind_names = ["AUTO", "SAU LUNG", "TREN"]
    particles = []
    t0 = time.time()
    frame_i, fails = 0, 0
    fps, prev_t = 0.0, t0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails > 30:
                    print("[!] Camera khong tra ve hinh.")
                    break
                cv2.waitKey(30)
                continue
            fails = 0
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()
            t = now - t0
            frame_i += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_img, frame_i * 33)

            n_people = 0
            wing_name, wing = wings[wing_idx]
            flap = math.sin(t * 6.0) * 18.0 if flap_on else 0.0   # goc dap canh
            orig = frame.copy()          # frame goc (chua ve canh) de dua nguoi len tren
            front_mask = None            # gop mask cua cac nguoi dang quay mat truoc
            heads = []                   # (x, y_dinh_dau, be_rong_vai) de ve mui ten

            if res.pose_landmarks:
                masks = res.segmentation_masks or []
                for idx, lm in enumerate(res.pose_landmarks):
                    ls = lm[L_SHOULDER]
                    rs = lm[R_SHOULDER]
                    lx, ly = ls.x * w, ls.y * h
                    rx, ry = rs.x * w, rs.y * h
                    sh_w = math.hypot(lx - rx, ly - ry)
                    if sh_w < 15:
                        continue
                    n_people += 1
                    wing_w = scale * sh_w
                    # diem gan: hoi vao trong & xuong duoi vai mot chut (ra sau lung)
                    lt = (lx + sh_w * 0.15, ly + sh_w * 0.15)
                    rt = (rx - sh_w * 0.15, ry + sh_w * 0.15)

                    # canh trai (ben vai trai nguoi) va canh phai (lat guong)
                    tip_l = place_wing(frame, wing, lt, wing_w, False,
                                       -flap, anchor_ratio)
                    tip_r = place_wing(frame, wing, rt, wing_w, True,
                                       flap, anchor_ratio)

                    # Quyet dinh co dua nguoi de len tren canh (canh moc tu SAU lung).
                    # AUTO: mat truoc = mui gan camera hon vai (z nho hon) -> canh ra
                    # sau lung; quay lung -> canh ve tren (thay lung, canh o ba vai).
                    if behind_mode == 1:
                        do_behind = True
                    elif behind_mode == 2:
                        do_behind = False
                    else:
                        nose_z = lm[0].z
                        sh_z = (lm[L_SHOULDER].z + lm[R_SHOULDER].z) / 2.0
                        do_behind = nose_z < sh_z          # mui truoc vai = mat truoc
                    if do_behind and idx < len(masks):
                        m = np.asarray(masks[idx].numpy_view())
                        if m.ndim == 3:            # (H,W,1) -> (H,W)
                            m = m[..., 0]
                        if m.shape[:2] != (h, w):
                            m = cv2.resize(m, (w, h))
                        front_mask = m if front_mask is None else np.maximum(front_mask, m)

                    if effect_on and frame_i % 2 == 0:
                        for tip in (tip_l, tip_r):
                            for _ in range(2):
                                ang = random.uniform(0, 2 * math.pi)
                                spd = random.uniform(0.5, 2.5)
                                particles.append({
                                    "x": tip[0] + random.uniform(-8, 8),
                                    "y": tip[1] + random.uniform(-8, 8),
                                    "vx": math.cos(ang) * spd,
                                    "vy": math.sin(ang) * spd - 0.6,
                                    "life": 1.0,
                                    "size": random.randint(2, 5),
                                    "col": random.choice([(255, 255, 255),
                                                          (255, 240, 180),
                                                          (255, 200, 255),
                                                          (200, 240, 255)]),
                                })

                    # dinh dau (tren vai) de ve mui ten chi vao
                    heads.append(((lx + rx) / 2.0, min(ly, ry) - sh_w * 0.6, sh_w))
            else:
                cv2.putText(frame, "Dung truoc camera de moc canh...",
                            (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 220, 255), 2)

            # Cho nguoi quay mat truoc: dua vung nguoi (orig) de len tren canh
            # -> canh nhu moc tu sau lung.
            if front_mask is not None:
                m3 = (front_mask > 0.5).astype(np.float32)[..., None]
                frame[:] = (orig * m3 + frame * (1 - m3)).astype(np.uint8)

            # cap nhat + ve hat lap lanh (ve tren cung)
            alive = []
            for p in particles:
                p["x"] += p["vx"]
                p["y"] += p["vy"]
                p["vy"] += 0.03
                p["life"] -= 0.03
                if p["life"] > 0:
                    c = tuple(int(v * p["life"]) for v in p["col"])
                    cv2.circle(frame, (int(p["x"]), int(p["y"])),
                               max(1, int(p["size"] * p["life"])), c, -1, cv2.LINE_AA)
                    alive.append(p)
            particles = alive[-500:]

            # mui ten (arrow*.png) chi vao dau, nhun len-xuong
            if arrow is not None:
                bob = (math.sin(t * 5.0) * 0.5 + 0.5)      # 0..1
                for hx, hy_top, sh_w in heads:
                    aw = max(1, int(sh_w * 0.9))
                    ah = max(1, int(aw * arrow.shape[0] / arrow.shape[1]))
                    a_img = cv2.resize(arrow, (aw, ah), interpolation=cv2.INTER_LINEAR)
                    gap = sh_w * 0.2 + bob * sh_w * 0.3     # khoang cach dau -> mui ten
                    tip_y = hy_top - gap                    # day mui ten (mui chi xuong)
                    overlay_bgra(frame, a_img, hx - aw / 2.0, tip_y - ah)

            now2 = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now2 - prev_t, 1e-6))
            prev_t = now2
            cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.putText(frame, f"FPS:{fps:4.1f} nguoi:{n_people} "
                        f"vay(F):{'BAT' if flap_on else 'TAT'} "
                        f"hieu ung(E):{'BAT' if effect_on else 'TAT'} "
                        f"kieu(B):{behind_names[behind_mode]} "
                        f"canh(N):{wing_name} +/-:co Q:thoat",
                        (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 180), 1)

            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            elif k == ord("f"):
                flap_on = not flap_on
            elif k == ord("e"):
                effect_on = not effect_on
            elif k == ord("b"):
                behind_mode = (behind_mode + 1) % 3
            elif k == ord("n"):
                wing_idx = (wing_idx + 1) % len(wings)
                print(f"[i] Doi canh -> {wings[wing_idx][0]}")
            elif k in (ord("+"), ord("=")):
                scale = min(4.5, scale + 0.2)
            elif k in (ord("-"), ord("_")):
                scale = max(1.0, scale - 0.2)
    except Exception as e:  # noqa
        print(f"[!] Loi: {e}")
        traceback.print_exc()
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
