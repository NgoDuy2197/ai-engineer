"""
Thu thap anh khuon mat tu webcam de lam du lieu train (tien loi hon copy tay).

Chay:  python main.py enroll --name "Nguyen Van A"
  SPACE : chup 1 anh (chi luu khi trong khung co dung 1 mat)
  A     : bat/tat che do tu dong chup (moi --interval giay 1 tam)
  Q/ESC : thoat

Anh duoc luu vao __data/<ten>/ , sau do chay 'python main.py train'.
"""
import os
import time
import traceback

import cv2

from recognizer import (DATA_DIR, FaceEngine, ensure_dirs, imwrite_unicode,
                        open_camera)


def run(name, source="0", device="auto", target=20, interval=0.4):
    name = name.strip()
    if not name:
        print("[!] Thieu ten nguoi (--name).")
        return
    ensure_dirs()
    person_dir = os.path.join(DATA_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    print(f"[i] Thu mat cho: {name}")
    print(f"    Anh se luu vao: {person_dir}")

    try:
        engine = FaceEngine(device=device)
    except Exception as e:  # noqa
        print(f"[!] Loi khoi tao engine nhan dien: {e}")
        traceback.print_exc()
        return

    cap = open_camera(source)
    if cap is None:
        print(f"[!] Khong mo duoc camera/nguon: {source}")
        print("    - Dong cac app dang dung webcam (Zoom/Teams/Camera) roi thu lai.")
        return

    # Cua so dung ten ASCII co dinh (ten co dau ve len khung bang putText ben duoi)
    win = "Enroll - Thu mat"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    count = len([f for f in os.listdir(person_dir) if f.lower().endswith(".jpg")])
    auto, last = False, 0.0
    fails = 0

    def save(frame):
        nonlocal count
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(person_dir, f"{ts}_{count:03d}.jpg")
        if imwrite_unicode(path, frame):
            count += 1
            print(f"[i] Da luu {count} anh -> {path}")
        else:
            print(f"[!] Khong ghi duoc anh: {path}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # mot so webcam tra ve loi o vai frame dau -> thu lai vai lan
                fails += 1
                if fails > 30:
                    print("[!] Camera khong tra ve hinh (thu lai 30 lan that bai).")
                    break
                cv2.waitKey(30)
                continue
            fails = 0

            clean = frame.copy()
            faces = engine.detect(frame)
            n = len(faces)
            for row in faces:
                x, y, fw, fh = row[:4].astype(int)
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 200, 0), 2)

            now = time.time()
            if auto and n == 1 and now - last > interval and count < target:
                save(clean)
                last = now

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
            hud = (f"da luu:{count}/{target} | mat:{n} | "
                   f"tu dong:{'BAT' if auto else 'TAT'} | SPACE:chup  A:auto  Q:thoat")
            cv2.putText(frame, hud, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 180), 1)
            cv2.imshow(win, frame)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            elif k == ord("a"):
                auto = not auto
            elif k == 32:  # SPACE
                if n == 1:
                    save(clean)
                elif n == 0:
                    print("[!] Chua thay mat nao trong khung.")
                else:
                    print(f"[!] Can dung 1 mat trong khung (dang co {n}).")
    except Exception as e:  # noqa
        print(f"[!] Loi trong qua trinh thu mat: {e}")
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[OK] Tong {count} anh cho '{name}'. Buoc tiep: python main.py train")
