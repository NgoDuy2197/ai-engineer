"""
Service camera nhan dien khuon mat truc tiep.

Dua vao db.pkl (da train), voi moi khung hinh:
  - Phat hien tat ca khuon mat.
  - Trich vector va so khop -> ten nguoi (hoac "Khong ro").
  - Ve khung + nhan ten (co dau tieng Viet) len tung nguoi.

Phim tat:
  Q / ESC : thoat
  + / -   : tang / giam nguong nhan dien
"""
import os
import time
import traceback

import cv2

from draw import put_text_unicode
from recognizer import (COSINE_THRESHOLD, DB_PATH, FaceEngine, identify,
                        load_db, open_camera)


def run(source="0", device="auto", threshold=COSINE_THRESHOLD, det_size=320):
    if not os.path.exists(DB_PATH):
        print("[!] Chua co db.pkl. Hay chay 'python main.py train' truoc.")
        return
    db = load_db()
    if not db:
        print("[!] db.pkl rong. Hay enroll/label roi train lai.")
        return

    engine = FaceEngine(device=device, input_size=(det_size, det_size))

    cap = open_camera(source)
    if cap is None:
        print(f"[!] Khong mo duoc camera/nguon: {source}")
        print("    - Dong cac app dang dung webcam (Zoom/Teams/Camera) roi thu lai.")
        return

    win = "Face Recognition Camera"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    fps, prev_t = 0.0, time.time()
    fails = 0

    print(f"[i] Da nap {len(db)} nguoi: {', '.join(db.keys())}")

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

            faces = engine.detect(frame)
            for row in faces:
                x, y, w, h = row[:4].astype(int)
                emb = engine.embed(frame, row)
                name, score = identify(emb, db, threshold)

                known = name != "Khong ro"
                color = (0, 200, 0) if known else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{name} ({score:.2f})"
                frame = put_text_unicode(frame, label, (x, max(0, y - 28)),
                                         size=22, color=(255, 255, 255), bg=color)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now
            hud = f"FPS:{fps:4.1f} [{engine.device}]  nguong:{threshold:.2f}  +/-:chinh  Q:thoat"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(frame, hud, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (180, 220, 180), 1)

            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            elif k in (ord("+"), ord("=")):
                threshold = min(1.0, threshold + 0.02)
            elif k in (ord("-"), ord("_")):
                threshold = max(0.0, threshold - 0.02)
    except Exception as e:  # noqa
        print(f"[!] Loi trong qua trinh nhan dien: {e}")
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
