"""
Service: Smart Security Cam (camera an ninh phat hien nguoi).

- YOLO phat hien nguoi trong khung hinh.
- Khi CO nguoi: khung do + trang thai "CANH BAO", keu beep (Windows),
  va tu dong chup anh bang chung vao data/ (gian cach --cooldown giay).
- Khi KHONG co ai: trang thai "AN TOAN" (xanh).
- Ghi log su kien vao data/events.log.

Toi uu Intel UHD 620 / Iris Xe qua OpenVINO; fallback CPU chung (case default).

Phim tat:
  M : bat/tat tieng beep
  S : chup anh ngay
  C : xoa bo dem su kien
  Q / ESC : thoat
"""
import argparse
import os
import time

import cv2

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LOG_PATH = os.path.join(DATA_DIR, "events.log")


def load_model(weights, device, imgsz, reexport):
    """Nap YOLO, uu tien OpenVINO cho Intel, fallback PyTorch CPU."""
    from ultralytics import YOLO

    if device in ("auto", "intel:cpu", "intel:gpu"):
        try:
            import openvino  # noqa: F401
            ov_dir = os.path.splitext(weights)[0] + "_openvino_model"
            if reexport and os.path.isdir(ov_dir):
                import shutil
                shutil.rmtree(ov_dir)
            if not os.path.isdir(ov_dir):
                print("[i] Xuat model sang OpenVINO (chi lam 1 lan)...")
                YOLO(weights).export(format="openvino", imgsz=imgsz)
            model = YOLO(ov_dir, task="detect")
            ov_device = "intel:gpu" if device == "intel:gpu" else "intel:cpu"
            print(f"[i] Backend: OpenVINO ({ov_device})")
            return model, ov_device
        except Exception as e:  # noqa
            print(f"[!] OpenVINO khong dung duoc: {e} -> Fallback CPU thuong.")
    print("[i] Backend: PyTorch CPU")
    return YOLO(weights), "cpu"


def beep():
    """Beep ngan; chi tren Windows, loi thi bo qua."""
    try:
        import winsound
        winsound.Beep(880, 150)
    except Exception:  # noqa
        pass


def log_event(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "  " + msg
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print("[event]", line)


def main():
    ap = argparse.ArgumentParser(description="Camera an ninh phat hien nguoi")
    ap.add_argument("--source", default="0", help="0=webcam hoac duong dan video")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "intel:cpu", "intel:gpu", "cpu"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--cooldown", type=float, default=3.0,
                    help="Giay toi thieu giua 2 lan tu dong chup anh")
    ap.add_argument("--reexport", action="store_true")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    model, dev = load_model(args.weights, args.device, args.imgsz, args.reexport)

    src = int(args.source) if args.source.isdigit() else args.source
    use_dshow = os.name == "nt" and isinstance(src, int)
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if use_dshow else 0)
    if not cap.isOpened():
        print(f"[!] Khong mo duoc nguon: {args.source}")
        return

    win = "Smart Security Cam"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    sound_on = True
    events = 0
    was_present = False
    last_shot = 0.0
    fps = 0.0
    prev_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        now = time.time()

        results = model(frame, classes=[0], conf=args.conf, imgsz=args.imgsz,
                        device=dev, verbose=False)
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        n = len(boxes)
        present = n > 0

        color = (0, 0, 255) if present else (0, 200, 0)
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # canh do quanh khung khi canh bao
        if present:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

        # su kien: canh len (khong ai -> co nguoi)
        if present and not was_present:
            events += 1
            log_event(f"CANH BAO: phat hien {n} nguoi")
            if sound_on:
                beep()
        was_present = present

        # tu dong chup bang chung khi dang co nguoi
        if present and now - last_shot > args.cooldown:
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
            path = os.path.join(DATA_DIR, f"alert_{ts}.jpg")
            cv2.imwrite(path, frame)
            last_shot = now

        now2 = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now2 - prev_t, 1e-6))
        prev_t = now2

        status = f"CANH BAO ({n} nguoi)" if present else "AN TOAN"
        cv2.rectangle(frame, (0, 0), (w, 96), (0, 0, 0), -1)
        cv2.putText(frame, status, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"So su kien: {events}   Tieng: {'BAT' if sound_on else 'TAT'}",
                    (14, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)
        cv2.putText(frame, f"FPS: {fps:4.1f} [{dev}]  |  M:tieng  S:chup  C:reset  Q:thoat",
                    (14, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        elif k == ord("m"):
            sound_on = not sound_on
        elif k == ord("c"):
            events = 0
        elif k == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
            path = os.path.join(DATA_DIR, f"manual_{ts}.jpg")
            cv2.imwrite(path, frame)
            print(f"[i] Da luu: {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
