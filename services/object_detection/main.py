"""
Service: Object Detection (phat hien + gan nhan vat the qua camera).

Toi uu cho Intel UHD 620 / Iris Xe qua OpenVINO; co fallback CPU chung (case default).

Vi du:
  python main.py                      # auto: OpenVINO tren CPU Intel (mac dinh)
  python main.py --device intel:gpu   # thu chay tren iGPU Intel (UHD 620 / Iris Xe)
  python main.py --device cpu         # PyTorch CPU thuong (may bat ky)
  python main.py --source video.mp4   # doc tu file thay vi webcam
  python main.py --imgsz 480          # nhe hon, nhanh hon (can --reexport neu doi kich thuoc)

Phim tat: Q hoac ESC de thoat.
"""
import argparse
import os
import time

import cv2


def load_model(weights, device, imgsz, reexport):
    """Tra ve (model, device_string) theo backend phu hop."""
    from ultralytics import YOLO

    want_openvino = device in ("auto", "intel:cpu", "intel:gpu")
    if want_openvino:
        try:
            import openvino  # noqa: F401  (chi kiem tra co cai hay khong)

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
            print(f"[!] OpenVINO khong dung duoc: {e}")
            print("    -> Fallback sang PyTorch CPU (case default chung).")

    print("[i] Backend: PyTorch CPU")
    return YOLO(weights), "cpu"


def main():
    ap = argparse.ArgumentParser(description="Phat hien vat the qua camera + gan nhan")
    ap.add_argument("--source", default="0", help="0=webcam, hoac duong dan video/anh")
    ap.add_argument("--weights", default="yolov8n.pt", help="Model YOLO (mac dinh nano)")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "intel:cpu", "intel:gpu", "cpu"])
    ap.add_argument("--imgsz", type=int, default=640, help="Kich thuoc anh dau vao")
    ap.add_argument("--conf", type=float, default=0.35, help="Nguong tin cay")
    ap.add_argument("--reexport", action="store_true", help="Xuat lai model OpenVINO")
    args = ap.parse_args()

    model, dev = load_model(args.weights, args.device, args.imgsz, args.reexport)

    src = int(args.source) if args.source.isdigit() else args.source
    use_dshow = os.name == "nt" and isinstance(src, int)
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if use_dshow else 0)
    if not cap.isOpened():
        print(f"[!] Khong mo duoc nguon: {args.source}")
        return

    win = "Object Detection - Q/ESC de thoat"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, device=dev, imgsz=args.imgsz,
                                conf=args.conf, verbose=False)
        annotated = results[0].plot()  # ve box + nhan (person, cat, ...)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now
        cv2.putText(annotated, f"FPS: {fps:4.1f}  [{dev}]", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(win, annotated)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
