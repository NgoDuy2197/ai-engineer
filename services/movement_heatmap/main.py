"""
Service: Movement Heatmap (ban do nhiet di chuyen / dwell heatmap).

Y tuong:
- Theo doi camera bang YOLO tracking (chi class 'person').
- Moi frame, cong "nhiet" vao vi tri chan cua tung nguoi -> dung cang lau /
  di qua cang nhieu lan thi vung do cong don cang nhieu -> mau cang dam (do).
- Lam mo (Gaussian blur) de tao vet muot, roi phu colormap len khung hinh.
- Nho vay thay ro: nguoi di chuyen cho nao, dung dau lau.

Toi uu Intel UHD 620 / Iris Xe qua OpenVINO; fallback CPU chung (case default).

Phim tat:
  H : bat/tat lop heatmap
  B : bat/tat khung + id nguoi
  S : luu anh heatmap hien tai vao data/
  C : reset (xoa het nhiet)
  Q / ESC : thoat
"""
import argparse
import os
import time

import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LATEST_PATH = os.path.join(DATA_DIR, "heatmap_latest.png")


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


def colorize(heat, gamma, max_alpha, colormap):
    """Chuyen buffer nhiet (float) thanh lop mau + alpha theo tung pixel.

    - Chuan hoa theo percentile 99.5 de 1 diem qua nong khong lam trang het.
    - gamma < 1 lam noi bat vung nhiet thap (contrast).
    - alpha ti le voi do nong -> vung nguoi khong dung se trong suot.
    """
    ref = np.percentile(heat, 99.5) if np.any(heat > 0) else 0.0
    if ref <= 1e-6:
        return None, None
    norm = np.clip(heat / ref, 0.0, 1.0) ** gamma
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
    alpha = (norm * max_alpha)[..., None]  # (h, w, 1)
    return color, alpha


def main():
    ap = argparse.ArgumentParser(description="Ban do nhiet di chuyen cua nguoi")
    ap.add_argument("--source", default="0", help="0=webcam hoac duong dan video")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "intel:cpu", "intel:gpu", "cpu"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--gain", type=float, default=0.6,
                    help="Nhiet cong them moi frame cho 1 nguoi (cang cao -> len mau cang nhanh)")
    ap.add_argument("--radius", type=int, default=26,
                    help="Ban kinh vung nhiet quanh chan nguoi (pixel)")
    ap.add_argument("--decay", type=float, default=1.0,
                    help="He so mo dan moi frame (1.0=cong don mai; 0.99=heatmap 'song')")
    ap.add_argument("--gamma", type=float, default=0.6, help="Do tuong phan mau")
    ap.add_argument("--alpha", type=float, default=0.75, help="Do dam toi da cua lop mau (0..1)")
    ap.add_argument("--full-body", action="store_true",
                    help="Tinh nhiet ca vung than nguoi thay vi chi diem chan")
    ap.add_argument("--reexport", action="store_true",
                    help="Xuat lai model OpenVINO (khi doi imgsz)")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    model, dev = load_model(args.weights, args.device, args.imgsz, args.reexport)

    src = int(args.source) if args.source.isdigit() else args.source
    use_dshow = os.name == "nt" and isinstance(src, int)
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if use_dshow else 0)
    if not cap.isOpened():
        print(f"[!] Khong mo duoc nguon: {args.source}")
        return

    win = "Movement Heatmap"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    heat = None
    seen_ids = set()
    show_heat = True
    show_boxes = True
    fps = 0.0
    prev_t = time.time()
    last_write = 0.0
    colormap = cv2.COLORMAP_TURBO

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if heat is None:
            heat = np.zeros((h, w), np.float32)

        results = model.track(frame, persist=True, classes=[0], conf=args.conf,
                              imgsz=args.imgsz, device=dev, verbose=False,
                              tracker="bytetrack.yaml")
        r = results[0]
        current = 0
        now = time.time()

        # ==== cong nhiet vao vi tri nguoi ====
        add = np.zeros((h, w), np.float32)
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            xyxy = r.boxes.xyxy.cpu().numpy()
            current = len(ids)
            for tid, box in zip(ids, xyxy):
                x1, y1, x2, y2 = box
                seen_ids.add(tid)
                if args.full_body:
                    cv2.rectangle(add, (int(x1), int(y1)), (int(x2), int(y2)),
                                  args.gain, -1)
                else:
                    cx = int((x1 + x2) / 2.0)
                    cy = int(y2)  # diem chan
                    cv2.circle(add, (cx, cy), args.radius, args.gain, -1)
                if show_boxes:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 200, 0), 2)
                    cv2.putText(frame, f"#{tid}", (int(x1), int(y1) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

        # lam mem vung nhiet vua them (bôi mờ) roi cong don co mo dan
        if np.any(add > 0):
            add = cv2.GaussianBlur(add, (0, 0), sigmaX=args.radius / 2.0)
        heat = heat * args.decay + add

        # ==== phu lop mau ====
        if show_heat:
            color, alpha = colorize(heat, args.gamma, args.alpha, colormap)
            if color is not None:
                frame = (frame.astype(np.float32) * (1 - alpha)
                         + color.astype(np.float32) * alpha).astype(np.uint8)

        # ==== panel thong tin ====
        now2 = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now2 - prev_t, 1e-6))
        prev_t = now2
        panel = [
            f"Dang trong khung: {current}   |   Tong nguoi thay: {len(seen_ids)}",
            f"Nhiet cao nhat: {heat.max():7.1f}   Che do: "
            + ("song (mo dan)" if args.decay < 1.0 else "cong don"),
            f"FPS: {fps:4.1f} [{dev}]  |  H:heatmap  B:khung  S:luu  C:reset  Q:thoat",
        ]
        cv2.rectangle(frame, (0, 0), (w, 92), (0, 0, 0), -1)
        for i, txt in enumerate(panel):
            col = (0, 255, 180) if i < 2 else (180, 180, 180)
            cv2.putText(frame, txt, (12, 26 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        cv2.imshow(win, frame)

        # tu dong luu anh heatmap moi 2 giay de xem lai
        if now - last_write > 2.0:
            cv2.imwrite(LATEST_PATH, frame)
            last_write = now

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        elif k == ord("h"):
            show_heat = not show_heat
        elif k == ord("b"):
            show_boxes = not show_boxes
        elif k == ord("c"):
            heat[:] = 0.0
            seen_ids.clear()
        elif k == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
            path = os.path.join(DATA_DIR, f"heatmap_{ts}.png")
            cv2.imwrite(path, frame)
            print(f"[i] Da luu: {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
