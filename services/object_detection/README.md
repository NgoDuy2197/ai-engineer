# Object Detection

Phát hiện vật thể qua camera và gán nhãn (người, mèo, chó, xe, ...) bằng **YOLOv8** + **OpenVINO**.

## Chạy

```bat
REM Cài đặt 1 lần
__bat\setup_object_detection.bat

REM Bật service
__bat\start_object_detection.bat
```

Hoặc chạy tay (sau khi đã kích hoạt venv):

```bat
python services\object_detection\main.py
```

## Tham số hữu ích

| Tham số | Ý nghĩa | Ví dụ |
|---|---|---|
| `--device` | `auto` (OpenVINO CPU), `intel:gpu` (iGPU Intel), `cpu` (PyTorch) | `--device intel:gpu` |
| `--imgsz` | Kích thước đầu vào (nhỏ hơn = nhanh hơn) | `--imgsz 480 --reexport` |
| `--source` | `0` = webcam, hoặc file video/ảnh | `--source clip.mp4` |
| `--conf` | Ngưỡng tin cậy | `--conf 0.5` |
| `--weights` | Model khác (s/m nặng hơn, chính xác hơn) | `--weights yolov8s.pt` |

> Lần đầu chạy sẽ tự tải `yolov8n.pt` và export sang OpenVINO (mất chút thời gian).
> Đổi `--imgsz` thì cần thêm `--reexport` để xuất lại model.

**Phím tắt:** `Q` hoặc `ESC` để thoát.
