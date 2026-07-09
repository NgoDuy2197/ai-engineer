# AI Engineer – Bộ sưu tập Services

Mỗi thư mục trong `services/` là **một ứng dụng độc lập** (có môi trường ảo riêng).
Bạn bật service nào thì bật service đó — không cần chạy tất cả.

## Cấu trúc

```
ai-engineer/
├─ run_dashboard.bat          # Bật server + mở trang điều hướng (http://localhost:8000)
├─ web/                       # Trang dashboard điều hướng vào các service
│  ├─ index.html
│  └─ server.py
├─ __bat/                     # Các file .bat để cài đặt / bật từng service riêng lẻ
│  ├─ setup_object_detection.bat
│  ├─ start_object_detection.bat
│  ├─ setup_hand_gesture.bat
│  ├─ start_hand_gesture.bat
│  └─ start_dashboard.bat
└─ services/
   ├─ object_detection/       # Phát hiện vật thể qua camera + gán nhãn (người, mèo, ...)
   ├─ hand_gesture_controller/# Điều khiển ảnh bằng cử chỉ tay
   ├─ people_counter/         # Đếm người + đếm vào/ra qua vạch + dashboard theo phút
   ├─ movement_heatmap/       # Bản đồ nhiệt: người đi/đứng lâu chỗ nào thì màu càng đậm
   ├─ air_painter/            # Vẽ trong không khí bằng ngón tay (MediaPipe Hands)
   ├─ fitness_coach/          # Đếm squat / gập tay qua góc khớp (MediaPipe Pose)
   └─ security_cam/           # Camera an ninh: có người thì báo động + chụp ảnh + log
```

## Yêu cầu

- **Windows** + **Python 3.10 hoặc 3.11** (khuyến nghị — MediaPipe chưa hỗ trợ mọi bản mới nhất).
- Kiểm tra: `python --version` (Python phải nằm trong PATH).
- Webcam.

## Cách chạy nhanh

### Cách 1 — Dùng Dashboard (khuyến nghị)

1. Cài đặt **một lần** cho service muốn dùng (double‑click):
   - `__bat\setup_object_detection.bat`
   - `__bat\setup_hand_gesture.bat`
2. Chạy `run_dashboard.bat` → trình duyệt mở trang điều hướng.
3. Bấm nút **Khởi chạy** của service tương ứng (mỗi service mở trong cửa sổ riêng).

### Cách 2 — Chạy trực tiếp bằng .bat

- Bật phát hiện vật thể: `__bat\start_object_detection.bat`
- Bật điều khiển cử chỉ: `__bat\start_hand_gesture.bat`
- Bật đếm người: `__bat\start_people_counter.bat` (xem dashboard tại `http://localhost:8000/counter.html`)
- Bật bản đồ nhiệt: `__bat\start_movement_heatmap.bat`
- Bật vẽ trong không khí: `__bat\start_air_painter.bat`
- Bật đếm số lần tập: `__bat\start_fitness_coach.bat`
- Bật camera an ninh: `__bat\start_security_cam.bat`

> Lần đầu chạy `start_*.bat`, nếu chưa setup thì nó sẽ tự chạy setup trước.

## Tối ưu cho Intel UHD 620 / Iris Xe

- **Object Detection** dùng **OpenVINO** (tối ưu cho phần cứng Intel). Mặc định `--device auto`
  sẽ chạy OpenVINO trên CPU Intel; nếu máy khác/không có OpenVINO sẽ tự **fallback về PyTorch CPU**
  (case default chung).
  - Thử chạy trên iGPU Intel: thêm `--device intel:gpu`
    (ví dụ sửa `start_object_detection.bat` hoặc chạy tay: `python services\object_detection\main.py --device intel:gpu`).
  - Muốn nhẹ hơn nữa: giảm `--imgsz 480`.
- **Hand Gesture** dùng **MediaPipe** với `model_complexity=0`, chạy tốt trên CPU (không cần GPU rời).

> [Unverified] Các con số hiệu năng phụ thuộc driver, độ phân giải và tải máy; hãy đo thực tế trên máy bạn.

## Ảnh cho Hand Gesture

Copy ảnh (`.jpg/.png/...`) vào `services/hand_gesture_controller/images/`.
Theo mặc định các file ảnh **không được commit** (đã ignore trong `.gitignore`).

> Lưu ý: mã nguồn ở đây **chưa được chạy thử trong môi trường này** — hãy chạy setup rồi test trên máy bạn.
