# Smart Security Cam – Camera an ninh phát hiện người

Dùng camera + YOLO để canh chừng: có người xuất hiện thì báo động, chụp ảnh bằng
chứng và ghi log — hợp làm camera trông nhà / trông phòng đơn giản.

## Hoạt động

- **Có người** → khung đỏ + viền đỏ quanh màn hình, trạng thái `CANH BAO`, kêu
  beep (Windows), tự chụp ảnh vào `data/alert_*.jpg`.
- **Không có ai** → trạng thái `AN TOAN` (xanh).
- Mỗi lần chuyển từ "không ai" → "có người" ghi 1 dòng vào `data/events.log`.

## Phím tắt

| Phím | Chức năng |
|------|-----------|
| `M` | Bật/tắt tiếng beep |
| `S` | Chụp ảnh ngay |
| `C` | Xóa bộ đếm sự kiện |
| `Q` / `ESC` | Thoát |

## Chạy

```bat
__bat\setup_security_cam.bat   :: cài 1 lần
__bat\start_security_cam.bat   :: chạy

:: Giãn cách tự chụp ảnh (giây) & dùng video thay webcam
python services\security_cam\main.py --cooldown 5 --source path\to\video.mp4
```

> Tối ưu Intel qua OpenVINO (`--device auto`), tự fallback CPU nếu không có.
> [Unverified] Hiệu năng tùy máy — hãy đo thực tế.
