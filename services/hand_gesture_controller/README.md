# Hand Gesture Controller

Điều khiển ảnh bằng cử chỉ tay (dùng **MediaPipe Hands**, chạy trên CPU).

## Chạy

```bat
REM Cài đặt 1 lần
__bat\setup_hand_gesture.bat

REM Bật service
__bat\start_hand_gesture.bat
```

## Cử chỉ

| Cử chỉ | Hành động |
|---|---|
| Quẹt trái | Ảnh trước |
| Quẹt phải | Ảnh sau |
| Giơ 5 ngón tay (giữ ~0.8s) | Vào chế độ **XOAY** |
| Nghiêng bàn tay (trong chế độ xoay) | Xoay ảnh theo tay |
| Nắm tay (giữ ~0.8s) | Thoát chế độ xoay |
| `Q` / `ESC` | Thoát |

## Ảnh

Copy ảnh (`.jpg .jpeg .png .bmp .webp`) vào thư mục [images/](images/) cạnh file này.
Ảnh được sắp theo thứ tự tên file.

## Tùy chọn

```bat
python services\hand_gesture_controller\main.py --camera 1 --width 1280 --height 800
```
