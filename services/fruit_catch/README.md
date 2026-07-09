# Fruit Catch – Game hứng hoa quả bằng khuôn mặt

Điều khiển cái rổ ở đáy màn hình bằng **khuôn mặt** (di chuyển đầu sang trái/phải),
hứng hoa quả emoji rơi lộn từ trên xuống.

## Luật chơi

- MediaPipe Face Detection định vị mặt → rổ chạy theo vị trí mặt.
- Hoa quả (🍎🍊🍌🍇🍓…) rơi + xoay lộn. **Hứng trúng → +1 điểm** + tiếng "tinh" nhẹ.
- Cứ đủ **10 quả → bắn pháo hoa** đẹp ~3 giây. 🎆

## Phím tắt

| Phím | Chức năng |
|------|-----------|
| `C` | Chơi lại (reset điểm) |
| `Q` / `ESC` | Thoát |

## Chạy

```bat
__bat\setup_fruit_catch.bat   :: cài 1 lần
__bat\start_fruit_catch.bat   :: chạy

:: Quả to hơn / rơi nhanh hơn
python services\fruit_catch\main.py --size 90 --speed 1.4
```

## Ghi chú

- Emoji vẽ bằng **Pillow + font Segoe UI Emoji** (`C:\Windows\Fonts\seguiemj.ttf`).
  Máy không có font emoji màu → tự động vẽ quả bằng **hình tròn màu** (vẫn chơi được).
- Tiếng "tinh" khi hứng dùng `winsound` (chỉ kêu trên Windows; HĐH khác chơi im lặng).
- Chạy trên CPU — nhẹ cho Intel UHD 620 / Iris Xe.
