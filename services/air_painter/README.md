# Air Painter – Vẽ trong không khí

Dùng camera + MediaPipe Hands để vẽ bằng đầu ngón tay, không cần chạm màn hình.

## Cử chỉ

- **1 ngón trỏ** → vẽ theo đầu ngón.
- **2 ngón (trỏ + giữa)** → chế độ chọn (nhấc bút); đưa lên thanh màu trên cùng để đổi màu/công cụ.
- **Nắm tay** → nghỉ.

Ô `TAY` trên thanh màu = cục tẩy (nét to, xóa).

## Phím tắt

| Phím | Chức năng |
|------|-----------|
| `C` | Xóa toàn bộ tranh |
| `S` | Lưu tranh vào `data/` |
| `+` / `-` | Tăng / giảm cỡ bút |
| `Q` / `ESC` | Thoát |

## Chạy

```bat
__bat\setup_air_painter.bat   :: cài 1 lần
__bat\start_air_painter.bat   :: chạy
```

Chạy trên CPU (MediaPipe `model_complexity=0`) — nhẹ cho Intel UHD 620 / Iris Xe.
