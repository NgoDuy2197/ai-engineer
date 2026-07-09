# Fitness Coach – Đếm số lần tập

Dùng camera + MediaPipe Pose để đo góc khớp và **đếm số lần tập** tự động.

## Bài tập hỗ trợ

- `squat` (mặc định) — đo góc **đầu gối** (hông–gối–cổ chân).
- `curl` — đo góc **khuỷu tay** (vai–khuỷu–cổ tay), đếm động tác gập tay.

Cách đếm: khi góc khớp đi từ "duỗi thẳng" → "gập sâu" thì tính **1 lần**. Có thanh
phần trăm độ sâu để canh động tác cho chuẩn.

## Phím tắt

| Phím | Chức năng |
|------|-----------|
| `E` | Đổi bài tập (squat ↔ curl) |
| `C` | Reset bộ đếm |
| `Q` / `ESC` | Thoát |

## Chạy

```bat
__bat\setup_fitness_coach.bat   :: cài 1 lần
__bat\start_fitness_coach.bat   :: chạy (mặc định squat)

:: Chọn bài tập gập tay
python services\fitness_coach\main.py --exercise curl
```

> Đứng/ngồi sao cho camera thấy rõ toàn thân (hoặc cả cánh tay với bài `curl`).
