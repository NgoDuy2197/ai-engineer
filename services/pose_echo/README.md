# Pose Echo — Bóng Que Trễ 👻

Ghi lại chuyển động của bạn rồi vẽ **người que trễ ~3 giây** — như có một cái bóng
làm lại đúng động tác bạn vừa làm 3 giây trước.

## Cách hoạt động
- Camera liên tục ghi lại tư thế (pose) của bạn vào bộ nhớ đệm.
- Trong ~3 giây đầu: hiện đếm ngược *"Đang ghi... bóng hiện sau X.Xs"*.
- Sau đó: **người que xanh** xuất hiện và lặp lại đúng chuyển động của bạn 3 giây trước,
  chạy sau bạn đúng khoảng trễ đó → thử vẫy tay rồi xem cái bóng vẫy theo sau!

## Phím tắt
- `+` / `-` : tăng / giảm độ trễ (0.5s – 6s)
- `L` : bật/tắt khung xương hiện tại (màu xám mờ để tham chiếu)
- `C` : xóa bộ nhớ, ghi lại từ đầu
- `Q` / `ESC` : thoát

## Model
Cần file `models/pose_landmarker_lite.task`. Nếu thiếu, tải tại:
`https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`
