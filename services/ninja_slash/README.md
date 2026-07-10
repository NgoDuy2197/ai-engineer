# Ninja Slash 🥷

Game chém hoa quả bay bằng đầu ngón tay (MediaPipe Hands, chạy CPU).

## Cách chơi
- Giơ tay trước camera, **vung nhanh đầu ngón trỏ** để chém hoa quả bay lên.
- Chém liên tiếp nhiều quả → **COMBO** nhân điểm.
- ⚫ **Bom** (màu đen, có ngòi lửa): chém trúng là **thua ngay** — tránh ra!
- Để hoa quả rơi khỏi màn hình (không chém) → mất 1 mạng. Hết 3 mạng → thua.

## Phím tắt
- `C` : chơi lại
- `Q` / `ESC` : thoát

## Model
Cần file `models/hand_landmarker.task`. Nếu thiếu, tải tại:
`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
