# Angel Wings

Game camera: bắt người (1 hoặc nhiều người) rồi gắn đôi cánh vào lưng họ.
Có chế độ vẫy cánh và chế độ cánh phát hiệu ứng lấp lánh.

## Cách dùng ảnh của bạn (bỏ vào `__data/wings/`)
- **Cánh**: các file PNG **cánh trái** (nền trong suốt). Cánh phải = tự lật gương.
  Thư mục trống thì game tự vẽ cánh mặc định. Nhiều PNG → bấm **`N`** để đổi cánh
  lần lượt **theo tên** (a→b→c...). Tất cả PNG (trừ `arrow*`) đều được coi là cánh.
- **Mũi tên**: đặt file có chữ **`arrow`** trong tên (vd `arrow_here.png`,
  `arrow_down.png`), nền trong. Nếu có, game gắn mũi tên chỉ vào đầu từng người,
  cách đầu 1 đoạn và **nhún lên xuống**. (File `arrow*` không bị nhầm thành cánh.)
- Trên Dashboard có nút **📂 Mở thư mục cánh** để mở nhanh nơi thả file.

## Cánh mọc từ sau lưng
- **Đứng mặt trước**: game đưa người đè lên cánh (segmentation) → cánh **mọc từ sau lưng**.
- **Quay lưng**: giữ cánh vẽ trên (thấy lưng nên cánh nằm ở bả vai, không bị khuất).
- Phân biệt trước/sau bằng **độ sâu z của mũi so với vai** (tự động).
- Nếu auto đoán sai, bấm **`B`** để đổi: `AUTO` → `SAU LƯNG` (luôn) → `TRÊN` (luôn hiện cánh).

> Quy ước hướng ảnh: cánh trái nên vẽ với **gốc cánh nằm ở cạnh PHẢI-giữa** của
> ảnh (chỗ gắn vào vai), phần cánh xòe sang trái. Nếu cánh lệch, chỉnh
> `anchor_ratio` trong `main.py` hoặc dùng `--scale` để đổi kích thước.

## Công nghệ
- **MediaPipe PoseLandmarker (Tasks API)** — lấy vai để đặt cánh, hỗ trợ nhiều người.
- Model `pose_landmarker_lite.task` tự tải về `models/` lần đầu (nhẹ, hợp Iris Xe).
- Chạy tốt trên CPU, không cần CUDA.

## Cài đặt & chạy
```powershell
cd services\angel_wings
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
Hoặc `__bat\setup_angel_wings.bat` rồi `__bat\start_angel_wings.bat`, hoặc bấm nút trên Dashboard.

## Phím tắt
| Phím | Chức năng |
|---|---|
| `F` | Bật/tắt vẫy cánh |
| `E` | Bật/tắt hiệu ứng lấp lánh từ cánh |
| `B` | Đổi kiểu cánh: AUTO / luôn sau lưng / luôn trên |
| `N` | Đổi cánh kế tiếp (lần lượt theo tên file) |
| `+` / `-` | Phóng to / thu nhỏ cánh |
| `Q` / `ESC` | Thoát |

## Tham số
| Tham số | Ý nghĩa |
|---|---|
| `--camera` | Chỉ số webcam (mặc định 0) |
| `--source` | Đường dẫn video thay webcam |
| `--max-poses` | Số người tối đa (mặc định 3) |
| `--scale` | Bề rộng cánh = scale × bề rộng vai (mặc định 2.4) |
| `--front-th` | Ngưỡng nhận "mặt trước" để cho cánh ra sau lưng (mặc định 0.5) |
| `--flap` | Bật vẫy cánh ngay khi mở |
| `--effect` | Bật hiệu ứng ngay khi mở |
