# People Counter

Đếm người qua camera + đếm người **vào/ra quán** khi đi qua một vạch bạn vẽ.
Dùng **YOLOv8 tracking (ByteTrack)** + **OpenVINO**. Ghi thống kê ra `data/stats.json`
để **Dashboard web** phân tích theo phút.

## Chạy

```bat
REM Cài đặt 1 lần
__bat\setup_people_counter.bat

REM Bật service (cửa sổ camera)
__bat\start_people_counter.bat
```

Xem dashboard phân tích: mở `run_dashboard.bat` → bấm **Mở Dashboard** ở thẻ People Counter
(hoặc vào `http://localhost:8000/counter.html`). Nên chạy **song song**: một cửa sổ camera + một tab dashboard.

## Cách dùng cửa sổ camera

| Thao tác | Ý nghĩa |
|---|---|
| Click chuột trái 2 điểm | Vẽ vạch đếm |
| `R` | Xóa vạch để vẽ lại |
| `F` | Đảo chiều "vào/ra" (nếu đếm ngược) |
| `C` | Reset bộ đếm |
| `Q` / `ESC` | Thoát |

- **Tổng người thấy**: mỗi người mới xuất hiện +1 (dựa trên track ID).
- **Vào quán**: mỗi lần một người đi qua vạch theo chiều "vào" +1.
- **Tỉ lệ chuyển đổi** (dashboard) = Vào quán / Tổng người thấy.

## Chỉ số trên Dashboard

- Đang trong khung, Tổng người thấy, Tổng vào, Tổng ra, Đang trong quán, Tỉ lệ chuyển đổi.
- Biểu đồ **vào/ra theo từng phút** (15 phút gần nhất).

> Mặc định có sẵn 1 vạch dọc giữa khung để chạy ngay; click 2 điểm để vẽ vạch của bạn.
