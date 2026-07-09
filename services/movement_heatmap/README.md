# Movement Heatmap – Bản đồ nhiệt di chuyển

Theo dõi người qua camera và **cộng dồn "nhiệt"** vào vị trí họ đứng/đi. Chỗ nào
có nhiều người đi qua hoặc đứng lâu thì màu càng đậm (xanh → vàng → đỏ). Dùng để
thấy **luồng di chuyển** và **điểm dừng lâu** trong khung hình.

## Cách hoạt động

- YOLOv8 + ByteTrack phát hiện & bám theo từng người (chỉ class `person`).
- Mỗi frame, cộng một lượng nhiệt (`--gain`) vào **điểm chân** của mỗi người
  (hoặc cả thân nếu bật `--full-body`), làm mờ Gaussian để tạo vệt mượt.
- Đứng yên lâu ⇒ cùng một chỗ được cộng liên tục ⇒ nóng lên (đỏ).
- Phủ colormap (TURBO) lên khung hình với độ đậm tỉ lệ theo mức nhiệt.

## Phím tắt

| Phím | Chức năng |
|------|-----------|
| `H`  | Bật/tắt lớp heatmap |
| `B`  | Bật/tắt khung + ID người |
| `S`  | Lưu ảnh heatmap hiện tại vào `data/` |
| `C`  | Reset (xóa hết nhiệt) |
| `Q` / `ESC` | Thoát |

## Chạy

```bat
:: Cài đặt 1 lần
__bat\setup_movement_heatmap.bat

:: Chạy
__bat\start_movement_heatmap.bat
```

Ảnh heatmap tự lưu vào `data/heatmap_latest.png` mỗi 2 giây; nhấn `S` để lưu bản đặt tên theo thời gian.

## Tùy chọn hữu ích

```bat
:: "Heatmap sống" - nhiệt mờ dần theo thời gian (thấy hoạt động gần đây)
python services\movement_heatmap\main.py --decay 0.99

:: Lên màu nhanh hơn / vùng nhiệt to hơn
python services\movement_heatmap\main.py --gain 1.0 --radius 36

:: Tính nhiệt cả thân người thay vì chỉ điểm chân
python services\movement_heatmap\main.py --full-body

:: Dùng video có sẵn thay webcam
python services\movement_heatmap\main.py --source path\to\video.mp4
```

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--gain` | 0.6 | Nhiệt cộng thêm mỗi frame cho 1 người |
| `--radius` | 26 | Bán kính vùng nhiệt quanh chân (px) |
| `--decay` | 1.0 | 1.0 = cộng dồn mãi; <1 = mờ dần ("sống") |
| `--gamma` | 0.6 | Độ tương phản màu |
| `--alpha` | 0.75 | Độ đậm tối đa của lớp màu |

> Tối ưu Intel qua OpenVINO (`--device auto`), tự fallback CPU nếu không có.
> [Unverified] Hiệu năng tùy máy — hãy đo thực tế.
