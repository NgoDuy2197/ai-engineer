# Face Emoji Party

Game camera vui: tu dong dan emoji "nhay nhay ngoay ngoay" len mat, mui, tai, mom.
Cu 5-15 giay doi 1 emoji ngau nhien. Ha mom thi emoji ban ra lien tuc nhu voi phun,
ngam mom lai thi dung.

## Cong nghe
- **MediaPipe FaceLandmarker (Tasks API)** — 478 landmark khuon mat + blendshape.
- Phat hien **ha mom** bang blendshape `jawOpen` (chinh xac hon do hinh hoc).
- Emoji mau ve bang **Pillow + Segoe UI Emoji**; fallback hinh tron mau neu thieu font.
- Chay tot tren CPU (Intel Iris Xe), khong can CUDA. Model `face_landmarker.task`
  tu tai ve `models/` lan dau (can mang).

## Cai dat & chay
```powershell
cd services\face_emoji
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
Hoac dung: `__bat\setup_face_emoji.bat` roi `__bat\start_face_emoji.bat`,
hoac bam nut tren Dashboard (`run_dashboard.bat`).

## Phim tat
| Phim | Chuc nang |
|---|---|
| `R` | Doi ngay toan bo emoji |
| `M` | Bat/tat tieng |
| `+` / `-` | Chinh nguong "ha mom" |
| `Q` / `ESC` | Thoat |

## Tham so
| Tham so | Y nghia |
|---|---|
| `--camera` | Chi so webcam (mac dinh 0) |
| `--source` | Duong dan video thay cho webcam |
| `--max-faces` | So khuon mat toi da xu ly cung luc (mac dinh 4) |
| `--mouth-th` | Nguong jawOpen 0..1 (mac dinh 0.4) |
| `--min-gap` / `--max-gap` | Khoang giay giua 2 lan doi emoji (mac dinh 5-15) |

## Bo emoji
30 emoji mat/thu/bieu tuong (`main.py` -> `EMOJIS`) va 10 emoji ban ra tu mom
(`SHOOT_EMOJIS`). Muon them/bot chi can sua 2 danh sach do.
