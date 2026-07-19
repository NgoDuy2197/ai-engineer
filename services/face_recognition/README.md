# Face Recognition Service

Nhan dien khuon mat va dan nhan ten nguoi tu camera, dua tren du lieu ban tu gan nhan.

## Cong nghe
- **Phat hien mat**: YuNet (OpenCV Zoo, ~0.3 MB)
- **Nhan dien**: SFace (OpenCV Zoo, ~37 MB) -> vector 128 chieu, so bang cosine
- **Toi uu cho may khong co CUDA (Intel Iris Xe / UHD)**: ca 2 model rat nho nen
  chay real-time tot ngay tren **CPU**. Mac dinh `--device auto` = CPU.

> [Da kiem chung tren may test - OpenCV 5.0.0]: target **OpenCL** KHONG nhanh hon CPU
> (graph engine moi cua OpenCV bo qua target GPU), thậm chí `opencl_fp16` con cham hon.
> Vi vay mac dinh de CPU. Tren build/phien ban OpenCV khac, OpenCL co the co ich —
> ban co the thu `--device opencl` va so FPS tren HUD.
>
> [Inference] Nhan xet ve toc do tren may Iris Xe cu the cua ban chi la suy doan
> (chua do truc tiep tren may do). Hay dua vao chi so FPS hien tren cua so camera.

## Cau truc thu muc
```
face_recognition/
  main.py          # diem vao: enroll | label | train | camera
  recognizer.py    # engine dung chung (detector + embedder + db)
  enroll.py        # thu mat tu webcam
  label_gui.py     # man hinh gan nhan (tkinter)
  trainer.py       # tao embeddings -> db.pkl
  camera.py        # nhan dien truc tiep + dan nhan
  draw.py          # ve chu co dau tieng Viet len frame
  models/          # YuNet + SFace .onnx (tu tai lan dau)
  __data/
    _inbox/        # anh tho ban nhet vao de gan nhan
    <ten_nguoi>/   # anh da gan nhan theo tung nguoi
    db.pkl         # embeddings (tu sinh sau khi train)
```

## Cai dat
```powershell
cd services\face_recognition
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
Lan chay dau tien se tu tai 2 file model tu OpenCV Zoo vao `models/` (can mang).

## Cach dung

### 1. Tao du lieu train
Co 2 cach:

**A. Thu tu webcam (nhanh nhat):**
```powershell
python main.py enroll --name "Nguyen Van A"
```
Bam `SPACE` de chup, `A` de bat che do tu dong chup, `Q` de thoat.
Nen chup 15-20 anh nhieu goc/anh sang. Anh luu vao `__data/Nguyen Van A/`.

**B. Nhet anh co san roi gan nhan bang man hinh:**
1. Copy anh vao `__data/_inbox/`
2. `python main.py label`
3. Voi moi anh: go ten (hoac chon ten da co) -> "Luu & Tiep".
   Anh se duoc chuyen vao `__data/<ten>/`.

### 2. Train
```powershell
python main.py train
```
Tao `__data/db.pkl` tu tat ca thu muc nguoi.

### 3. Chay camera nhan dien
```powershell
python main.py camera
```
Trong cua so: `+`/`-` chinh nguong nhan dien, `Q`/`ESC` thoat.
Nguoi khong khop se hien **"Khong ro"** (khung do); nguoi khop hien ten (khung xanh).

## Tham so huu ich
| Tham so | Y nghia |
|---|---|
| `--device` | `auto` / `opencl_fp16` / `opencl` / `cpu` |
| `--source` | `0` (webcam) hoac duong dan file video |
| `--threshold` | Nguong cosine (mac dinh 0.363; cao hon = chat hon) |
| `--det-size` | Kich thuoc dau vao detector; nho hon (vd 256) = nhanh hon |

## Meo chay muot tren may yeu (Iris Xe)
- Giam `--det-size` (vd `--det-size 256`) — anh huong lon nhat den FPS.
- Cu de `--device auto` (CPU). Chi thu `--device opencl` neu muon so sanh FPS.
- Chup 15-20 anh train chat luong cao, da dang goc/anh sang cho moi nguoi.
- Tang `--threshold` (vd 0.4-0.45) neu bi nhan nham nguoi; giam neu hay bao "Khong ro".
