"""
Man hinh gan nhan du lieu train (tkinter).

Luong dung:
  1. Nhet anh (jpg/png...) vao thu muc __data/_inbox/
  2. Chay:  python main.py label
  3. Voi moi anh: app tu phat hien khuon mat lon nhat va ve khung.
     - Go ten nguoi (hoac bam chon ten da co ben phai).
     - Bam "Luu & Tiep" -> anh duoc chuyen vao __data/<ten>/, xoa khoi inbox.
     - "Bo qua" de sang anh khac; "Xoa anh" neu anh hong.
  4. Xong thi chay:  python main.py train

Ten co dau tieng Viet duoc giu nguyen trong ten thu muc.
"""
import os
import shutil
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from recognizer import (DATA_DIR, INBOX_DIR, IMAGE_EXTS, FaceEngine, ensure_dirs,
                        imread_unicode)

PREVIEW_MAX = 520  # canh dai toi da khi hien thi preview


def list_inbox():
    return sorted(
        f for f in os.listdir(INBOX_DIR)
        if f.lower().endswith(IMAGE_EXTS)
    )


def list_people():
    """Cac thu muc nguoi da co trong __data (bo qua _inbox va file he thong)."""
    out = []
    for name in sorted(os.listdir(DATA_DIR)):
        p = os.path.join(DATA_DIR, name)
        if os.path.isdir(p) and not name.startswith("_"):
            out.append(name)
    return out


class LabelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gan nhan du lieu train - Face Recognition")
        self.engine = FaceEngine(device="auto")

        self.files = list_inbox()
        self.idx = 0
        self.cur_img_bgr = None

        # ---- Bo cuc ----
        main = ttk.Frame(root, padding=10)
        main.pack(fill="both", expand=True)

        # Cot trai: preview
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="n")
        self.canvas = tk.Label(left, background="#222")
        self.canvas.pack()
        self.info = ttk.Label(left, text="", font=("Segoe UI", 10))
        self.info.pack(pady=(8, 0))

        # Cot phai: dieu khien
        right = ttk.Frame(main, padding=(16, 0, 0, 0))
        right.grid(row=0, column=1, sticky="n")

        ttk.Label(right, text="Ten nguoi:", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.name_var = tk.StringVar()
        entry = ttk.Entry(right, textvariable=self.name_var, width=28, font=("Segoe UI", 11))
        entry.pack(anchor="w", pady=(2, 10))
        entry.bind("<Return>", lambda e: self.save_next())

        ttk.Label(right, text="Nguoi da co (bam de chon):").pack(anchor="w")
        self.people_box = tk.Listbox(right, height=10, width=28, exportselection=False)
        self.people_box.pack(anchor="w", pady=(2, 10))
        self.people_box.bind("<<ListboxSelect>>", self._pick_person)

        ttk.Button(right, text="Luu & Tiep  (Enter)", command=self.save_next).pack(fill="x", pady=2)
        ttk.Button(right, text="Bo qua", command=self.skip).pack(fill="x", pady=2)
        ttk.Button(right, text="Xoa anh", command=self.delete_current).pack(fill="x", pady=2)
        ttk.Button(right, text="Thoat", command=root.destroy).pack(fill="x", pady=(12, 2))

        self.refresh_people()
        self.show_current()
        entry.focus_set()

    def refresh_people(self):
        self.people_box.delete(0, tk.END)
        for p in list_people():
            self.people_box.insert(tk.END, p)

    def _pick_person(self, _evt):
        sel = self.people_box.curselection()
        if sel:
            self.name_var.set(self.people_box.get(sel[0]))

    def show_current(self):
        if self.idx >= len(self.files):
            self.canvas.config(image="", text="Da xong het anh trong _inbox!",
                               font=("Segoe UI", 14), width=40, height=15)
            self.info.config(text="Hay chay:  python main.py train")
            return

        fname = self.files[self.idx]
        path = os.path.join(INBOX_DIR, fname)
        img = imread_unicode(path)
        if img is None:
            self.skip()
            return
        self.cur_img_bgr = img

        # phat hien mat de ve khung xem truoc
        faces = self.engine.detect(img)
        disp = img.copy()
        n = len(faces)
        if n:
            areas = faces[:, 2] * faces[:, 3]
            x, y, w, h = faces[int(areas.argmax())][:4].astype(int)
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 200, 0), 2)

        # resize giu ti le cho vua khung preview
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((PREVIEW_MAX, PREVIEW_MAX))
        self.tkimg = ImageTk.PhotoImage(pil)
        self.canvas.config(image=self.tkimg, text="", width=0, height=0)

        state = f"{n} mat" if n else "KHONG thay mat (nen bo qua/xoa)"
        self.info.config(text=f"[{self.idx + 1}/{len(self.files)}]  {fname}   -   {state}")

    def save_next(self):
        if self.idx >= len(self.files):
            return
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Thieu ten", "Hay nhap ten nguoi truoc khi luu.")
            return
        if self.engine.detect(self.cur_img_bgr).shape[0] == 0:
            if not messagebox.askyesno("Khong co mat",
                                       "Anh nay khong phat hien duoc mat. Van luu?"):
                return

        person_dir = os.path.join(DATA_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        fname = self.files[self.idx]
        dst = os.path.join(person_dir, fname)
        # tranh ghi de trung ten
        base, ext = os.path.splitext(fname)
        k = 1
        while os.path.exists(dst):
            dst = os.path.join(person_dir, f"{base}_{k}{ext}")
            k += 1
        shutil.move(os.path.join(INBOX_DIR, fname), dst)

        self.refresh_people()
        self.idx += 1
        self.show_current()

    def skip(self):
        self.idx += 1
        self.show_current()

    def delete_current(self):
        if self.idx >= len(self.files):
            return
        fname = self.files[self.idx]
        if messagebox.askyesno("Xoa anh", f"Xoa han '{fname}'?"):
            os.remove(os.path.join(INBOX_DIR, fname))
            del self.files[self.idx]
            self.show_current()


def run():
    ensure_dirs()
    if not list_inbox():
        print(f"[!] Chua co anh nao trong: {INBOX_DIR}")
        print("    Hay copy anh can gan nhan vao thu muc do roi chay lai.")
        return
    root = tk.Tk()
    LabelApp(root)
    root.mainloop()


if __name__ == "__main__":
    run()
