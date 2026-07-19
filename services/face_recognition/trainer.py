"""
Tao co so du lieu embeddings tu du lieu da gan nhan.

Doc cac thu muc __data/<ten_nguoi>/*.jpg, trich vector 128 chieu cho tung anh,
gom lai theo tung nguoi va luu vao __data/db.pkl.

Chay:  python main.py train
"""
import os

import numpy as np

from recognizer import (DATA_DIR, IMAGE_EXTS, FaceEngine, ensure_dirs,
                        imread_unicode, save_db)


def _list_people():
    out = []
    for name in sorted(os.listdir(DATA_DIR)):
        p = os.path.join(DATA_DIR, name)
        if os.path.isdir(p) and not name.startswith("_"):
            out.append(name)
    return out


def build(device="auto"):
    ensure_dirs()
    engine = FaceEngine(device=device)
    people = _list_people()
    if not people:
        print(f"[!] Chua co du lieu trong {DATA_DIR}. Hay enroll hoac label truoc.")
        return

    db = {}
    for name in people:
        person_dir = os.path.join(DATA_DIR, name)
        vecs = []
        files = [f for f in sorted(os.listdir(person_dir))
                 if f.lower().endswith(IMAGE_EXTS)]
        for f in files:
            img = imread_unicode(os.path.join(person_dir, f))
            if img is None:
                print(f"    [bo qua] khong doc duoc anh: {name}/{f}")
                continue
            emb = engine.embed_largest(img)
            if emb is None:
                print(f"    [bo qua] khong thay mat: {name}/{f}")
                continue
            vecs.append(emb)
        if vecs:
            db[name] = np.vstack(vecs).astype(np.float32)
            print(f"[i] {name}: {len(vecs)}/{len(files)} anh co mat")
        else:
            print(f"[!] {name}: khong trich duoc mat nao -> bo qua")

    if not db:
        print("[!] Khong tao duoc embedding nao. Kiem tra lai anh train.")
        return
    save_db(db)
    total = sum(len(v) for v in db.values())
    print(f"[OK] Da luu db.pkl: {len(db)} nguoi, {total} vector -> {DATA_DIR}/db.pkl")


if __name__ == "__main__":
    build()
