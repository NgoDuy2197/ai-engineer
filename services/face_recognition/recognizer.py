"""
Loi nhan dien khuon mat dung chung cho ca service.

- Phat hien mat : YuNet (face_detection_yunet_2023mar.onnx)
- Trich dac trung: SFace (face_recognition_sface_2021dec.onnx) -> vector 128 chieu
- So khop      : cosine similarity (nguong mac dinh 0.363 theo SFace)

Toi uu cho may khong co CUDA (Intel Iris Xe / UHD):
  - 2 model (YuNet + SFace) rat nho -> chay real-time tot ngay tren CPU.
  - Mac dinh device='auto' dung CPU voi graph engine cua OpenCV (nhanh, on dinh).
  - Con co tuy chon 'opencl' / 'opencl_fp16' cho may co build OpenCV ton trong
    target GPU.

[Da kiem chung tren may test - OpenCV 5.0.0]: target OpenCL KHONG nhanh hon CPU
(graph engine moi bo qua target), nen CPU la lua chon nhanh nhat o day. Tren
build/phien ban OpenCV khac, OpenCL co the co ich - hay do bang phim tren HUD.
[Inference] Cac nhan xet ve toc do tren may khac chi la suy doan, chua do truc tiep.
"""
import os
import pickle
import urllib.request

import cv2
import numpy as np

# Giam log WARN cua OpenCV (vd canh bao "targets not supported by graph engine")
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:  # noqa
    pass

# ---- Duong dan chuan trong service ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "__data")
INBOX_DIR = os.path.join(DATA_DIR, "_inbox")      # anh tho ban nhet vao de gan nhan
DB_PATH = os.path.join(DATA_DIR, "db.pkl")        # embeddings sau khi train

DETECTOR_FILE = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
RECOGNIZER_FILE = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODEL_URLS = {
    DETECTOR_FILE: f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    RECOGNIZER_FILE: f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

COSINE_THRESHOLD = 0.363  # >= nguong nay coi la cung mot nguoi (khuyen nghi cua SFace)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def ensure_dirs():
    for d in (MODELS_DIR, DATA_DIR, INBOX_DIR):
        os.makedirs(d, exist_ok=True)


def open_camera(source="0"):
    """Mo camera/nguon video, thu nhieu backend tren Windows. None neu that bai."""
    src = int(source) if str(source).isdigit() else source
    if os.name == "nt" and isinstance(src, int):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
    else:
        backends = [0]
    for be in backends:
        cap = cv2.VideoCapture(src, be)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """Doc anh chiu duoc duong dan co dau (tieng Viet).

    cv2.imread tren Windows KHONG doc duoc path Unicode -> dung np.fromfile +
    cv2.imdecode. Tra None neu loi.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:  # noqa
        return None


def imwrite_unicode(path, img):
    """Ghi anh chiu duoc duong dan co dau (tieng Viet).

    cv2.imwrite tren Windows bao 'True' nhung KHONG tao file khi path co Unicode
    -> dung cv2.imencode + ndarray.tofile. Tra True/False theo ket qua thuc te.
    """
    try:
        ext = os.path.splitext(path)[1] or ".jpg"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(path)
        return os.path.exists(path)
    except Exception:  # noqa
        return False


def download_models():
    """Tai model tu OpenCV Zoo neu chua co (chi lam 1 lan)."""
    ensure_dirs()
    for path, url in MODEL_URLS.items():
        if os.path.exists(path):
            continue
        print(f"[i] Tai model: {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)
        print(f"    -> luu tai {path}")


def _resolve_target(device):
    """Chon (backend_id, target_id) cho OpenCV DNN theo lua chon nguoi dung.

    - cpu   : dung backend mac dinh (graph engine moi, nhanh nhat, khong canh bao).
    - opencl: ep backend OpenCV + target GPU (chi co ich neu build OpenCV ho tro).
    """
    ocv = cv2.dnn.DNN_BACKEND_OPENCV
    table = {
        "cpu": (cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU),
        "opencl": (ocv, cv2.dnn.DNN_TARGET_OPENCL),
        "opencl_fp16": (ocv, cv2.dnn.DNN_TARGET_OPENCL_FP16),
    }
    return table.get(device, table["cpu"])


class FaceEngine:
    """Gom detector + recognizer, dung chung cho enroll/train/camera."""

    def __init__(self, device="auto", score_threshold=0.7, input_size=(320, 320)):
        download_models()
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.device = self._init_backends(device)

    def _init_backends(self, device):
        """Tao detector/recognizer theo device; auto = CPU (nhanh & on dinh nhat)."""
        order = ["cpu"] if device == "auto" else [device, "cpu"]
        last_err = None
        for dev in order:
            backend, target = _resolve_target(dev)
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    DETECTOR_FILE, "", self.input_size,
                    self.score_threshold, 0.3, 5000, backend, target,
                )
                self.recognizer = cv2.FaceRecognizerSF.create(
                    RECOGNIZER_FILE, "", backend, target,
                )
                print(f"[i] Backend nhan dien: OpenCV DNN target={dev}")
                return dev
            except Exception as e:  # noqa
                last_err = e
                print(f"[!] Target '{dev}' khong dung duoc: {e}")
        raise RuntimeError(f"Khong khoi tao duoc engine: {last_err}")

    def detect(self, frame):
        """Tra ve mang Nx15 (toa do + landmark + score), hoac mang rong."""
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

    def embed(self, frame, face_row):
        """Can chinh mat theo landmark roi trich vector 128 chieu, chuan hoa L2.

        Chuan hoa L2 de sau nay tich vo huong (dot) = cosine similarity.
        """
        aligned = self.recognizer.alignCrop(frame, face_row)
        feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

    def embed_largest(self, frame):
        """Lay khuon mat lon nhat trong anh -> vector. None neu khong co mat."""
        faces = self.detect(frame)
        if len(faces) == 0:
            return None
        # cot 2,3 la w,h cua box -> chon dien tich lon nhat
        areas = faces[:, 2] * faces[:, 3]
        return self.embed(frame, faces[int(np.argmax(areas))])

    def cosine(self, a, b):
        """Cosine similarity giua 2 vector da chuan hoa L2."""
        return float(np.dot(a, b))


# ------------------- Co so du lieu embeddings -------------------

def load_db():
    """Doc db.pkl: dict {ten_nguoi: np.ndarray (K, 128)}."""
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "rb") as f:
        return pickle.load(f)


def save_db(db):
    ensure_dirs()
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)


def identify(embedding, db, threshold=COSINE_THRESHOLD):
    """So vector voi toan bo db. Tra ve (ten, diem). ('Khong ro', diem) neu duoi nguong."""
    best_name, best_score = "Khong ro", -1.0
    for name, mat in db.items():
        # cosine voi tung mau, lay diem cao nhat cua nguoi do
        scores = mat @ embedding
        s = float(scores.max())
        if s > best_score:
            best_name, best_score = name, s
    if best_score < threshold:
        return "Khong ro", best_score
    return best_name, best_score
