
import threading
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from skimage.feature import local_binary_pattern


# ============================================================
# 1. KONFIGURASI MEDIAPIPE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)

LANDMARK_COUNT = 478

LANDMARK_PAIRS = [
    (172, 397),
    (0, 14),
    (14, 17),
    (468, 473),
    (133, 362),
    (33, 263),
    (98, 327),
    (168, 2),
    (168, 4),
    (61, 291),
    (168, 152),
    (2, 152),
    (159, 145),
    (386, 374),
    (33, 133),
    (263, 362),
    (468, 159),
    (468, 145),
    (473, 386),
    (473, 374),
    (37, 267),
    (2, 0),
]

ANTHROPOMETRY_FEATURE_NAMES = [
    "Mandible Width",
    "Upper Vermilion Height",
    "Lower Vermilion Height",
    "Interpupillary Distance (IPD)",
    "Intercanthal Width",
    "Biocular Width",
    "Nasal Width",
    "Nasal Height",
    "Nasal Length",
    "Mouth Width",
    "Morphological Facial Height",
    "Lower Facial Height",
    "PFH Left",
    "PFH Right",
    "HPA Left",
    "HPA Right",
    "Approx MRD1 Left",
    "Approx MRD2 Left",
    "Approx MRD1 Right",
    "Approx MRD2 Right",
    "Approx Philtrum Width",
    "Approx Philtrum Height",
]

_face_landmarker = None
_LANDMARK_LOCK = threading.RLock()


# ============================================================
# 2. MEMUAT MODEL MEDIAPIPE
# ============================================================

def _ensure_model_file():

    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0:
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MODEL_PATH.with_suffix(".task.tmp")

    try:
        urllib.request.urlretrieve(MODEL_URL, str(tmp_path))

        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            raise RuntimeError("File model hasil unduhan kosong.")

        tmp_path.replace(MODEL_PATH)

    except Exception as exc:

        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass

        raise FileNotFoundError(
            f"Model MediaPipe tidak ditemukan: {MODEL_PATH}. "
            "Letakkan face_landmarker.task di folder models."
        ) from exc


def _create_face_landmarker():

    _ensure_model_file()

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )

    return vision.FaceLandmarker.create_from_options(options)


def _get_face_landmarker():

    global _face_landmarker

    with _LANDMARK_LOCK:

        if _face_landmarker is None:
            _face_landmarker = _create_face_landmarker()

        return _face_landmarker


# ============================================================
# 3. VALIDASI GAMBAR BGR
# ============================================================

def _validate_bgr_image(image_bgr):

    if image_bgr is None:
        raise ValueError("Gambar kosong atau None.")

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("Input gambar harus berupa numpy.ndarray.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Input harus berupa gambar tiga channel BGR.")

    if image_bgr.size == 0:
        raise ValueError("Gambar kosong.")

    if image_bgr.dtype != np.uint8:
        raise TypeError(
            f"Format pixel harus uint8, bukan {image_bgr.dtype}."
        )


# ============================================================
# 4. DETEKSI LANDMARK
# ============================================================

def detect_landmarks(image_bgr):
    """
    Input:
        Satu crop wajah BGR uint8.

    Output:
        List array float32 berukuran (478, 3).
        Jika wajah tidak ditemukan, mengembalikan [].
    """

    if image_bgr is None:
        return []

    _validate_bgr_image(image_bgr)

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    height, width = image_rgb.shape[:2]

    image_rgb = np.ascontiguousarray(
        image_rgb,
        dtype=np.uint8
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    with _LANDMARK_LOCK:

        landmarker = _get_face_landmarker()
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return []

    all_landmarks = []

    for face in result.face_landmarks:

        points = np.asarray(
            [
                [
                    landmark.x * width,
                    landmark.y * height,
                    landmark.z * width
                ]
                for landmark in face
            ],
            dtype=np.float32
        )

        if points.shape == (LANDMARK_COUNT, 3):
            all_landmarks.append(points)

    return all_landmarks


# ============================================================
# 5. ANTROPOMETRI 22 FITUR
# ============================================================

def extract_antropometri22(landmarks):
    """
    Menghitung 22 jarak Euclidean 2D sesuai urutan training.

    Output:
        Array float32 berukuran (22,).
    """

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32
    )

    if landmarks.shape != (LANDMARK_COUNT, 3):

        raise ValueError(
            f"Landmark harus berukuran "
            f"({LANDMARK_COUNT}, 3), "
            f"tetapi diperoleh {landmarks.shape}."
        )

    if not np.isfinite(landmarks).all():
        raise ValueError(
            "Landmark mengandung NaN atau infinity."
        )

    features = []

    for idx1, idx2 in LANDMARK_PAIRS:

        point1 = landmarks[idx1, :2]
        point2 = landmarks[idx2, :2]

        distance = np.linalg.norm(point1 - point2)

        features.append(float(distance))

    features = np.asarray(
        features,
        dtype=np.float32
    )

    if features.shape != (22,):
        raise ValueError(
            f"Jumlah fitur antropometri salah: {features.shape}"
        )

    return features


# ============================================================
# 6. LBP GRID 3x3
# ============================================================

def extract_lbp(image_bgr):
    """
    Pure LBP:
    P=8, R=1, method=default.
    Grid 3x3.
    256 bin per cell.
    Normalisasi histogram per cell.

    Output:
        Array float32 berukuran (2304,).
    """

    _validate_bgr_image(image_bgr)

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    height, width = gray.shape

    grid_rows = 3
    grid_cols = 3

    radius = 1
    n_points = 8 * radius
    num_bins = 256

    if height < grid_rows or width < grid_cols:
        raise ValueError(
            "Ukuran gambar terlalu kecil untuk grid LBP 3x3."
        )

    lbp_matrix = local_binary_pattern(
        gray,
        n_points,
        radius,
        method="default"
    )

    features = []

    for row in range(grid_rows):

        y0 = row * height // grid_rows
        y1 = (row + 1) * height // grid_rows

        for col in range(grid_cols):

            x0 = col * width // grid_cols
            x1 = (col + 1) * width // grid_cols

            cell = lbp_matrix[y0:y1, x0:x1]

            hist, _ = np.histogram(
                cell.ravel(),
                bins=num_bins,
                range=(0, 256)
            )

            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-6

            features.extend(hist)

    features = np.asarray(
        features,
        dtype=np.float32
    )

    if features.shape != (2304,):
        raise ValueError(
            f"Jumlah fitur LBP salah: {features.shape}"
        )

    return features


# ============================================================
# 7. MENUTUP RESOURCE MEDIAPIPE
# ============================================================

def close_face_landmarker():

    global _face_landmarker

    with _LANDMARK_LOCK:

        if _face_landmarker is not None:

            _face_landmarker.close()
            _face_landmarker = None
