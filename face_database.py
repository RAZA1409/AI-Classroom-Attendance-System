"""
Face Database & Biometric Recognition Module
============================================
Handles face database loading, genuine face detection inside person crops,
unbypassable face quality validation (blur and minimum size), BGR-standardized
embedding extraction using FaceNet512, and zero-division safe cosine similarity.

Privacy Contract:
Does not write student names, biometric embeddings, or student image paths
to persistent logs.
"""

import os
import pickle
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from deepface import DeepFace

from config import config
from logger import get_logger

logger = get_logger("face_database")


def load_database(path: Optional[str] = None) -> Dict[str, List[Any]]:
    """
    Safely load pre-computed face embeddings from a pickle file.
    Validates file existence, readability, and dictionary structure.
    """
    db_path = path or config.DATABASE_PATH

    if not os.path.exists(db_path):
        logger.warning(
            "Face database file not found. Live recognition will classify all detections as 'Unknown'."
        )
        return {}

    try:
        with open(db_path, "rb") as f:
            database = pickle.load(f)

        if not isinstance(database, dict):
            logger.error("Face database file does not contain a valid dictionary structure.")
            return {}

        logger.info("Loaded enrolled students database: %d identities present.", len(database))
        return database
    except Exception as e:
        logger.error("Failed to load face database: %s", e)
        return {}


def cosine_similarity(a: Any, b: Any) -> float:
    """
    Compute cosine similarity with zero-division, non-finite value,
    and dimension mismatch safeguards.
    """
    if a is None or b is None:
        return 0.0

    try:
        arr_a = np.asarray(a, dtype=np.float32)
        arr_b = np.asarray(b, dtype=np.float32)
    except (ValueError, TypeError):
        return 0.0

    # Dimension and shape mismatch check
    if arr_a.shape != arr_b.shape or arr_a.size == 0 or arr_b.size == 0:
        return 0.0

    # Non-finite values check (NaN, Inf)
    if not np.all(np.isfinite(arr_a)) or not np.all(np.isfinite(arr_b)):
        return 0.0

    norm_a = float(np.linalg.norm(arr_a))
    norm_b = float(np.linalg.norm(arr_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot = float(np.dot(arr_a, arr_b))
    return float(dot / (norm_a * norm_b))


def validate_face_quality(
    face_img: Any,
    min_width: Optional[int] = None,
    min_height: Optional[int] = None,
    blur_threshold: Optional[float] = None,
) -> Tuple[bool, str, float]:
    """
    Validate quality of a detected face image.
    Enforces minimum dimensions and minimum sharpness via Laplacian variance.

    Returns:
        (is_valid: bool, reason: str, blur_variance: float)
    """
    min_w = min_width if min_width is not None else config.MIN_FACE_WIDTH
    min_h = min_height if min_height is not None else config.MIN_FACE_HEIGHT
    blur_th = blur_threshold if blur_threshold is not None else config.BLUR_THRESHOLD

    if face_img is None or not isinstance(face_img, np.ndarray) or face_img.size == 0:
        return False, "Empty or invalid face image", 0.0

    if len(face_img.shape) < 2:
        return False, "Invalid image dimensions", 0.0

    h, w = face_img.shape[:2]
    if w <= 0 or h <= 0:
        return False, "Zero-area face image", 0.0

    if w < min_w or h < min_h:
        return False, f"Face dimensions ({w}x{h}) beneath minimum required ({min_w}x{min_h})", 0.0

    # Calculate blur using Laplacian variance
    try:
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img

        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception as e:
        return False, f"Blur calculation error: {e}", 0.0

    if variance < blur_th:
        return False, f"Face image too blurry (variance {variance:.1f} < {blur_th:.1f})", variance

    return True, "OK", variance


def detect_single_face(
    image_or_crop: np.ndarray,
    detector_backend: Optional[str] = None,
    min_width: Optional[int] = None,
    min_height: Optional[int] = None,
    blur_threshold: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], Optional[dict], str]:
    """
    Detect exactly one genuine face inside a candidate image or person crop.
    Enforces standardized BGR face output and unbypassable quality validation.

    Rejects:
    - Empty or invalid inputs.
    - No face detected (enforce_detection=True).
    - Multiple faces (ambiguous identity association).
    - Missing or non-positive facial-area coordinates.
    - Facial-area coordinates completely outside input image bounds.
    - Empty extracted raw face crop.
    - Sub-minimum face dimensions.
    - Blurry face images beneath Laplacian variance threshold.

    Returns:
        (extracted_bgr_face: Optional[np.ndarray], facial_area: Optional[dict], status_reason: str)
    """
    if image_or_crop is None or not isinstance(image_or_crop, np.ndarray) or image_or_crop.size == 0:
        return None, None, "Empty or invalid input image"

    h, w = image_or_crop.shape[:2]
    min_w = min_width if min_width is not None else config.MIN_FACE_WIDTH
    min_h = min_height if min_height is not None else config.MIN_FACE_HEIGHT
    blur_th = blur_threshold if blur_threshold is not None else config.BLUR_THRESHOLD

    if w < min_w or h < min_h:
        return None, None, f"Input crop ({w}x{h}) smaller than minimum face dimensions ({min_w}x{min_h})"

    backend = detector_backend or config.FACE_DETECTOR_BACKEND

    # 1. Run genuine face detection with standardized BGR output (unnormalized 0-255 pixels)
    try:
        faces = DeepFace.extract_faces(
            img_path=image_or_crop,
            detector_backend=backend,
            enforce_detection=True,
            align=True,
            color_face="bgr",
            normalize_face=False,
        )
    except ValueError:
        # DeepFace raises ValueError when enforce_detection=True and 0 faces located
        return None, None, "No face detected"
    except Exception as e:
        logger.debug("Face detector exception: %s", e)
        return None, None, f"Face detector error: {e}"

    if not faces:
        return None, None, "No face detected"

    if len(faces) > 1:
        logger.debug("Conservative rejection: %d faces detected in person region (ambiguous)", len(faces))
        return None, None, f"Multiple faces detected ({len(faces)})"

    face_entry = faces[0]
    facial_area = face_entry.get("facial_area")

    # 2. Strict validation of facial-area coordinates
    if not isinstance(facial_area, dict):
        return None, None, "Missing facial-area coordinates"

    if "x" not in facial_area or "y" not in facial_area or "w" not in facial_area or "h" not in facial_area:
        return None, facial_area, "Incomplete facial-area coordinates"

    fx = facial_area.get("x", 0)
    fy = facial_area.get("y", 0)
    fw = facial_area.get("w", 0)
    fh = facial_area.get("h", 0)

    # Require positive width and height
    if fw <= 0 or fh <= 0:
        return None, facial_area, f"Zero-area face box ({fw}x{fh})"

    # Require coordinates to overlap the input image
    if fx >= w or fy >= h or (fx + fw) <= 0 or (fy + fh) <= 0:
        return None, facial_area, "Facial area outside image boundaries"

    # Enforce minimum face dimensions on detected facial area
    if fw < min_w or fh < min_h:
        return None, facial_area, f"Detected face ({fw}x{fh}) smaller than minimum ({min_w}x{min_h})"

    # 3. Extract and validate raw face crop for blur & quality (UNBYPASSABLE)
    cy1 = max(0, fy)
    cy2 = min(h, fy + fh)
    cx1 = max(0, fx)
    cx2 = min(w, fx + fw)

    if cy2 <= cy1 or cx2 <= cx1:
        return None, facial_area, "Empty raw face crop"

    raw_face_crop = image_or_crop[cy1:cy2, cx1:cx2]
    if raw_face_crop is None or raw_face_crop.size == 0:
        return None, facial_area, "Empty raw face crop"

    is_valid, reason, _ = validate_face_quality(raw_face_crop, min_w, min_h, blur_th)
    if not is_valid:
        return None, facial_area, reason

    # 4. Verified face crop output (standardized BGR, unnormalized [0, 255] range)
    extracted_face = face_entry.get("face")
    if extracted_face is None or not isinstance(extracted_face, np.ndarray) or extracted_face.size == 0:
        return None, facial_area, "Empty extracted aligned face"

    return extracted_face, facial_area, "OK"


def extract_embedding(
    face_arr: np.ndarray,
    model_name: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Generate facial representation embedding from an aligned BGR face array
    using skip detector backend (face is already detected and aligned).
    Explicitly applies normalization='base' during representation.
    """
    if face_arr is None or not isinstance(face_arr, np.ndarray) or face_arr.size == 0:
        return None

    model = model_name or config.FACE_RECOGNITION_MODEL

    try:
        rep = DeepFace.represent(
            img_path=face_arr,
            model_name=model,
            detector_backend="skip",
            enforce_detection=False,
            normalization="base",
        )
        if rep and len(rep) > 0 and "embedding" in rep[0]:
            return np.array(rep[0]["embedding"], dtype=np.float32)
        return None
    except Exception as e:
        logger.debug("Failed to extract embedding: %s", e)
        return None


def recognize_face_detailed(
    face_or_crop: np.ndarray,
    database: Dict[str, List[Any]],
    detector_backend: Optional[str] = None,
    threshold: Optional[float] = None,
) -> Tuple[str, float, Optional[dict], str]:
    """
    Full recognition pipeline returning match name, confidence score,
    detected facial area, and descriptive status reason.
    """
    if not database:
        return "Unknown", 0.0, None, "Database empty"

    if face_or_crop is None or not isinstance(face_or_crop, np.ndarray) or face_or_crop.size == 0:
        return "Unknown", 0.0, None, "Invalid image"

    sim_threshold = threshold if threshold is not None else config.RECOGNITION_THRESHOLD

    # 1. Detect single face and validate quality (unbypassable)
    extracted_face, facial_area, status = detect_single_face(
        face_or_crop,
        detector_backend=detector_backend,
    )
    if extracted_face is None:
        return "Unknown", 0.0, facial_area, status

    # 2. Extract embedding from verified detected face
    embedding = extract_embedding(extracted_face)
    if embedding is None:
        return "Unknown", 0.0, facial_area, "Embedding extraction failed"

    # 3. Match against database
    best_match = "Unknown"
    best_score = -1.0

    for name, embeddings in database.items():
        for db_emb in embeddings:
            score = cosine_similarity(embedding, db_emb)
            if score > best_score:
                best_score = score
                best_match = name

    if best_score >= sim_threshold:
        return best_match, best_score, facial_area, "Match confirmed"
    else:
        return "Unknown", best_score, facial_area, f"Similarity below threshold ({best_score:.2f} < {sim_threshold:.2f})"


def recognize_face(
    face_or_crop: np.ndarray,
    database: Dict[str, List[Any]],
    debug: bool = False,
    detector_backend: Optional[str] = None,
    threshold: Optional[float] = None,
) -> str:
    """
    Convenience wrapper preserving the legacy interface: returns student name or 'Unknown'.
    Does not log student identifying names.
    """
    name, score, _, reason = recognize_face_detailed(
        face_or_crop=face_or_crop,
        database=database,
        detector_backend=detector_backend,
        threshold=threshold,
    )

    if debug:
        logger.debug(
            "Recognition evaluated: match_status=%s | score=%.3f | reason=%s",
            "Confirmed" if name != "Unknown" else "Unknown",
            score,
            reason,
        )

    return name
