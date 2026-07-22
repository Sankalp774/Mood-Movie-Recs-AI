"""
OpenCV facial expression recognition.

Pipeline:
  image bytes → decode → grayscale → Haar face detect → crop →
  64×64 normalize → ONNX FER+ emotion model → mood mapping
"""

from __future__ import annotations

import base64
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# FER+ 8-class labels (ONNX model zoo emotion-ferplus)
FERPLUS_LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

# Map facial expressions → recommender moods
EXPRESSION_TO_MOOD = {
    "happiness": "whimsical",
    "surprise": "curious",
    "sadness": "emotional",
    "anger": "intense",
    "disgust": "dark",
    "fear": "tense",
    "contempt": "dark",
    "neutral": "reflective",
}

EXPRESSION_TO_QUERY = {
    "happiness": "warm light feel-good uplifting cozy",
    "surprise": "mind-bending curious twist wonder",
    "sadness": "emotional bittersweet melancholy reflective",
    "anger": "intense dark thrilling ambitious",
    "disgust": "dark tense gritty",
    "fear": "tense thriller horror suspense",
    "contempt": "dark satirical crime",
    "neutral": "reflective calm drama thoughtful",
}

ONNX_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/"
    "emotion_ferplus/model/emotion-ferplus-8.onnx"
)
ONNX_PATH = MODELS_DIR / "emotion-ferplus-8.onnx"

_session = None
_face_cascade = None


def _get_cascade():
    global _face_cascade
    if _face_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(path)
        if _face_cascade.empty():
            raise RuntimeError("Failed to load OpenCV Haar cascade for faces")
    return _face_cascade


def _ensure_onnx_model() -> Path:
    if ONNX_PATH.exists() and ONNX_PATH.stat().st_size > 10_000:
        return ONNX_PATH
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(ONNX_URL, ONNX_PATH)
    return ONNX_PATH


def _get_session():
    """Lazy-load ONNX Runtime session for FER+."""
    global _session
    if _session is not None:
        return _session
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError(
            "onnxruntime is required for expression recognition. "
            "pip install onnxruntime"
        ) from e
    model_path = str(_ensure_onnx_model())
    _session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _session


def decode_image(image_b64: str) -> np.ndarray:
    """Decode data-URL or raw base64 into BGR image."""
    raw = image_b64
    if "," in raw and raw.strip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    data = base64.b64decode(raw)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def detect_largest_face(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    cascade = _get_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if len(faces) == 0:
        return None
    # largest area
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def preprocess_face(gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    # slight padding
    pad = int(0.1 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad)
    y1 = min(gray.shape[0], y + h + pad)
    face = gray[y0:y1, x0:x1]
    face = cv2.resize(face, (64, 64), interpolation=cv2.INTER_AREA)
    face = face.astype(np.float32)
    # FER+ expects NCHW float
    face = np.expand_dims(face, axis=(0, 1))  # 1x1x64x64
    return face


def classify_expression(face_tensor: np.ndarray) -> dict[str, Any]:
    session = _get_session()
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: face_tensor})
    logits = outputs[0].reshape(-1)
    # softmax
    e = np.exp(logits - np.max(logits))
    probs = e / e.sum()
    idx = int(np.argmax(probs))
    label = FERPLUS_LABELS[idx]
    scores = {FERPLUS_LABELS[i]: float(probs[i]) for i in range(len(FERPLUS_LABELS))}
    return {
        "expression": label,
        "confidence": float(probs[idx]),
        "scores": scores,
    }


def analyze_image_b64(image_b64: str) -> dict[str, Any]:
    """
    Full OpenCV pipeline: face detect + FER+ expression → mood for recommender.
    """
    img = decode_image(image_b64)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    box = detect_largest_face(gray)
    if box is None:
        return {
            "ok": False,
            "error": "no_face",
            "message": "No face detected. Face the camera with good lighting.",
            "opencv": True,
        }

    face_tensor = preprocess_face(gray, box)
    try:
        result = classify_expression(face_tensor)
    except Exception as e:
        # OpenCV face detected but model failed — still return face box
        return {
            "ok": False,
            "error": "model_error",
            "message": str(e),
            "face_box": {"x": box[0], "y": box[1], "w": box[2], "h": box[3]},
            "opencv": True,
        }

    expression = result["expression"]
    mood = EXPRESSION_TO_MOOD.get(expression, "curious")
    query = EXPRESSION_TO_QUERY.get(expression, "")

    return {
        "ok": True,
        "opencv": True,
        "face_box": {"x": box[0], "y": box[1], "w": box[2], "h": box[3]},
        "expression": expression,
        "confidence": result["confidence"],
        "scores": result["scores"],
        "mood": mood,
        "query_hint": query,
        "engine": "opencv-haar + emotion-ferplus-onnx",
    }
