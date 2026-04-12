"""Embeddings wrapper for the project.

Provides load_embedder, embed_crop, and embed_boxes functions that delegate to
References.embeddings_mobilenet when present; otherwise a fast OpenCV fallback
is used.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple
import numpy as np
import cv2

_ref = None
try:
    from References import embeddings_mobilenet as _ref
except Exception:
    _ref = None
    
def _embed_fallback(bgr: np.ndarray) -> np.ndarray:
        if bgr is None or getattr(bgr, 'size', 0) == 0:
            return np.zeros((64 * 64 * 3,), dtype=np.float32)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
        vec = small.astype(np.float32) / 255.0
        return vec.ravel()

def load_embedder() -> Tuple[Optional[object], Callable[[np.ndarray], np.ndarray]]:
    """Load an embedder. Returns (model_or_none, embed_fn).

    If References.embeddings_mobilenet is available it will be used; otherwise
    a simple fallback that resizes to 64x64 and flattens will be returned.
    """
    if _ref is not None:
        try:
            import warnings
            with warnings.catch_warnings():
                # Suppress torchvision MobileNet deprecation warnings —
                # the model loads correctly, these warnings are just noise.
                warnings.filterwarnings('ignore', category=UserWarning,
                                        module='torchvision')
                return _ref.load_embedder()
        except Exception:
            pass

    return None, _embed_fallback


def embed_crop(bgr_crop: np.ndarray, model_or_none=None) -> np.ndarray:
    if _ref is not None:
        try:
            return _ref.embed_crop(bgr_crop, model_or_none)
        except Exception:
            pass

    return _embed_fallback(bgr_crop)


def embed_boxes(original_bgr, boxes, model_or_none=None):
    if _ref is not None:
        try:
            return _ref.embed_boxes(original_bgr, boxes, model_or_none)
        except Exception:
            pass
    embeddings = []
    for (x, y, w, h) in boxes:
        if w <= 0 or h <= 0:
            embeddings.append(np.zeros((1,), dtype=np.float32))
            continue
        embeddings.append(_embed_fallback(original_bgr[y:y+h, x:x+w]))
    return embeddings