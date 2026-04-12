"""DBSCAN wrapper for the project.

This module provides a stable project-local API for clustering binary images.
It delegates to References.dbscan_cluster.cluster_binary when available, and
falls back to a lightweight contour-based implementation otherwise.
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
import cv2

_ref = None
try:
    from References import dbscan_cluster as _ref
except Exception:
    _ref = None


def cluster_binary(bin_img: np.ndarray, eps: float = 5.0, min_samples: int = 10, min_area: int = 1) -> List[Tuple[int, int, int, int]]:
    """Cluster white pixels in a binary image and return bounding boxes per cluster.

    Delegates to References.dbscan_cluster.cluster_binary when available.
    """
    if _ref is not None:
        try:
            return _ref.cluster_binary(bin_img, eps=eps, min_samples=min_samples, min_area=min_area)
        except Exception:
            pass

    # Fallback: simple contour-based bounding boxes
    if bin_img is None or getattr(bin_img, 'size', 0) == 0:
        return []
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < int(min_area):
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes
