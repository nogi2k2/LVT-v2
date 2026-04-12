"""DBSCAN clustering helper for binary images.

Tries to use sklearn.cluster.DBSCAN when available for performance. If sklearn
is not installed, falls back to connected-components bounding boxes.

Public API:
 - cluster_binary(bin_img, eps=5, min_samples=10) -> list of (x,y,w,h)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2


def _cc_bboxes(bin_img: np.ndarray, min_area: int = 1) -> List[Tuple[int, int, int, int]]:
    # Connected components fallback: find contours and bounding boxes
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def cluster_binary(bin_img: np.ndarray, eps: float = 5.0, min_samples: int = 10, min_area: int = 1) -> List[Tuple[int, int, int, int]]:
    """Cluster white pixels in a binary image and return bounding boxes per cluster.

    Parameters:
      bin_img: 8-bit single-channel binary image (foreground==255)
      eps: DBSCAN eps parameter (in pixels)
      min_samples: DBSCAN min_samples parameter
      min_area: minimum area (pixels) to keep a bbox

    Returns:
      List of (x,y,w,h) bounding boxes for each cluster (sorted top-to-bottom).
    """
    # Validate input
    if bin_img is None or bin_img.size == 0:
        return []

    # Extract foreground coordinates
    ys, xs = np.where(bin_img == 255)
    if len(xs) == 0:
        return []

    pts = np.column_stack((xs, ys))  # shape (N,2)

    try:
        # Coerce and validate parameters coming from the UI
        eps = float(eps)
        if eps <= 0:
            eps = 0.5
        min_samples = int(min_samples)
        if min_samples < 1:
            min_samples = 1
        min_area = int(min_area)
        if min_area < 1:
            min_area = 1

        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit_predict(pts)
        boxes = []
        for lab in np.unique(labels):
            if lab == -1:
                continue
            mask = labels == lab
            cluster_pts = pts[mask]
            x0, y0 = cluster_pts.min(axis=0)
            x1, y1 = cluster_pts.max(axis=0)
            w = int(x1 - x0 + 1)
            h = int(y1 - y0 + 1)
            if w * h < min_area:
                continue
            boxes.append((int(x0), int(y0), int(w), int(h)))
        boxes.sort(key=lambda b: (b[1], b[0]))
        return boxes
    except Exception:
        # sklearn not available or DBSCAN failed; fallback to connected components
        return _cc_bboxes(bin_img, min_area=min_area)
