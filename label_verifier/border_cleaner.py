"""Border cleaner: HSV-V adaptive threshold -> denoise -> area filter -> dilate -> bbox crop

This module implements a compact border/background cleaner used by the
label verification pipeline. It follows a simple pipeline tuned for scanned
labels and documents where dark symbols, text and barcodes should be
grouped into a single region and cropped.

Algorithm (high level):
1) Convert to HSV and extract the V (brightness) channel.
2) Apply Otsu thresholding on V to detect dark objects.
3) Median filter + small morphological opening to remove speckles.
4) Remove very small connected components (area threshold).
5) Dilate with a large rectangular kernel (approx 12% of image size) to
   merge separate label parts into a single connected region.
6) Compute bounding rectangle, add padding (~1%) and return cropped image.

Public entry points:
    clean_label(img, config, input_path, page_index)
        -> cropped BGR image (or original on failure)

    clean_label_vplane(img, config)
        -> (V-channel crop, (x1, y1, x2, y2)) tuple

Configuration keys (all optional):
    store_debug_images   : bool  - write debug images to debug_image_dir
    debug_image_dir      : str   - directory for debug outputs
    cleaner_min_area_frac: float - min component area fraction (default 1e-5)
    cleaner_open_frac    : float - small open kernel fraction (default 0.0002)
    cleaner_dilate_frac  : float - dilation kernel fraction (default 0.12)
    cleaner_dilate_iters : int   - dilation iterations (default 1)
    cleaner_pad_frac     : float - padding fraction for final bbox (default 0.01)
"""

__border_cleaner_version__ = "2"

import os
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _odd(n: int) -> int:
    """Return n if odd, else n+1. Minimum value is 3."""
    n = max(3, n)
    return n if n % 2 != 0 else n + 1


# ── Shared detection pipeline ──────────────────────────────────────────────

def _detect_bbox(
    img: np.ndarray,
    config: dict,
) -> Optional[Tuple[int, int, int, int]]:
    """Run the shared detection pipeline and return (x1, y1, x2, y2) or None.

    This is the single source of truth for the HSV-Otsu-morphology pipeline
    used by both clean_label() and clean_label_vplane(). Any tuning to the
    algorithm should be made here only.
    """
    H, W = img.shape[:2]

    # 1) HSV → V channel
    V  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    Vb = cv2.GaussianBlur(V, (3, 3), 0)

    # 2) Otsu threshold — dark regions become foreground (white)
    _, bw = cv2.threshold(Vb, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3) Median filter + small morphological opening to remove speckles
    bw = cv2.medianBlur(bw, 3)
    open_frac = float(config.get('cleaner_open_frac', 0.0002))
    k_open    = _odd(int(round(min(H, W) * open_frac)))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # 4) Remove tiny blobs by area
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    min_area_frac = float(config.get('cleaner_min_area_frac', 1e-5))
    min_area      = max(9, int(min_area_frac * H * W))
    keep = np.zeros_like(bw)
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            keep[labels == i] = 255

    # 5) Large dilation to merge label parts into one region
    dilate_frac  = float(config.get('cleaner_dilate_frac', 0.12))
    kx           = _odd(int(round(dilate_frac * W)))
    ky           = _odd(int(round(dilate_frac * H)))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    dil_iters    = int(config.get('cleaner_dilate_iters', 1))
    merged       = cv2.dilate(keep, kernel_dilate, iterations=dil_iters)

    # 6) Bounding rect + padding
    pts = cv2.findNonZero(merged)
    if pts is None or len(pts) == 0:
        return None

    x, y, w, h = cv2.boundingRect(pts)
    pad = int(round(float(config.get('cleaner_pad_frac', 0.01)) * min(H, W)))
    x1  = max(0, x - pad)
    y1  = max(0, y - pad)
    x2  = min(W, x + w + pad)
    y2  = min(H, y + h + pad)
    return x1, y1, x2, y2


# ── Public API ─────────────────────────────────────────────────────────────

def clean_label(
    img: np.ndarray,
    config: dict,
    input_path: Optional[str] = None,
    page_index: Optional[int] = None,
) -> np.ndarray:
    """Crop the input image to the main dark-content region.

    Returns the cropped BGR image on success, or the original image on failure.
    """
    try:
        bbox = _detect_bbox(img, config)
        if bbox is None:
            return img

        x1, y1, x2, y2 = bbox
        crop = img[y1:y2, x1:x2].copy()

        # Optional debug output
        if config.get('store_debug_images', False):
            dbg_dir = config.get('debug_image_dir') or 'output/cleaned_images'
            _ensure_dir(dbg_dir)
            base      = os.path.splitext(os.path.basename(input_path or 'input'))[0]
            page_part = f"_page{page_index}" if page_index is not None else ''
            out_prefix = os.path.join(dbg_dir, f"{base}{page_part}")
            try:
                cv2.imwrite(out_prefix + '_final_cropped.png', crop)
            except Exception:
                pass

        return crop

    except Exception:
        logger.debug('clean_label failed, returning original image', exc_info=True)
        return img


def clean_label_vplane(
    img: np.ndarray,
    config: dict,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Return a grayscale V-plane crop and the bounding box (x1, y1, x2, y2).

    Useful for SIFT preprocessing where a consistent grayscale source is needed.
    Returns the full-image V-plane and a full-image bbox if detection fails.
    """
    H, W = img.shape[:2]
    V    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]

    bbox = _detect_bbox(img, config)
    if bbox is None:
        return V.copy(), (0, 0, W, H)

    x1, y1, x2, y2 = bbox
    return V[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)