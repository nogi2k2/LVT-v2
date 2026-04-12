"""
Edge detection and border processing for candidate generation.
Handles Canny edge detection, CLAHE enhancement, border/frame heuristics,
and inner contour recovery.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .. import preprocessor
from . import debug_utils as debug

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]


def compute_edge_boxes(
    label_img: np.ndarray,
    config:    dict,
) -> List[Box]:
    """Compute edge-connected bounding boxes with border filtering.

    Args:
        label_img : BGR input image.
        config    : Pipeline configuration dict.

    Returns:
        List of (x, y, w, h) bounding boxes around detected edge regions.
    """
    edge_boxes: List[Box] = []
    contours_e            = []

    # ── Step 1: Edge detection ─────────────────────────────────────────────
    edge_img = None
    try:
        edge_img = preprocessor.get_edges(
            label_img,
            low_thresh=float(config.get('edge_low_thresh',  50)),
            high_thresh=float(config.get('edge_high_thresh', 150)),
            dilate_iter=int(config.get('edge_dilate_iter',   0)),
            use_clahe=False,
        )
    except Exception:
        logger.debug('get_edges failed', exc_info=True)

    # ── Step 2: Debug images ───────────────────────────────────────────────
    gray_img: Optional[np.ndarray] = None
    if config.get('debug_image_dir'):
        try:
            gray_img = preprocessor.to_grayscale(label_img)
        except Exception:
            logger.debug('to_grayscale failed for debug output', exc_info=True)
    _save_edge_debug_images(label_img, edge_img, config, gray=gray_img)

    # ── Step 3: Contour detection ──────────────────────────────────────────
    if edge_img is not None and getattr(edge_img, 'size', 0) > 0:
        try:
            cnts, _ = cv2.findContours(
                edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # Image dimensions for area/size thresholds
            gray_for_size = preprocessor.to_grayscale(label_img)
            H, W = (gray_for_size.shape[:2] if gray_for_size is not None
                    else edge_img.shape[:2])
            min_area  = max(10, 0.00005 * H * W)
            max_frac  = float(config.get('edge_max_box_fraction', 0.3))
            min_box_a = int(config.get('dbscan_min_area', 16))

            conts_f    = [c for c in cnts if cv2.contourArea(c) >= min_area]
            contours_e = conts_f

            for c in conts_f:
                x, y, w, h = cv2.boundingRect(c)
                if w * h < min_box_a:
                    continue
                if w > max_frac * W or h > max_frac * H:
                    continue
                edge_boxes.append((x, y, w, h))
        except Exception:
            logger.debug('Contour detection failed', exc_info=True)
            edge_boxes = []

    # ── Step 4: Debug overlays ─────────────────────────────────────────────
    _save_final_debug_overlays(label_img, edge_boxes, edge_img, contours_e, config)

    return edge_boxes


# ── Private debug helpers ──────────────────────────────────────────────────

def _save_edge_debug_images(
    label_img: np.ndarray,
    edge_img:  Optional[np.ndarray],
    config:    dict,
    gray:      Optional[np.ndarray] = None,
) -> None:
    """Save edge mask and grayscale debug images if debug_image_dir is set."""
    dbg_dir = config.get('debug_image_dir')
    if not dbg_dir or not debug.ensure_dir(dbg_dir):
        return

    seq = debug.next_debug_seq()

    if gray is not None:
        debug.safe_imwrite(os.path.join(dbg_dir, f'edge_gray_s{seq}.png'), gray)

    if edge_img is not None:
        mask = (edge_img > 0).astype('uint8') * 255
        debug.safe_imwrite(os.path.join(dbg_dir, f'edge_mask_s{seq}.png'), mask)


def _save_final_debug_overlays(
    label_img:  np.ndarray,
    edge_boxes: List[Box],
    edge_img:   Optional[np.ndarray],
    contours_e: list,
    config:     dict,
) -> None:
    """Save contour and bounding box overlays for debugging."""
    dbg_dir = config.get('debug_image_dir')
    if not dbg_dir or not debug.ensure_dir(dbg_dir):
        return

    seq = debug.next_debug_seq()

    # Contours + final boxes on original image
    try:
        vis = label_img.copy()
        for c in (contours_e or []):
            try:
                rx, ry, rw, rh = cv2.boundingRect(c)
                cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 1)
            except Exception:
                pass
        debug.draw_boxes(vis, edge_boxes, (0, 255, 0), 2)
        debug.safe_imwrite(
            os.path.join(dbg_dir, f'edge_contours_s{seq}.png'), vis
        )
    except Exception:
        logger.debug('Failed to save contour overlay', exc_info=True)

    # Contours on edge image
    if edge_img is not None and getattr(edge_img, 'size', 0) > 0:
        try:
            vis_e = (cv2.cvtColor((edge_img > 0).astype('uint8') * 255,
                                   cv2.COLOR_GRAY2BGR)
                     if edge_img.ndim == 2 else edge_img.copy())
            if contours_e:
                cv2.drawContours(vis_e, contours_e, -1, (0, 0, 255), 1)
            debug.safe_imwrite(
                os.path.join(dbg_dir, f'edge_all_contours_s{seq}.png'), vis_e
            )
        except Exception:
            logger.debug('Failed to save edge contour overlay', exc_info=True)