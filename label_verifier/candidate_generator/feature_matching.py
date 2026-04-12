"""
Feature matching and point clustering for candidate generation.
Handles multi-detector matching, point rasterisation, and DBSCAN clustering.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .. import preprocessor
from . import debug_utils as debug

logger = logging.getLogger(__name__)

try:
    from .. import dbscan as dbscan_module
except Exception:
    logger.debug('dbscan module unavailable — will fall back to contours')
    dbscan_module = None

Box = Tuple[int, int, int, int]

# Detectors and their corresponding distance norms
_DETECTORS = {
    'sift':  (cv2.SIFT_create,  cv2.NORM_L2),
    'akaze': (cv2.AKAZE_create, cv2.NORM_HAMMING),
    'brisk': (cv2.BRISK_create, cv2.NORM_HAMMING),
}


def match_point_candidates(
    label_img: np.ndarray,
    ref_icon:  np.ndarray,
    config:    dict,
    debug_dir: Optional[str] = None,
) -> Tuple[List[Box], Dict]:
    """Generate candidate boxes using feature matching and point clustering.

    Runs SIFT, AKAZE, and BRISK detectors against the reference icon, collects
    match points on the label image, rasterises them into a binary mask, then
    clusters the mask into bounding boxes via DBSCAN (or contour fallback).

    Args:
        label_img : BGR page/label image to search.
        ref_icon  : BGR reference icon image.
        config    : Pipeline configuration dict.
        debug_dir : Optional directory for debug image output.

    Returns:
        (boxes, debug_info) — list of (x, y, w, h) candidates and a debug dict.
    """
    debug_info: Dict = {}
    if label_img is None or ref_icon is None:
        return [], debug_info

    # ── Grayscale conversion ───────────────────────────────────────────────
    try:
        page_gray = preprocessor.to_grayscale(label_img)
        icon_gray = preprocessor.to_grayscale(ref_icon)
    except Exception:
        logger.debug('Grayscale conversion failed', exc_info=True)
        return [], debug_info

    # ── Raster rectangle size (1 % of image dims by default) ──────────────
    rect_frac = float(config.get('match_point_rect_frac', 0.01))
    h_img, w_img = label_img.shape[:2]
    rect_w = max(1, int(round(w_img * rect_frac)))
    rect_h = max(1, int(round(h_img * rect_frac)))

    # ── Multi-detector matching ────────────────────────────────────────────
    pts_map: Dict[Tuple[int, int], set] = {}   # (x, y) → set of detector names
    all_match_points: List[Tuple[int, int]]    = []

    for det_name, (factory, norm) in _DETECTORS.items():
        try:
            det        = factory()
            kp_p, dp   = det.detectAndCompute(page_gray, None)
            kp_i, di   = det.detectAndCompute(icon_gray, None)
            if dp is None or di is None:
                continue
            bf      = cv2.BFMatcher(norm)
            matches = bf.match(di, dp)
            for m in matches:
                try:
                    pt = kp_p[m.trainIdx].pt
                    x, y = int(pt[0]), int(pt[1])
                    pts_map.setdefault((x, y), set()).add(det_name)
                    all_match_points.append((x, y))
                except Exception:
                    continue
        except Exception:
            logger.debug('Detector %s failed', det_name, exc_info=True)

    if not pts_map:
        return [], debug_info

    good_pts = [(x, y, frozenset(detset)) for (x, y), detset in pts_map.items()]

    # ── Rasterise match points into a binary mask ──────────────────────────
    point_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for item in good_pts:
        try:
            mx, my = int(item[0]), int(item[1])
            x0 = max(0,        mx - rect_w // 2)
            y0 = max(0,        my - rect_h // 2)
            x1 = min(w_img - 1, x0 + rect_w)
            y1 = min(h_img - 1, y0 + rect_h)
            if x1 > x0 and y1 > y0:
                cv2.rectangle(point_mask, (x0, y0), (x1, y1), 255, -1)
        except Exception:
            logger.debug('Point rasterisation failed for %s', item, exc_info=True)

    point_mask_bin = (point_mask > 0).astype(np.uint8) * 255

    # ── Cluster into bounding boxes ────────────────────────────────────────
    boxes: List[Box] = []
    try:
        if dbscan_module is not None:
            boxes = dbscan_module.cluster_binary(
                point_mask_bin,
                eps=float(config.get('dbscan_eps',         5.0)),
                min_samples=int(config.get('dbscan_min_samples', 8)),
                min_area=int(config.get('dbscan_min_area',      16)),
            )
        else:
            contours, _ = cv2.findContours(
                point_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            boxes = [cv2.boundingRect(c) for c in contours]
    except Exception:
        logger.debug('Clustering failed', exc_info=True)

    # ── Debug output ───────────────────────────────────────────────────────
    if debug_dir:
        _save_debug_output(
            label_img, boxes, all_match_points, good_pts,
            config, debug_dir, debug_info,
        )

    return boxes, debug_info


# ── Private debug helper ───────────────────────────────────────────────────

def _save_debug_output(
    label_img:        np.ndarray,
    boxes:            List[Box],
    all_match_points: List[Tuple[int, int]],
    good_pts:         list,
    config:           dict,
    debug_dir:        str,
    debug_info:       Dict,
) -> None:
    """Write debug visualisations for match points and clusters."""
    if not debug.ensure_dir(debug_dir):
        return

    seq = debug.next_debug_seq()

    # Store match points for callers
    try:
        debug_info['match_points'] = [(int(p[0]), int(p[1])) for p in good_pts]
    except Exception:
        pass

    # Cluster bounding box overlay
    try:
        vis = label_img.copy()
        debug.draw_boxes(vis, boxes, (255, 0, 0), 2)
        path = os.path.join(debug_dir, f'match_point_clusters_bbox_overlay_s{seq}.png')
        debug.safe_imwrite(path, vis)
        debug_info['point_clusters_bbox_overlay'] = path
        debug_info['bbox_overlay']                = path   # backward compat
    except Exception:
        logger.debug('Failed to save cluster overlay', exc_info=True)

    # Match points on edge image overlay
    try:
        edge_img = preprocessor.get_edges(
            label_img,
            low_thresh=float(config.get('edge_low_thresh',  50)),
            high_thresh=float(config.get('edge_high_thresh', 150)),
            dilate_iter=int(config.get('edge_dilate_iter',   1)),
            use_clahe=False,
        )
        if edge_img is not None and getattr(edge_img, 'size', 0) > 0:
            vis_e = (cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
                     if edge_img.ndim == 2 else edge_img.copy())
            for mx, my in all_match_points:
                try:
                    cv2.circle(vis_e, (mx, my), 2, (0, 0, 255), -1)
                except Exception:
                    pass
            path = os.path.join(debug_dir,
                                f'match_points_on_edges_s{seq}.png')
            debug.safe_imwrite(path, vis_e)
            debug_info['matches_on_edges_overlay'] = path
    except Exception:
        logger.debug('Failed to save match-points-on-edges overlay', exc_info=True)