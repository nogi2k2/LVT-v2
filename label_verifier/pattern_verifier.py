import logging
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from . import preprocessor

logger = logging.getLogger(__name__)

# Weight distributed equally across the three detectors (total 0.8)
_PER_DETECTOR_WEIGHT = 0.8 / 3.0


def _match_comp(n_good: int, cap: int) -> float:
    """Map a raw good-match count to a weighted score component."""
    try:
        return min(int(n_good), cap) / float(cap) * _PER_DETECTOR_WEIGHT
    except Exception:
        return 0.0


def compute_combined_score(
    crop_bgr: np.ndarray,
    icon_bgr: np.ndarray,
    cap: int = 25,
) -> Tuple[float, Dict[str, Any]]:
    """Compute the combined confidence score for a candidate crop vs. a reference icon.

    Components (weights sum to 1.0):
        SIFT  good-match count  mapped 0..cap  → weight 0.8 / 3
        AKAZE good-match count  mapped 0..cap  → weight 0.8 / 3
        BRISK good-match count  mapped 0..cap  → weight 0.8 / 3
        Topology score from edge masks         → weight 0.20

    Args:
        crop_bgr : Candidate region as a BGR numpy array.
        icon_bgr : Reference icon as a BGR numpy array.
        cap      : Good-match count that maps to full detector weight.

    Returns:
        (combined_score, details_dict)
        combined_score : float in [0, 1]
        details_dict   : per-component breakdown for reporting
    """
    try:
        # ── Feature detectors ─────────────────────────────────────────────
        dets: Dict[str, Dict[str, Any]] = {}
        for method in ('sift', 'akaze', 'brisk'):
            try:
                result = preprocessor.compute_feature_matching_components(
                    crop_bgr, icon_bgr, method=method
                )
                dets[method] = result or {}
            except Exception:
                logger.debug('Detector %s failed', method, exc_info=True)
                dets[method] = {'num_good_matches': 0, 'final_score': 0.0}

        sift_good  = int(dets['sift'].get('num_good_matches',  0))
        akaze_good = int(dets['akaze'].get('num_good_matches', 0))
        brisk_good = int(dets['brisk'].get('num_good_matches', 0))

        sift_comp  = _match_comp(sift_good,  cap)
        akaze_comp = _match_comp(akaze_good, cap)
        brisk_comp = _match_comp(brisk_good, cap)

        # ── Topology score ────────────────────────────────────────────────
        topo_score = 0.0
        try:
            a_edges    = preprocessor.compute_edge_mask(crop_bgr)
            b_edges    = preprocessor.compute_edge_mask(icon_bgr)
            topo_score = float(preprocessor.compute_topology_score(a_edges, b_edges))
        except Exception:
            logger.debug('Topology score failed', exc_info=True)

        topo_comp = 0.20 * topo_score
        combined  = sift_comp + akaze_comp + brisk_comp + topo_comp

        details: Dict[str, Any] = {
            'topology_score': topo_score,
            'topology_comp':  topo_comp,
            'sift_good':      sift_good,
            'sift_comp':      sift_comp,
            'akaze_good':     akaze_good,
            'akaze_comp':     akaze_comp,
            'brisk_good':     brisk_good,
            'brisk_comp':     brisk_comp,
        }
        return float(combined), details

    except Exception:
        logger.debug('compute_combined_score failed', exc_info=True)
        return 0.0, {}