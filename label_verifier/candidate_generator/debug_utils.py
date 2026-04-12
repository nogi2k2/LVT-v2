"""
Debug utilities for candidate generation.
Centralised sequence counting and safe file I/O operations.
"""

import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Global debug sequence counter — incremented per saved debug image set
_debug_seq: int = 0


def next_debug_seq() -> str:
    """Return the next zero-padded debug sequence number (e.g. '0001')."""
    global _debug_seq
    _debug_seq += 1
    return f"{_debug_seq:04d}"


def ensure_dir(path: str) -> bool:
    """Create directory (and parents) if it does not exist.

    Returns True on success, False on failure.
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception:
        logger.debug('ensure_dir failed for %s', path, exc_info=True)
        return False


def safe_imwrite(path: str, img: np.ndarray) -> bool:
    """Write an image to disk, returning True on success."""
    try:
        return cv2.imwrite(path, img)
    except Exception:
        logger.debug('safe_imwrite failed for %s', path, exc_info=True)
        return False


def draw_boxes(
    img:       np.ndarray,
    boxes:     List[Tuple[int, int, int, int]],
    color:     Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on an image in-place and return it.

    Args:
        img       : BGR numpy array to draw on.
        boxes     : List of (x, y, w, h) boxes.
        color     : BGR draw colour.
        thickness : Line thickness in pixels.
    """
    for box in boxes:
        try:
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        except Exception:
            logger.debug('draw_boxes: failed to draw box %s', box, exc_info=True)
    return img