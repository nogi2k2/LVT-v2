"""
Box utilities for candidate generation.
Pure numeric operations — no image processing dependencies.
"""

from typing import List, Tuple

# Type alias for a bounding box (x, y, w, h)
Box = Tuple[int, int, int, int]


def iou(a: Box, b: Box) -> float:
    """Calculate Intersection over Union for two boxes (x, y, w, h)."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter  = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


def rect_gap(a: Box, b: Box) -> int:
    """Calculate minimum gap between two boxes (x, y, w, h)."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    x_gap = max(0, bx1 - ax2) if bx1 > ax2 else max(0, ax1 - bx2)
    y_gap = max(0, by1 - ay2) if by1 > ay2 else max(0, ay1 - by2)
    return max(x_gap, y_gap)


def intersection_area(a: Box, b: Box) -> int:
    """Calculate intersection area between two boxes (x, y, w, h)."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def merge_boxes_simple(boxes: List[Box], iou_thresh: float = 0.0) -> List[Box]:
    """Merge overlapping boxes whose IoU exceeds iou_thresh."""
    boxes = list(boxes)
    changed = True
    while changed and len(boxes) > 1:
        changed  = False
        new_boxes: List[Box] = []
        used      = [False] * len(boxes)
        for i, bi in enumerate(boxes):
            if used[i]:
                continue
            merged = bi
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if iou(merged, boxes[j]) > iou_thresh:
                    x1 = min(merged[0], boxes[j][0])
                    y1 = min(merged[1], boxes[j][1])
                    x2 = max(merged[0] + merged[2], boxes[j][0] + boxes[j][2])
                    y2 = max(merged[1] + merged[3], boxes[j][1] + boxes[j][3])
                    merged    = (x1, y1, x2 - x1, y2 - y1)
                    used[j]   = True
                    changed   = True
            used[i] = True
            new_boxes.append(merged)
        boxes = new_boxes
    return boxes


def merge_close(
    boxes:      List[Box],
    iou_thresh: float = 0.05,
    gap_thresh: int   = 6,
) -> List[Box]:
    """Merge boxes that overlap or are within gap_thresh pixels of each other."""
    boxes = list(boxes)
    changed = True
    while changed and len(boxes) > 1:
        changed   = False
        new_boxes: List[Box] = []
        used       = [False] * len(boxes)
        for i, bi in enumerate(boxes):
            if used[i]:
                continue
            merged = bi
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if (iou(merged, boxes[j]) >= iou_thresh
                        or rect_gap(merged, boxes[j]) <= gap_thresh):
                    x1 = min(merged[0], boxes[j][0])
                    y1 = min(merged[1], boxes[j][1])
                    x2 = max(merged[0] + merged[2], boxes[j][0] + boxes[j][2])
                    y2 = max(merged[1] + merged[3], boxes[j][1] + boxes[j][3])
                    merged  = (x1, y1, x2 - x1, y2 - y1)
                    used[j] = True
                    changed = True
            used[i] = True
            new_boxes.append(merged)
        boxes = new_boxes
    return boxes