import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Basic image utilities ──────────────────────────────────────────────────

def preprocess(img: np.ndarray, config: dict) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Minimal preprocessing — returns the image unchanged with scale (1.0, 1.0).

    Heavy preprocessing is intentionally deferred to make_binary_from_vplane.
    """
    return img, (1.0, 1.0)


def to_grayscale(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Convert a BGR image to grayscale. Returns None on failure."""
    if img is None:
        return None
    try:
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.copy()
    except Exception:
        logger.debug('to_grayscale failed', exc_info=True)
        return None


def resize_for_match(
    img: Optional[np.ndarray],
    size: Tuple[int, int] = (128, 128),
) -> Optional[np.ndarray]:
    """Resize image to a canonical size for comparison (ignores aspect ratio).

    Returns a grayscale uint8 image of the requested size, or None on failure.
    """
    gray = to_grayscale(img)
    if gray is None:
        return None
    try:
        return cv2.resize(gray, (int(size[0]), int(size[1])),
                          interpolation=cv2.INTER_AREA)
    except Exception:
        logger.debug('resize_for_match failed', exc_info=True)
        try:
            return cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        except Exception:
            return gray


# ── Binary mask creation ───────────────────────────────────────────────────

def make_binary_from_vplane(vplane: np.ndarray, config: dict) -> np.ndarray:
    """Create a binary mask from a V (brightness) plane.

    Uses adaptive threshold parameters from config with safe fallbacks.
    Returns a single-channel uint8 image with foreground == 255.
    """
    if vplane is None or getattr(vplane, 'size', 0) == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    try:
        block_frac = float(config.get('cleaner_adapt_block_frac', 0.02))
        block = max(15, int(round(min(vplane.shape[:2]) * block_frac)))
        if block % 2 == 0:
            block += 1
        C = int(config.get('cleaner_adapt_C', 8))
        bin_img = cv2.adaptiveThreshold(
            vplane, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block, C,
        )
    except Exception:
        logger.debug('adaptiveThreshold failed, falling back to Otsu', exc_info=True)
        try:
            bin_img = cv2.threshold(
                vplane, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )[1]
        except Exception:
            bin_img = np.where(vplane > 128, 0, 255).astype(np.uint8)

    # Optional small opening to remove speckles
    try:
        open_frac = float(config.get('cleaner_open_frac', 0.0002))
        H, W      = vplane.shape[:2]
        k         = max(3, int(round(min(H, W) * open_frac)))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    except Exception:
        logger.debug('make_binary_from_vplane opening step failed', exc_info=True)

    return bin_img


def closing(bin_img: np.ndarray, ksize: int = 17) -> np.ndarray:
    """Apply morphological closing to merge nearby foreground regions.

    Returns the closed binary image, or the original on failure.
    """
    if bin_img is None or getattr(bin_img, 'size', 0) == 0:
        return bin_img
    try:
        k      = cv2.getStructuringElement(cv2.MORPH_RECT, (int(ksize), int(ksize)))
        return cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k)
    except Exception:
        logger.debug('closing failed', exc_info=True)
        return bin_img


# ── Edge utilities ─────────────────────────────────────────────────────────

def compute_edge_mask(
    img: Optional[np.ndarray],
    low_thresh:  int = 50,
    high_thresh: int = 150,
    dilate_iter: int = 1,
) -> Optional[np.ndarray]:
    """Return a binary edge mask (0/1 uint8) using Canny + optional dilation."""
    if img is None:
        return None
    try:
        gray  = to_grayscale(img)
        edges = cv2.Canny(gray, low_thresh, high_thresh)
        if dilate_iter > 0:
            k     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, k, iterations=dilate_iter)
        return (edges > 0).astype(np.uint8)
    except Exception:
        logger.debug('compute_edge_mask failed', exc_info=True)
        return None


def get_edges(
    img: Optional[np.ndarray],
    low_thresh:  int  = 50,
    high_thresh: int  = 150,
    dilate_iter: int  = 0,
    use_clahe:   bool = False,
) -> Optional[np.ndarray]:
    """Return a uint8 edge image (0 or 255).

    Lightweight preprocessing step before comparison. Optionally applies CLAHE
    to the grayscale image before running Canny.
    """
    if img is None:
        return None
    try:
        gray = to_grayscale(img)
        if gray is None:
            return None
        if use_clahe:
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray  = clahe.apply(gray)
            except Exception:
                logger.debug('CLAHE failed in get_edges', exc_info=True)
        edges = cv2.Canny(gray, int(low_thresh), int(high_thresh))
        if dilate_iter and int(dilate_iter) > 0:
            kel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kel, iterations=int(dilate_iter))
        return (edges > 0).astype(np.uint8) * 255
    except Exception:
        logger.debug('get_edges failed', exc_info=True)
        return None


def enhance_edges(
    img: Optional[np.ndarray],
    clahe_clip:      float = 2.0,
    clahe_tile:      Tuple[int, int] = (8, 8),
    bilateral_d:     int   = 5,
    bilateral_sigma: float = 75.0,
    unsharp_amount:  float = 1.5,
    gaussian_ksize:  int   = 3,
    canny_low:       int   = 50,
    canny_high:      int   = 150,
    dilate_iter:     int   = 1,
) -> Optional[np.ndarray]:
    """Enhance edges for improved matching.

    Pipeline:
      1. Convert to grayscale
      2. CLAHE for local contrast enhancement
      3. Bilateral filter to reduce noise while preserving edges
      4. Unsharp mask (sharpening)
      5. Canny edges + dilation to thicken edges
      6. Combine sharpened image with thick edges

    Returns a BGR uint8 image suitable for downstream comparison.
    Returns the original image unchanged if any step fails.
    """
    if img is None:
        return img
    try:
        gray = to_grayscale(img)
        if gray is None:
            return img

        # 1. CLAHE
        try:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                                    tileGridSize=clahe_tile)
            gray  = clahe.apply(gray)
        except Exception:
            logger.debug('enhance_edges CLAHE failed', exc_info=True)

        # 2. Bilateral filter
        try:
            gray = cv2.bilateralFilter(gray, bilateral_d,
                                       bilateral_sigma, bilateral_sigma)
        except Exception:
            logger.debug('enhance_edges bilateral filter failed', exc_info=True)

        # 3. Unsharp mask
        try:
            blurred = cv2.GaussianBlur(gray, (gaussian_ksize, gaussian_ksize), 0)
            gray    = cv2.addWeighted(gray, 1.0 + unsharp_amount,
                                      blurred, -unsharp_amount, 0)
        except Exception:
            logger.debug('enhance_edges unsharp mask failed', exc_info=True)

        # 4. Canny + dilation
        try:
            edges = cv2.Canny(gray, canny_low, canny_high)
            if dilate_iter > 0:
                k     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                edges = cv2.dilate(edges, k, iterations=dilate_iter)
        except Exception:
            logger.debug('enhance_edges Canny failed', exc_info=True)
            edges = np.zeros_like(gray)

        # 5. Combine sharpened gray with edges → BGR output
        combined = cv2.addWeighted(gray, 0.7, edges, 0.3, 0)
        return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

    except Exception:
        logger.debug('enhance_edges failed, returning original', exc_info=True)
        return img


def compute_edge_iou(
    a: Optional[np.ndarray],
    b: Optional[np.ndarray],
    low_thresh:  int = 50,
    high_thresh: int = 150,
    dilate_iter: int = 1,
) -> float:
    """Compute IoU between edge masks of two images. Returns 0.0 on failure."""
    try:
        em1 = compute_edge_mask(a, low_thresh, high_thresh, dilate_iter)
        em2 = compute_edge_mask(b, low_thresh, high_thresh, dilate_iter)
        if em1 is None or em2 is None:
            return 0.0
        if em1.shape != em2.shape:
            em2 = cv2.resize(em2, (em1.shape[1], em1.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        inter = np.logical_and(em1, em2).sum()
        union = np.logical_or(em1,  em2).sum()
        return float(inter) / float(union) if union > 0 else 0.0
    except Exception:
        logger.debug('compute_edge_iou failed', exc_info=True)
        return 0.0


# ── Topology / skeleton ────────────────────────────────────────────────────

def _skeletonize(bin_img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Morphological skeleton via iterative erosion/opening.

    Input : binary image with values 0 or 1 (or 0/255).
    Returns: binary skeleton (0/1 uint8), or None on failure.
    """
    if bin_img is None:
        return None
    try:
        img     = (bin_img > 0).astype(np.uint8) * 255
        skel    = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(img, element)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
            skel   = cv2.bitwise_or(skel, cv2.subtract(eroded, opened))
            img    = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return (skel > 0).astype(np.uint8)
    except Exception:
        logger.debug('_skeletonize failed', exc_info=True)
        return None


def compute_topology_score(
    a_edges: Optional[np.ndarray],
    b_edges: Optional[np.ndarray],
    angle_bins: int = 12,
) -> float:
    """Topology similarity score between two edge masks (0.0–1.0).

    Uses skeletonisation to extract stroke structure, counts holes via contour
    hierarchy, counts endpoints & junctions, and builds a stroke-angle histogram.
    Returns 1.0 when topology is identical, 0.0 on failure.
    """
    if a_edges is None or b_edges is None:
        return 0.0
    try:
        ae = (a_edges > 0).astype(np.uint8)
        be = (b_edges > 0).astype(np.uint8)

        def _hole_object_counts(bin_img: np.ndarray) -> Tuple[int, int]:
            img_u8 = (bin_img * 255).astype(np.uint8)
            try:
                contours, hierarchy = cv2.findContours(
                    img_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
                )
            except Exception:
                return 0, 0
            if hierarchy is None or len(contours) == 0:
                return 0, 0
            hierarchy = hierarchy[0]
            objs  = sum(1 for h in hierarchy if int(h[3]) == -1)
            holes = sum(1 for h in hierarchy if int(h[3]) != -1)
            return objs, holes

        _, a_holes = _hole_object_counts(ae)
        _, b_holes = _hole_object_counts(be)
        hole_score = 1.0 / (1.0 + float(abs(a_holes - b_holes)))

        # Skeleton endpoints & junctions
        a_skel = _skeletonize(ae)
        b_skel = _skeletonize(be)
        if a_skel is None or b_skel is None:
            end_junc_score = 0.0
        else:
            kern = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
            a_nb = cv2.filter2D(a_skel, -1, kern).astype(np.uint8)
            b_nb = cv2.filter2D(b_skel, -1, kern).astype(np.uint8)
            diff_ej = (
                abs(int(((a_skel == 1) & (a_nb == 1)).sum()) -
                    int(((b_skel == 1) & (b_nb == 1)).sum())) +
                abs(int(((a_skel == 1) & (a_nb >= 3)).sum()) -
                    int(((b_skel == 1) & (b_nb >= 3)).sum()))
            )
            end_junc_score = 1.0 / (1.0 + float(diff_ej))

        # Stroke angle histogram intersection
        def _angle_hist(skel: Optional[np.ndarray]) -> np.ndarray:
            if skel is None:
                return np.zeros(angle_bins, dtype=np.float32)
            offsets = [(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0)]
            ox = np.array([o[0] for o in offsets], dtype=np.int8)
            oy = np.array([o[1] for o in offsets], dtype=np.int8)
            ys, xs = np.where(skel == 1)
            angles = []
            for y, x in zip(ys, xs):
                vx = vy = 0.0
                for dx, dy in zip(ox, oy):
                    nx, ny = x + int(dx), y + int(dy)
                    if 0 <= nx < skel.shape[1] and 0 <= ny < skel.shape[0]:
                        if skel[ny, nx] == 1:
                            vx += float(dx)
                            vy += float(dy)
                if vx != 0.0 or vy != 0.0:
                    angles.append(np.arctan2(vy, vx))
            if not angles:
                return np.zeros(angle_bins, dtype=np.float32)
            angs = (np.array(angles) + np.pi) % (2 * np.pi)
            hist, _ = np.histogram(angs, bins=angle_bins, range=(0, 2 * np.pi))
            hist = hist.astype(np.float32)
            s = hist.sum()
            return hist / float(s) if s > 0 else hist

        ang_inter = float(np.minimum(_angle_hist(a_skel), _angle_hist(b_skel)).sum())

        score = 0.35 * ang_inter + 0.35 * hole_score + 0.30 * end_junc_score
        return float(max(0.0, min(1.0, score)))
    except Exception:
        logger.debug('compute_topology_score failed', exc_info=True)
        return 0.0


# ── Similarity metrics ─────────────────────────────────────────────────────

def compute_ssim(
    a: Optional[np.ndarray],
    b: Optional[np.ndarray],
    ksize: int   = 11,
    sigma: float = 1.5,
) -> float:
    """Single-scale SSIM (mean over image) between two uint8 grayscale images.

    Returns a float in [0, 1] for similar images. Returns 0.0 on failure.
    """
    if a is None or b is None:
        return 0.0
    try:
        a_f = a.astype(np.float32)
        b_f = b.astype(np.float32)
        L   = 255.0
        C1  = (0.01 * L) ** 2
        C2  = (0.03 * L) ** 2
        gauss  = cv2.getGaussianKernel(ksize, sigma)
        window = gauss @ gauss.T
        s  = ksize // 2
        sl = slice(s, -s or None)

        def _filt(x: np.ndarray) -> np.ndarray:
            return cv2.filter2D(x, -1, window)[sl, sl]

        mu1    = _filt(a_f)
        mu2    = _filt(b_f)
        num    = (2 * mu1 * mu2 + C1) * (2 * (_filt(a_f * b_f) - mu1 * mu2) + C2)
        den    = (mu1**2 + mu2**2 + C1) * (
            _filt(a_f * a_f) - mu1**2 + _filt(b_f * b_f) - mu2**2 + C2
        )
        return float(np.mean(num / (den + 1e-12)))
    except Exception:
        logger.debug('compute_ssim fell back to NCC', exc_info=True)
        try:
            a_f = a.astype(np.float32).ravel()
            b_f = b.astype(np.float32).ravel()
            na, nb = np.linalg.norm(a_f), np.linalg.norm(b_f)
            return float(np.dot(a_f, b_f) / (na * nb)) if na and nb else 0.0
        except Exception:
            return 0.0


def compute_grad_ncc(
    a: Optional[np.ndarray],
    b: Optional[np.ndarray],
) -> float:
    """Normalized cross-correlation of image gradients.

    Returns a scalar (may be negative); caller may clip to [0, 1].
    """
    if a is None or b is None:
        return 0.0
    try:
        ga = to_grayscale(a).astype(np.float32)
        gb = to_grayscale(b).astype(np.float32)
        if ga.shape != gb.shape:
            gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]),
                            interpolation=cv2.INTER_AREA)
        ax = cv2.Sobel(ga, cv2.CV_32F, 1, 0, ksize=3)
        ay = cv2.Sobel(ga, cv2.CV_32F, 0, 1, ksize=3)
        bx = cv2.Sobel(gb, cv2.CV_32F, 1, 0, ksize=3)
        by = cv2.Sobel(gb, cv2.CV_32F, 0, 1, ksize=3)
        va = np.concatenate((ax.ravel(), ay.ravel()))
        vb = np.concatenate((bx.ravel(), by.ravel()))
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        return float(np.dot(va, vb) / (na * nb)) if na and nb else 0.0
    except Exception:
        logger.debug('compute_grad_ncc failed', exc_info=True)
        return 0.0


# ── High-level comparators ─────────────────────────────────────────────────

def compare_icons_pair(
    img1: Optional[np.ndarray],
    img2: Optional[np.ndarray],
    size:        Tuple[int, int] = (128, 128),
    edges_only:  bool            = False,
    edge_params: Optional[dict]  = None,
) -> Dict[str, Any]:
    """High-level comparator: SSIM, EdgeIoU, GradNCC, classical and combined.

    Returns a dict with keys:
        ssim, edge_iou, grad_ncc, grad_ncc_clipped, combined,
        classical, combined_with_classical,
        classical_tmpl, classical_kp,
        feat_method, feat_kp1, feat_kp2, feat_matches, feat_good_matches,
        feat_good_ratio, feat_inliers, feat_inlier_ratio, feat_score
    """
    if edges_only:
        ep     = edge_params or {}
        a_proc = resize_for_match(
            get_edges(img1, ep.get('low_thresh', 50), ep.get('high_thresh', 150),
                      ep.get('dilate_iter', 1), ep.get('use_clahe', False)),
            size=size,
        )
        b_proc = resize_for_match(
            get_edges(img2, ep.get('low_thresh', 50), ep.get('high_thresh', 150),
                      ep.get('dilate_iter', 1), ep.get('use_clahe', False)),
            size=size,
        )
    else:
        a_proc = resize_for_match(enhance_edges(img1) or img1, size=size)
        b_proc = resize_for_match(enhance_edges(img2) or img2, size=size)

    ssim_v      = compute_ssim(a_proc, b_proc)
    edge_v      = compute_edge_iou(a_proc, b_proc)
    grad_v      = compute_grad_ncc(a_proc, b_proc)
    grad_clipped = float(max(0.0, min(1.0, grad_v)))
    combined    = 0.5 * ssim_v + 0.3 * edge_v + 0.2 * grad_clipped

    try:
        cls_comps = compute_classical_components(img1, img2, size=size)
    except Exception:
        logger.debug('compute_classical_components failed', exc_info=True)
        cls_comps = {'tmpl_score': 0.0, 'kp_score': 0.0, 'classical': 0.0}
    classical = float(cls_comps.get('classical', 0.0))

    try:
        feat = compute_feature_matching_components(img1, img2)
    except Exception:
        logger.debug('compute_feature_matching_components failed', exc_info=True)
        feat = {}

    return {
        'ssim':                    float(ssim_v),
        'edge_iou':                float(edge_v),
        'grad_ncc':                float(grad_v),
        'grad_ncc_clipped':        grad_clipped,
        'combined':                float(combined),
        'classical':               classical,
        'combined_with_classical': float(0.5 * combined + 0.5 * classical),
        'classical_tmpl':          float(cls_comps.get('tmpl_score', 0.0)),
        'classical_kp':            float(cls_comps.get('kp_score',   0.0)),
        'feat_method':             feat.get('method', ''),
        'feat_kp1':                float(feat.get('num_kp1',               0)),
        'feat_kp2':                float(feat.get('num_kp2',               0)),
        'feat_matches':            float(feat.get('num_matches',           0)),
        'feat_good_matches':       float(feat.get('num_good_matches',      0)),
        'feat_good_ratio':         float(feat.get('good_ratio',          0.0)),
        'feat_inliers':            float(feat.get('homography_inliers',    0)),
        'feat_inlier_ratio':       float(feat.get('homography_inlier_ratio', 0.0)),
        'feat_score':              float(feat.get('final_score',         0.0)),
    }


def compute_classical_components(
    img1: Optional[np.ndarray],
    img2: Optional[np.ndarray],
    size: Tuple[int, int] = (128, 128),
) -> Dict[str, float]:
    """Template matching + ORB keypoint score.

    Returns: {'tmpl_score': float, 'kp_score': float, 'classical': float}
    """
    try:
        a = resize_for_match(img1, size=size)
        b = resize_for_match(img2, size=size)
        if a is None or b is None:
            return {'tmpl_score': 0.0, 'kp_score': 0.0, 'classical': 0.0}

        # Template matching
        try:
            res = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            tmpl_score = float((max_val + 1.0) / 2.0)   # map [-1,1] → [0,1]
        except Exception:
            logger.debug('template matching failed', exc_info=True)
            tmpl_score = 0.0

        # ORB keypoint matching
        try:
            orb       = cv2.ORB_create(nfeatures=500)
            kp1, des1 = orb.detectAndCompute(a, None)
            kp2, des2 = orb.detectAndCompute(b, None)
            if des1 is None or des2 is None or not kp1 or not kp2:
                kp_score = 0.0
            else:
                bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(des1, des2, k=2)
                good    = sum(
                    1 for m_n in matches
                    if len(m_n) == 2 and m_n[0].distance < 0.75 * m_n[1].distance
                )
                kp_score = float(good) / float(max(1, min(len(kp1), len(kp2))))
        except Exception:
            logger.debug('ORB matching failed', exc_info=True)
            kp_score = 0.0

        classical = float(max(0.0, min(1.0, 0.6 * tmpl_score + 0.4 * kp_score)))
        return {'tmpl_score': tmpl_score, 'kp_score': kp_score, 'classical': classical}
    except Exception:
        logger.debug('compute_classical_components failed', exc_info=True)
        return {'tmpl_score': 0.0, 'kp_score': 0.0, 'classical': 0.0}


def compute_feature_matching_components(
    img1:                Optional[np.ndarray],
    img2:                Optional[np.ndarray],
    size:                Tuple[int, int] = (256, 256),
    method:              str             = 'orb',
    ratio_thresh:        float           = 0.75,
    ransac_thresh:       float           = 5.0,
    max_matches_normalize: int           = 50,
) -> Dict[str, Any]:
    """Feature matching (SIFT / AKAZE / BRISK / ORB) with RANSAC homography.

    Returns a dict with keys:
        method, num_kp1, num_kp2, num_matches, num_good_matches,
        good_ratio, homography_inliers, homography_inlier_ratio, final_score
    """
    _empty = {
        'method': method, 'num_kp1': 0, 'num_kp2': 0,
        'num_matches': 0, 'num_good_matches': 0, 'good_ratio': 0.0,
        'homography_inliers': 0, 'homography_inlier_ratio': 0.0, 'final_score': 0.0,
    }
    try:
        a = resize_for_match(img1, size=size)
        b = resize_for_match(img2, size=size)
        if a is None or b is None:
            return _empty

        method_l       = (method or 'orb').lower()
        is_float_desc  = False
        try:
            if method_l == 'sift':
                detector      = cv2.SIFT_create()
                is_float_desc = True
            elif method_l == 'akaze':
                detector = cv2.AKAZE_create()
            elif method_l == 'brisk':
                detector = cv2.BRISK_create()
            else:
                detector = cv2.ORB_create(nfeatures=1000)
        except Exception:
            logger.debug('Detector creation failed, using ORB', exc_info=True)
            detector      = cv2.ORB_create(nfeatures=1000)
            is_float_desc = False

        kp1, des1 = detector.detectAndCompute(a, None)
        kp2, des2 = detector.detectAndCompute(b, None)
        n1 = len(kp1) if kp1 else 0
        n2 = len(kp2) if kp2 else 0

        if des1 is None or des2 is None or n1 == 0 or n2 == 0:
            return {**_empty, 'num_kp1': n1, 'num_kp2': n2}

        # Matcher
        if is_float_desc:
            index_params  = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        matches     = matcher.knnMatch(des1, des2, k=2)
        good        = [m for m_n in matches
                       if len(m_n) == 2 and m_n[0].distance < ratio_thresh * m_n[1].distance
                       for m in [m_n[0]]]
        num_good    = len(good)
        good_ratio  = float(num_good) / float(max(1, min(n1, n2)))

        # RANSAC homography
        homography_inliers = 0
        inlier_ratio       = 0.0
        if num_good >= 4:
            try:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
                if mask is not None:
                    homography_inliers = int(mask.sum())
                    inlier_ratio       = float(homography_inliers) / float(num_good)
            except Exception:
                logger.debug('findHomography failed', exc_info=True)

        match_norm  = float(min(num_good, max_matches_normalize)) / float(max_matches_normalize)
        final_score = float(max(0.0, min(1.0, 0.6 * inlier_ratio + 0.4 * match_norm)))

        return {
            'method':                  method_l,
            'num_kp1':                 int(n1),
            'num_kp2':                 int(n2),
            'num_matches':             int(len(matches)),
            'num_good_matches':        int(num_good),
            'good_ratio':              float(good_ratio),
            'homography_inliers':      int(homography_inliers),
            'homography_inlier_ratio': float(inlier_ratio),
            'final_score':             float(final_score),
        }
    except Exception:
        logger.debug('compute_feature_matching_components failed', exc_info=True)
        return _empty


def compute_classical_score(
    img1: Optional[np.ndarray],
    img2: Optional[np.ndarray],
    size: Tuple[int, int] = (128, 128),
) -> float:
    """Backward-compatible wrapper — returns only the combined classical score."""
    try:
        return float(compute_classical_components(img1, img2, size=size).get('classical', 0.0))
    except Exception:
        return 0.0