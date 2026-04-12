from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ResultRecord:
    """
    Stores the result of verifying one icon on one page of one input file.

    Core fields (required):
        input_path  : Path to the input file.
        page_index  : Page index in the input file.
        icon_name   : Name of the reference icon.
        icon_snip   : Reference icon image (numpy array or PIL Image).
        match_snip  : Matched region image (numpy array or PIL Image).
        decision    : 'Pass' or 'Fail'.
        score       : Primary match score shown in the report.

    Optional diagnostic fields:
        comment           : Free-text comment.
        debug_image_path  : Path to a debug visualisation image on disk.
        good_matches      : Number of good descriptor matches.
        sim               : Structural similarity score (0–1).
        siglip            : SigLIP semantic similarity score (0–1).
        combined_score    : Weighted combination of all scores (0–1).
        kp_counts         : Keypoint counts dict with keys:
                            'kp_icon', 'kp_image', 'good_matches', 'inliers'.
        pattern_details   : Pattern verifier details dict with keys:
                            'topology_score', 'sift_good', 'akaze_good',
                            'brisk_good', and per-component scores.
    """
    # ── Required fields ───────────────────────────────────────────────────
    input_path: str
    page_index: int
    icon_name: str
    icon_snip: Any
    match_snip: Any
    decision: str
    score: float

    # ── Optional general fields ───────────────────────────────────────────
    comment: str = ""
    debug_image_path: Optional[str] = None
    good_matches: int = 0

    # ── Optional score breakdown ──────────────────────────────────────────
    sim: Optional[float] = None               # Structural similarity
    siglip: Optional[float] = None            # SigLIP semantic score
    combined_score: Optional[float] = None    # Weighted combined score

    # ── Optional diagnostic detail dicts ─────────────────────────────────
    kp_counts: Optional[Dict[str, Any]] = None
    pattern_details: Optional[Dict[str, Any]] = None


@dataclass
class RunSummary:
    """
    Stores summary information for a verification run.

    Fields:
        total   : Total number of icon-vs-page checks performed.
        passed  : Number of passing checks.
        failed  : Number of failing checks.
        results : All ResultRecord objects from the run.
    """
    total: int
    passed: int
    failed: int
    results: List[ResultRecord] = field(default_factory=list)