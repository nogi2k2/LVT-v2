"""
gui_shared.py — Framework-agnostic helpers shared by all GUI frontends.

Both main_gui.py (Tkinter) and gui_qt.py (PySide6) and streamlit_app.py
import from here instead of duplicating logic.
"""

import logging
import os
import subprocess
import sys
from typing import List, Optional

logger = logging.getLogger(__name__)

# Supported icon image extensions
_ICON_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


# ── Icon library ───────────────────────────────────────────────────────────

def list_icon_paths(icon_dir: str) -> List[str]:
    """Return a sorted list of icon image paths from icon_dir.

    Lookup order:
      1. Subdirectory layout: each subfolder containing an icon.png
         (e.g. icon_dir/CE_Mark/icon.png)
      2. Flat layout: image files directly inside icon_dir

    Args:
        icon_dir: Path to the icon library directory.

    Returns:
        List of absolute paths to icon images, sorted alphabetically.
        Returns an empty list if the directory doesn't exist or is unreadable.
    """
    if not icon_dir or not os.path.isdir(icon_dir):
        logger.debug('list_icon_paths: directory not found: %s', icon_dir)
        return []

    try:
        entries = sorted(os.listdir(icon_dir))
    except Exception:
        logger.debug('list_icon_paths: failed to list %s', icon_dir, exc_info=True)
        return []

    # 1. Subdirectory layout (preferred)
    paths = [
        os.path.join(icon_dir, e, 'icon.png')
        for e in entries
        if os.path.isdir(os.path.join(icon_dir, e))
        and os.path.exists(os.path.join(icon_dir, e, 'icon.png'))
    ]

    # 2. Flat layout fallback
    if not paths:
        paths = [
            os.path.join(icon_dir, e)
            for e in entries
            if os.path.isfile(os.path.join(icon_dir, e))
            and os.path.splitext(e)[1].lower() in _ICON_EXTS
        ]

    return paths


# ── Result formatting ──────────────────────────────────────────────────────

def format_result_row(r) -> dict:
    """Return a display-ready dict for a ResultRecord.

    Used by all GUI frontends to populate result tables consistently.

    Returns:
        dict with keys: filename, icon, decision, score
    """
    return {
        'filename': os.path.basename(getattr(r, 'input_path', '') or ''),
        'icon':     str(getattr(r, 'icon_name', '') or ''),
        'decision': str(getattr(r, 'decision',  '') or ''),
        'score':    f"{float(getattr(r, 'score', 0.0) or 0.0):.3f}",
    }


def result_pass_fail_counts(results: list) -> tuple:
    """Return (passed, failed, total) counts from a list of ResultRecords."""
    passed = sum(1 for r in results if getattr(r, 'decision', '') == 'Pass')
    total  = len(results)
    return passed, total - passed, total


def format_summary_text(results: list) -> str:
    """Return a human-readable summary string, e.g. '3/5 tests passed'."""
    passed, _, total = result_pass_fail_counts(results)
    return f"{passed}/{total} tests passed"


# ── Progress calculation ───────────────────────────────────────────────────

def progress_percent(status: dict) -> int:
    """Convert a progress status dict to an integer percentage (0–100)."""
    total   = max(status.get('total',   1), 1)
    current = status.get('current', 0)
    return int(min(max(100 * current // total, 0), 100))


# ── Report opening ─────────────────────────────────────────────────────────

def find_latest_report(output_dir: str) -> Optional[str]:
    """Return the path to the most recently modified PDF in output_dir, or None."""
    try:
        pdfs = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.lower().endswith('.pdf')
        ]
        return max(pdfs, key=os.path.getmtime) if pdfs else None
    except Exception:
        logger.debug('find_latest_report failed', exc_info=True)
        return None


def open_report(output_dir: str) -> Optional[str]:
    """Open the most recent PDF report in the system's default viewer.

    Returns the path that was opened, or None on failure.
    Works on Windows (os.startfile), macOS (open), and Linux (xdg-open).
    """
    path = find_latest_report(output_dir)
    if path is None:
        logger.warning('open_report: no PDF found in %s', output_dir)
        return None
    try:
        if sys.platform == 'win32':
            os.startfile(path)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['xdg-open', path])
        logger.info('Opened report: %s', path)
        return path
    except Exception:
        logger.exception('open_report: failed to open %s', path)
        return None


# ── Image conversion for GUI display ──────────────────────────────────────

def bgr_array_to_pil(arr, thumb_size: Optional[tuple] = None):
    """Convert a BGR numpy array to a PIL Image, optionally thumbnailed.

    Args:
        arr        : BGR numpy array or PIL Image.
        thumb_size : Optional (width, height) to thumbnail to.

    Returns:
        PIL.Image in RGB mode, or None on failure.
    """
    if arr is None:
        return None
    try:
        from PIL import Image
        import numpy as np

        if isinstance(arr, Image.Image):
            pil = arr.convert('RGB')
        elif isinstance(arr, np.ndarray):
            if arr.ndim == 3 and arr.shape[2] == 3:
                pil = Image.fromarray(arr[:, :, ::-1])   # BGR → RGB
            else:
                pil = Image.fromarray(arr)
        else:
            return None

        if thumb_size:
            pil.thumbnail(thumb_size, Image.LANCZOS)
        return pil
    except Exception:
        logger.debug('bgr_array_to_pil failed', exc_info=True)
        return None