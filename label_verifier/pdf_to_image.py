import logging
import os
from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image

# PyMuPDF is an optional dependency — import lazily so that importing this
# module doesn't fail when PyMuPDF is not installed.
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

logger = logging.getLogger(__name__)

# Supported direct-load image extensions
_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def pdf_to_images(
    input_path:      str,
    dpi:             int            = 300,
    return_pil:      bool           = False,
    store_converted: bool           = False,
    debug_dir:       Optional[str]  = None,
    max_pages:       Optional[int]  = None,
) -> List[Union[Image.Image, np.ndarray]]:
    """Convert a PDF (or image file) to a list of page images.

    Args:
        input_path      : Path to a PDF or image file (.png/.jpg/.tif etc.).
        dpi             : Rendering resolution for PDF pages.
        return_pil      : If True, return PIL.Image objects; otherwise BGR numpy arrays.
        store_converted : If True, save each page image to debug_dir for inspection.
                          Pass the value of config['store_debug_images'] here.
        debug_dir       : Directory for debug images (used when store_converted=True).
        max_pages       : Maximum number of PDF pages to render (None = all).

    Returns:
        List of PIL.Image objects (if return_pil=True) or BGR numpy arrays.

    Raises:
        RuntimeError : If PyMuPDF is not installed and a PDF is supplied.
        Exception    : On unreadable files — callers should handle this.
    """
    ext = os.path.splitext(input_path)[1].lower()

    # ── Image file ─────────────────────────────────────────────────────────
    if ext in _IMAGE_EXTS:
        try:
            with Image.open(input_path) as img:
                img = img.convert('RGB')
                _maybe_store(img,
                             os.path.splitext(os.path.basename(input_path))[0],
                             0, store_converted, debug_dir)
                if return_pil:
                    return [img.copy()]
                arr = np.array(img)
                return [cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)]
        except Exception:
            logger.exception('Failed to load image file: %s', input_path)
            raise

    # ── PDF file ───────────────────────────────────────────────────────────
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF (fitz) is required to convert PDFs to images. "
            "Install it with: pip install pymupdf"
        )

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    images: List[Union[Image.Image, np.ndarray]] = []
    doc = None
    try:
        doc = fitz.open(input_path)
        for i, page in enumerate(doc):
            if max_pages is not None and i >= max_pages:
                break
            mat     = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix     = page.get_pixmap(matrix=mat, alpha=False)
            mode    = 'RGBA' if pix.alpha else 'RGB'
            img_pil = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            _maybe_store(img_pil, base_name, i, store_converted, debug_dir)
            if return_pil:
                images.append(img_pil.copy())
            else:
                arr = np.array(img_pil.convert('RGB'))
                images.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    except Exception:
        logger.exception('Failed to convert PDF to images: %s', input_path)
        raise
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                logger.exception('Failed to close PDF document: %s', input_path)

    return images


# ── Private helpers ────────────────────────────────────────────────────────

def _maybe_store(
    img_pil:         Image.Image,
    base_name:       str,
    page_idx:        int,
    store_converted: bool,
    debug_dir:       Optional[str],
) -> None:
    """Save a page image to debug_dir if store_converted is True."""
    if not store_converted or not debug_dir:
        return
    try:
        os.makedirs(debug_dir, exist_ok=True)
        out_path = os.path.join(debug_dir, f"{base_name}_page{page_idx + 1}.png")
        img_pil.save(out_path)
        logger.debug('Saved converted image: %s', out_path)
    except Exception:
        logger.exception('Failed to save debug converted image')