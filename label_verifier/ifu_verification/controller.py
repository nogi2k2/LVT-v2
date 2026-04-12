"""IFU verification controller.

Implements a simple IFU verification workflow:

1. Extract images from input pages (PDF or image files). Each extracted image is named
   using a serial-like naming convention including page index and location index.
2. Create SigLIP embeddings for each extracted IFU image.
3. Create SigLIP embedding for the reference icon(s) and compare against all IFU embeddings.
4. Choose best candidate by highest SigLIP similarity. If similarity > threshold (default 0.85), mark Pass.

Notes:
- SigLIP cosine similarity ranges from -1.0..1.0 so the threshold is 0.85 by default. The user
  requested "8.5"; we assume that was a typo and use 0.85. This can be overridden in config via
  key 'ifu_siglip_threshold'.
"""
import logging
import os
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2
import os
from datetime import datetime

from .. import pdf_to_image
from ..siglip_encoder import create_siglip_encoder
from .. import reporter
from .. import models


class IFUController:
    """Controller for IFU verification workflows.

    Public methods:
    - __init__(config, progress=None)
    - run(input_paths, icon_path, output_path=None) -> (results_list, summary_dict)
    - cancel()
    """

    def __init__(self, config: dict, progress: Optional[callable] = None):
        self.config = config or {}
        self.progress = progress
        self._cancel_requested = False
        self.logger = logging.getLogger(__name__)

        # Threshold for SigLIP similarity to mark Pass. Default to 0.85 (config may override)
        self.siglip_threshold = float(self.config.get('ifu_siglip_threshold', 0.85))

        # Create SigLIP encoder lazily when needed
        self._encoder = None

    def _ensure_encoder(self):
        if self._encoder is None:
            try:
                self._encoder = create_siglip_encoder(self.config)
            except Exception as e:
                self.logger.exception('Failed to create SigLIP encoder: %s', e)
                self._encoder = None

    def cancel(self):
        self._cancel_requested = True
        self.logger.info('IFUController: cancellation requested')

    def _extract_images_from_input(self, input_path: str) -> List[Tuple[str, np.ndarray, int, int]]:
        """Extract images from an input PDF or image file.

        Returns a list of tuples: (name, image_np_bgr, page_index, serial_index)

        - For PDFs, uses `pdf_to_image.pdf_to_images` to get page images.
        - For image files, returns a single entry as page_index=0.
        - Names follow pattern: "page{page_index+1}_img{serial_index+1}".
        """
        out: List[Tuple[str, np.ndarray, int, int]] = []
        try:
            if input_path.lower().endswith('.pdf'):
                # Prefer extracting embedded image objects from the PDF using PyMuPDF
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(input_path)
                    found_any = False
                    for pidx in range(doc.page_count):
                        try:
                            page = doc.load_page(pidx)
                            images = page.get_images(full=True)
                            if not images:
                                continue
                            for img_idx, img in enumerate(images):
                                try:
                                    xref = img[0]
                                    base_image = doc.extract_image(xref)
                                    image_bytes = base_image.get('image')
                                    if not image_bytes:
                                        continue
                                    arr = np.frombuffer(image_bytes, dtype=np.uint8)
                                    im_np = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                    if im_np is None:
                                        continue
                                    name = f"page{pidx+1}_img{img_idx+1}"
                                    out.append((name, im_np, pidx, img_idx))
                                    found_any = True
                                except Exception:
                                    continue
                        except Exception:
                            continue
                    try:
                        doc.close()
                    except Exception:
                        pass

                    # If no embedded images were found and config allows, render full pages as fallback
                    if not out:
                        if bool(self.config.get('ifu_extract_fullpage', False)):
                            try:
                                pages = pdf_to_image.pdf_to_images(input_path, dpi=int(self.config.get('dpi', 300)), return_pil=False)
                                for pidx, page_img in enumerate(pages):
                                    name = f"page{pidx+1}_img1"
                                    out.append((name, page_img, pidx, 0))
                            except Exception:
                                pass
                except Exception:
                    # PyMuPDF not available or failed; optionally fall back to rendering entire pages
                    if bool(self.config.get('ifu_extract_fullpage', False)) and pdf_to_image is not None:
                        try:
                            pages = pdf_to_image.pdf_to_images(input_path, dpi=int(self.config.get('dpi', 300)), return_pil=False)
                            for pidx, page_img in enumerate(pages):
                                name = f"page{pidx+1}_img1"
                                out.append((name, page_img, pidx, 0))
                        except Exception:
                            pass
            else:
                # Load image via OpenCV
                img = cv2.imread(input_path)
                if img is not None:
                    out.append(("page1_img1", img, 0, 0))
        except Exception as e:
            self.logger.exception('Error extracting images from %s: %s', input_path, e)

        return out

    def run(self, input_paths: List[str], icon_paths: List[str], output_path: Optional[str] = None) -> Tuple[List[Any], Dict[str, Any]]:
        """Run IFU verification.

        Args:
            input_paths: list of input PDFs/images to check
            icon_paths: list of reference icon file paths (we will use the first one for now)

        Returns:
            results: list of result-like objects (simple dicts) compatible with GUI
            summary: dict with counts
        """
        self._cancel_requested = False
        results = []
        total_processed = 0
        passed = 0
        failed = 0

        self._ensure_encoder()
        if self._encoder is None or not self._encoder.is_available():
            self.logger.error('SigLIP encoder not available for IFU verification')
            return [], {'workflow': 'IFU', 'processed': 0, 'passed': 0, 'failed': 0}

        # Use first icon as reference for now
        ref_icon_path = icon_paths[0] if icon_paths else None
        ref_embedding = None
        if ref_icon_path:
            try:
                import cv2
                ref_img = cv2.imread(ref_icon_path)
                if ref_img is not None:
                    ref_embedding = self._encoder.encode_image(ref_img)
            except Exception as e:
                self.logger.exception('Failed to load/encode reference icon %s: %s', ref_icon_path, e)

    # Iterate input files
        file_count = len(input_paths) if input_paths else 0
        processed_files = 0
        for inp in input_paths:
            if self._cancel_requested:
                break

            processed_files += 1
            # Extract images from this input
            imgs = self._extract_images_from_input(inp)
            if not imgs:
                continue

            for name, img_np, page_idx, serial_idx in imgs:
                if self._cancel_requested:
                    break

                total_processed += 1

                # Compute embeddings for IFU image
                emb = self._encoder.encode_image(img_np)
                best_score = -1.0
                best_match = None

                if emb is not None and ref_embedding is not None:
                    try:
                        score = self._encoder.compute_similarity(emb, ref_embedding)
                        best_score = float(score)
                        best_match = ref_icon_path
                    except Exception as e:
                        self.logger.debug('Similarity computation failed: %s', e)

                # Decision using threshold (config or default)
                decision = 'Fail'
                try:
                    if best_score >= float(self.siglip_threshold):
                        decision = 'Pass'
                except Exception:
                    # fallback
                    if best_score >= 0.85:
                        decision = 'Pass'

                # Build a ResultRecord compatible with reporter and GUI
                try:
                    icon_img = None
                    if ref_icon_path and os.path.exists(ref_icon_path):
                        try:
                            icon_img = cv2.imread(ref_icon_path)
                        except Exception:
                            icon_img = None
                    rr = models.ResultRecord(
                        input_path=inp,
                        page_index=page_idx,
                        icon_name=os.path.basename(ref_icon_path) if ref_icon_path else '',
                        icon_snip=icon_img,
                        match_snip=img_np,
                        decision=decision,
                        score=float(best_score) if best_score is not None else 0.0,
                        comment='',
                        debug_image_path=None,
                        good_matches=1
                    )
                    # Attach extra scoring details used by reporter/UI
                    rr.siglip = float(best_score) if best_score is not None else None
                    rr.sim = None
                    rr.combined_score = float(best_score) if best_score is not None else None
                    rr.pattern_details = {}
                except Exception:
                    # Fallback to a minimal dynamic object if dataclass creation fails
                    rr = type('R', (), {})()
                    rr.input_path = inp
                    rr.page_index = page_idx
                    rr.icon_name = os.path.basename(ref_icon_path) if ref_icon_path else ''
                    rr.icon_snip = None
                    rr.match_snip = img_np
                    rr.decision = decision
                    rr.score = float(best_score) if best_score is not None else 0.0
                    rr.siglip = float(best_score) if best_score is not None else None
                    rr.sim = None
                    rr.combined_score = float(best_score) if best_score is not None else None
                    rr.pattern_details = {}

                results.append(rr)

                if decision == 'Pass':
                    passed += 1
                else:
                    failed += 1

            # progress callback per file
            if self.progress:
                try:
                    self.progress({'status': f'Processed {processed_files}/{file_count} inputs', 'total': file_count, 'current': processed_files})
                except Exception:
                    pass

        # Build RunSummary and write report (timestamped filename)
        summary = models.RunSummary(total=total_processed, passed=passed, failed=failed, results=results)

        try:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = None
            if output_path:
                try:
                    if os.path.isdir(output_path):
                        out_dir = output_path
                        os.makedirs(out_dir, exist_ok=True)
                        report_path = os.path.join(out_dir, f"ifu_report_{ts}.pdf")
                    else:
                        out_dir = os.path.dirname(os.path.abspath(output_path)) or os.getcwd()
                        os.makedirs(out_dir, exist_ok=True)
                        base, ext = os.path.splitext(os.path.basename(output_path))
                        ext = ext or '.pdf'
                        report_path = os.path.join(out_dir, f"{base}_{ts}{ext}")
                except Exception:
                    report_path = output_path
            else:
                out_dir = (self.config.get('debug_image_dir') if self.config and self.config.get('debug_image_dir') else os.path.join('output'))
                os.makedirs(out_dir, exist_ok=True)
                report_path = os.path.join(out_dir, f"ifu_report_{ts}.pdf")

            try:
                reporter.build_report(results, summary, report_path, config=self.config)
            except Exception:
                # Non-fatal; proceed
                pass
        except Exception:
            pass

        return results, summary

