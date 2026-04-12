import logging
import os
import threading
from datetime import datetime

import cv2
import numpy as np

from . import (
    border_cleaner, models,
    pattern_verifier, reporter
)

from .candidate_generator import match_icon_candidates

# pdf_to_image requires PyMuPDF — import lazily so missing deps don't
# prevent the module from loading during tests or UI startup.
try:
    from . import pdf_to_image
except Exception:
    pdf_to_image = None

logger = logging.getLogger(__name__)

_FILTERED_SIM_W = 0.35
_FILTERED_SIG_W = 0.35
_FILTERED_PAT_W = 0.30


def _safe_float(val, default=0.0):
    """Convert val to float safely, returning default on failure."""
    try:
        return float(val)
    except Exception:
        return default


def _cosine_sim(a: np.ndarray, b: np.ndarray):
    """Return cosine similarity between two vectors, or None on failure."""
    try:
        a = a.astype('float32')
        b = b.astype('float32')
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na != 0 and nb != 0:
            return float(np.dot(a, b) / (na * nb))
    except Exception:
        logger.debug('_cosine_sim failed', exc_info=True)
    return None


class Controller:
    """Orchestrates the label verification pipeline."""

    def __init__(self, config: dict, progress=None):
        """
        Args:
            config  : Configuration dict (thresholds, flags, paths).
            progress: Optional progress emitter with an emit(dict) method.
        """
        self.config          = config or {}
        self.progress        = progress
        self._siglip_encoder = None
        self._cancelled      = False

    # ── Public API ─────────────────────────────────────────────────────────

    def cancel(self):
        """Request cancellation of a running pipeline."""
        self._cancelled = True

    def run(self, input_paths: list, icon_paths: list, output_path: str = None):
        """Run the full verification pipeline.

        Args:
            input_paths : List of input file paths (images or PDFs).
            icon_paths  : List of reference icon image paths.
            output_path : Optional path or folder for the PDF report.

        Returns:
            (results, summary) — list of ResultRecord and a RunSummary.
        """
        self._cancelled = False
        from . import embeddings as emb_module

        # ── Setup: embedder + SigLIP ───────────────────────────────────────
        embedder_model = self._load_embedder(emb_module)
        self._siglip_encoder = self._load_siglip()

        # ── Precompute icon embeddings ─────────────────────────────────────
        icon_embeddings        = {}
        icon_siglip_embeddings = {}
        for ip in (icon_paths or []):
            icon_img = self._read_image(ip)
            if icon_img is None:
                icon_embeddings[ip]        = None
                icon_siglip_embeddings[ip] = None
                continue
            icon_embeddings[ip]        = self._embed(emb_module, icon_img, embedder_model)
            icon_siglip_embeddings[ip] = self._siglip_embed(icon_img)

        # ── Count total work units for progress bar ────────────────────────
        total = self._count_total(input_paths, icon_paths)

        results = []
        passed  = 0
        failed  = 0
        idx     = 0
        n_files = len(input_paths or [])

        # ── Main loop ──────────────────────────────────────────────────────
        for file_idx, input_path in enumerate(input_paths or [], start=1):
            if self._cancelled:
                logger.info('Run cancelled by user.')
                break

            pages = self._load_pages(input_path)
            if not pages:
                logger.warning('No pages loaded from %s — skipping.', input_path)
                for ip in (icon_paths or []):
                    idx += 1
                    self._emit(idx, total,
                               f'Skipping unreadable file {file_idx}/{n_files}: '
                               f'{os.path.basename(input_path)}')
                continue

            for page_index, img in enumerate(pages):
                for icon_path in (icon_paths or []):
                    if self._cancelled:
                        break
                    try:
                        result, decision = self._process_pair(
                            img, input_path, page_index,
                            icon_path, icon_paths,
                            emb_module, embedder_model,
                            icon_embeddings, icon_siglip_embeddings,
                            output_path,
                        )
                        results.append(result)
                        if decision == 'Pass':
                            passed += 1
                        else:
                            failed += 1
                    except Exception:
                        logger.exception(
                            'Unhandled error processing %s / %s — marking as Fail.',
                            os.path.basename(input_path),
                            os.path.basename(icon_path),
                        )
                        failed += 1
                    finally:
                        idx += 1
                        self._emit(
                            idx, total,
                            f'Analyzing {file_idx}/{n_files}: '
                            f'{os.path.basename(input_path)} / '
                            f'{os.path.basename(icon_path)}'
                        )

        # ── Build report ───────────────────────────────────────────────────
        summary     = models.RunSummary(
            total=total, passed=passed, failed=failed, results=results
        )
        report_path = self._resolve_report_path(output_path)
        try:
            reporter.build_report(results, summary, report_path, config=self.config)
        except Exception:
            logger.exception('Failed to build PDF report at %s', report_path)

        return results, summary

    def run_async(self, *args, **kwargs):
        """Run the pipeline in a background thread. Returns the Thread."""
        thread = threading.Thread(target=self.run, args=args,
                                  kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    # ── Private helpers ────────────────────────────────────────────────────

    def _emit(self, current: int, total: int, status: str):
        if self.progress:
            try:
                self.progress.emit({'current': current, 'total': total,
                                    'status': status})
            except Exception:
                logger.debug('Progress emit failed', exc_info=True)

    def _read_image(self, path: str):
        """Read an image from disk. Returns None on failure."""
        try:
            img = cv2.imread(path)
            if img is None:
                logger.warning('cv2.imread returned None for %s', path)
            return img
        except Exception:
            logger.exception('Failed to read image: %s', path)
            return None

    def _load_embedder(self, emb_module):
        try:
            model, _ = emb_module.load_embedder()
            return model
        except Exception:
            logger.warning('Embedder unavailable — falling back to None.', exc_info=True)
            return None

    def _load_siglip(self):
        try:
            from .siglip_encoder import create_siglip_encoder
            enc = create_siglip_encoder(self.config)
            logger.info('SigLIP encoder loaded.')
            return enc
        except Exception:
            logger.info('SigLIP encoder not available (optional).')
            return None

    def _embed(self, emb_module, img, model):
        """Compute MobileNet embedding. Returns None on failure."""
        try:
            return emb_module.embed_crop(img, model)
        except Exception:
            logger.debug('embed_crop failed', exc_info=True)
            return None

    def _siglip_embed(self, img):
        """Compute SigLIP embedding. Returns None if unavailable."""
        if self._siglip_encoder is None:
            return None
        try:
            if self._siglip_encoder.is_available():
                return self._siglip_encoder.embed_crop(img)
        except Exception:
            logger.debug('SigLIP embed failed', exc_info=True)
        return None

    def _siglip_similarity(self, emb_a, emb_b):
        """Compute SigLIP similarity. Returns None if unavailable."""
        if self._siglip_encoder is None or emb_a is None or emb_b is None:
            return None
        try:
            return self._siglip_encoder.compute_similarity(emb_a, emb_b)
        except Exception:
            logger.debug('SigLIP similarity failed', exc_info=True)
            return None

    def _count_total(self, input_paths: list, icon_paths: list) -> int:
        """Count total work units (pages × icons) for the progress bar."""
        total = 0
        n_icons = max(1, len(icon_paths or []))
        for inp in (input_paths or []):
            ext = os.path.splitext(inp)[1].lower()
            if ext == '.pdf' and pdf_to_image is not None:
                page_count = self._count_pdf_pages(inp)
            else:
                page_count = 1
            total += page_count * n_icons
        return max(1, total)

    def _count_pdf_pages(self, path: str) -> int:
        try:
            import fitz
            doc = fitz.open(path)
            n   = doc.page_count
            doc.close()
            return n
        except Exception:
            logger.debug('fitz page count failed for %s', path, exc_info=True)
        try:
            pages = pdf_to_image.pdf_to_images(
                path, dpi=int(self.config.get('dpi', 300)),
                return_pil=False, store_converted=False
            )
            return len(pages)
        except Exception:
            logger.debug('pdf_to_images page count fallback failed', exc_info=True)
        return 1

    def _load_pages(self, input_path: str) -> list:
        """Convert an input file to a list of BGR numpy arrays."""
        ext = os.path.splitext(input_path)[1].lower()
        if ext == '.pdf':
            if pdf_to_image is None:
                logger.error('PyMuPDF not installed — cannot load PDF: %s', input_path)
                return []
            try:
                return pdf_to_image.pdf_to_images(
                    input_path,
                    dpi=int(self.config.get('dpi', 300)),
                    return_pil=False,
                    store_converted=bool(self.config.get('store_debug_images', False)),
                    debug_dir=self.config.get('debug_image_dir'),
                )
            except Exception:
                logger.exception('Failed to convert PDF to images: %s', input_path)
                return []
        else:
            img = self._read_image(input_path)
            return [img] if img is not None else []

    def _build_candidates(self, img, icon, icon_path,
                          emb_module, embedder_model,
                          icon_embeddings, icon_siglip_embeddings) -> list:
        """Run candidate generation, embedding, and SIM scoring."""
        debug_dir = self.config.get('debug_image_dir')
        try:
            kept_boxes, _ = match_icon_candidates(
                img, icon, None, self.config, debug_dir
            )
        except Exception:
            logger.debug('match_icon_candidates failed', exc_info=True)
            kept_boxes = []

        candidates = []
        for (bx, by, bw, bh) in (kept_boxes or []):
            try:
                x1, y1, w, h = int(bx), int(by), int(bw), int(bh)
                if w <= 0 or h <= 0:
                    continue
                crop  = img[y1:y1+h, x1:x1+w]
                entry = {
                    'bbox_abs': (x1, y1, w, h),
                    'bbox_raw': (x1, y1, w, h),
                    'crop':     crop,
                    'emb':      self._embed(emb_module, crop, embedder_model),
                    'siglip_emb': self._siglip_embed(crop),
                }
                # MobileNet cosine sim
                icon_emb = icon_embeddings.get(icon_path)
                entry['sim'] = (
                    _cosine_sim(entry['emb'], icon_emb)
                    if entry['emb'] is not None and icon_emb is not None
                    else None
                )
                candidates.append(entry)
            except Exception:
                logger.debug('Candidate build failed for bbox %s', (bx, by, bw, bh),
                             exc_info=True)
        return candidates

    def _apply_rotation_augmentation(self, candidates, icon_path,
                                     emb_module, embedder_model,
                                     icon_embeddings, icon_siglip_embeddings):
        """Try 90/180/270° rotations on low-signal candidates and keep the best."""
        rot_trigger = _safe_float(self.config.get('rotation_trigger_threshold', 0.8), 0.8)
        rot_max     = int(self.config.get('rotation_max_candidates', 10))
        rot_min_area = int(self.config.get('rotation_min_area', 100))

        # Determine max signal across all candidates
        icon_sl  = icon_siglip_embeddings.get(icon_path)
        max_signal = -999.0
        for pc in candidates:
            s   = pc.get('sim')   or -999.0
            sig = self._siglip_similarity(pc.get('siglip_emb'), icon_sl) or -999.0
            max_signal = max(max_signal, s, sig)

        logger.debug('[ROT-AUG] icon=%s max_signal=%.4f trigger=%.4f',
                     os.path.basename(icon_path), max_signal, rot_trigger)

        if max_signal >= rot_trigger or not candidates:
            return  # already good — skip augmentation

        rot_ops = [
            (90,  cv2.ROTATE_90_CLOCKWISE),
            (180, cv2.ROTATE_180),
            (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]
        to_augment = sorted(
            candidates,
            key=lambda p: p.get('sim') or -999.0,
            reverse=True
        )[:rot_max]

        for pc in to_augment:
            crop = pc.get('crop')
            if crop is None:
                continue
            h, w = crop.shape[:2]
            if w * h < rot_min_area:
                continue

            best_sim       = pc.get('sim')       or -999.0
            best_sig       = pc.get('siglip_sim')
            best_emb       = pc.get('emb')
            best_siglip_emb = pc.get('siglip_emb')

            for angle, op in rot_ops:
                try:
                    rot      = cv2.rotate(crop, op)
                    rot_emb  = self._embed(emb_module, rot, embedder_model)
                    icon_emb = icon_embeddings.get(icon_path)

                    sim_val = _cosine_sim(rot_emb, icon_emb) if rot_emb is not None and icon_emb is not None else None
                    if sim_val is not None and sim_val > best_sim:
                        best_sim, best_emb = sim_val, rot_emb

                    rot_sl  = self._siglip_embed(rot)
                    sig_val = self._siglip_similarity(rot_sl, icon_sl)
                    if sig_val is not None and (best_sig is None or sig_val > best_sig):
                        best_sig, best_siglip_emb = sig_val, rot_sl

                    logger.debug('[ROT-AUG] angle=%d sim=%.4f siglip=%s',
                                 angle, sim_val or -1.0, sig_val)
                except Exception:
                    logger.debug('Rotation %d failed', angle, exc_info=True)

            pc['sim']               = best_sim if best_sim != -999.0 else None
            pc['siglip_sim']        = best_sig
            pc['emb']               = best_emb
            pc['siglip_emb']        = best_siglip_emb

    def _score_candidates(self, candidates, icon, icon_path,
                          icon_siglip_embeddings) -> tuple:
        """Score all candidates and return (best_entry, best_score)."""
        sim_prefilter_th = _safe_float(self.config.get('sim_prefilter_threshold', 0.6), 0.6)
        icon_sl          = icon_siglip_embeddings.get(icon_path)

        filtered = [pc for pc in candidates
                    if pc.get('sim') is not None and pc['sim'] > sim_prefilter_th]
        pool     = filtered if filtered else candidates

        best_entry = None
        best_score = -999.0

        for pc in pool:
            try:
                # SigLIP similarity
                if pc.get('siglip_sim') is None:
                    pc['siglip_sim'] = self._siglip_similarity(pc.get('siglip_emb'), icon_sl)

                # Pattern verifier
                pat_score, pat_details = 0.0, {}
                if pc.get('crop') is not None:
                    try:
                        pat_score, pat_details = pattern_verifier.compute_combined_score(
                            pc['crop'], icon
                        )
                        pat_score   = _safe_float(pat_score)
                        pat_details = pat_details or {}
                    except Exception:
                        logger.debug('pattern_verifier failed', exc_info=True)
                pc['pattern_score']   = pat_score
                pc['pattern_details'] = pat_details

                sim_val = _safe_float(pc.get('sim'))
                sig_val = _safe_float(pc.get('siglip_sim')) if pc.get('siglip_sim') is not None else 0.0

                if filtered:
                    combined = _FILTERED_SIM_W * sim_val + _FILTERED_SIG_W * sig_val + _FILTERED_PAT_W * pat_score
                else:
                    sim_w  = _safe_float(self.config.get('sim_weight', 0.5), 0.5)
                    sig_w  = _safe_float(self.config.get('siglip_weight', 0.5), 0.5)
                    combined = (sim_w * sim_val + sig_w * sig_val
                                if sim_val > -999.0 or sig_val > -999.0
                                else -999.0)

                pc['combined_score'] = combined
                if combined > best_score:
                    best_score = combined
                    best_entry = pc
            except Exception:
                logger.debug('Candidate scoring failed', exc_info=True)

        return best_entry, best_score

    def _decide(self, best_entry) -> tuple:
        """Return (decision, score) from the best candidate entry."""
        if best_entry is None:
            return 'Fail', 0.0

        siglip_th = _safe_float(self.config.get('siglip_decision_threshold', 0.81), 0.81)
        sim_th    = _safe_float(self.config.get('sim_decision_threshold',    0.81), 0.81)

        best_sig = best_entry.get('siglip_sim')
        best_sim = best_entry.get('sim')
        score    = _safe_float(best_entry.get('combined_score')
                               or best_entry.get('sim'), 0.0)

        if best_sig is not None and self._siglip_encoder is not None:
            decision = 'Pass' if best_sig >= siglip_th else 'Fail'
            score    = float(best_sig)
        else:
            decision = 'Pass' if (best_sim is not None and best_sim >= sim_th) else 'Fail'

        return decision, score

    def _save_debug_overlay(self, img, candidates, input_path, icon_path) -> str:
        """Draw candidate bboxes on a copy of img and save to debug dir."""
        if not self.config.get('store_debug_images', False):
            return None
        try:
            dbg_dir    = self.config.get('debug_image_dir') or 'output/debug'
            os.makedirs(dbg_dir, exist_ok=True)
            base       = os.path.splitext(os.path.basename(input_path))[0]
            icon_base  = os.path.splitext(os.path.basename(icon_path))[0]
            vis        = img.copy()
            for pc in candidates:
                try:
                    bx, by, bw, bh = pc.get('bbox_abs', (0, 0, 0, 0))
                    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
                    sim_v = pc.get('sim')
                    if sim_v is not None:
                        cv2.putText(vis, f"sim={sim_v:.3f}",
                                    (max(0, bx), max(15, by - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                except Exception:
                    logger.debug('Debug overlay draw failed', exc_info=True)
            vis_path = os.path.join(dbg_dir, f"{base}_{icon_base}_candidates.png")
            cv2.imwrite(vis_path, vis)
            return vis_path
        except Exception:
            logger.debug('save_debug_overlay failed', exc_info=True)
            return None

    def _save_failure_report(self, icon, candidates, input_path,
                              icon_path, output_path):
        """Write a per-failure candidate PDF for investigation."""
        try:
            out_dir = (
                os.path.dirname(os.path.abspath(output_path))
                if output_path
                else self.config.get('debug_image_dir') or os.path.join('output', 'failures')
            )
            os.makedirs(out_dir, exist_ok=True)
            base      = os.path.splitext(os.path.basename(input_path))[0]
            icon_base = os.path.splitext(os.path.basename(icon_path))[0]
            ts        = datetime.now().strftime('%Y%m%d_%H%M%S')
            fail_pdf  = os.path.join(out_dir, f"{base}_{icon_base}_failure_{ts}.pdf")
            cand_list = [
                {
                    'crop':            pc.get('crop'),
                    'sim':             pc.get('sim'),
                    'siglip_sim':      pc.get('siglip_sim'),
                    'combined_score':  pc.get('combined_score'),
                    'combined_details': pc.get('pattern_details') or {},
                    'bbox':            pc.get('bbox_abs') or pc.get('bbox_raw'),
                }
                for pc in candidates
            ]
            reporter.build_icon_report(
                icon, cand_list, fail_pdf,
                title=f"Failure candidates: {base} — {icon_base}"
            )
        except Exception:
            logger.exception('Failed to write failure candidate report')

    def _process_pair(self, img, input_path, page_index,
                      icon_path, icon_paths,
                      emb_module, embedder_model,
                      icon_embeddings, icon_siglip_embeddings,
                      output_path) -> tuple:
        """Process one (page image, icon) pair and return (ResultRecord, decision)."""
        icon = self._read_image(icon_path)
        if icon is None:
            logger.warning('Could not read icon: %s', icon_path)
            result = models.ResultRecord(
                input_path=input_path, page_index=page_index,
                icon_name=os.path.basename(icon_path),
                icon_snip=None, match_snip=None,
                decision='Fail', score=0.0,
                comment='Icon could not be read.',
            )
            return result, 'Fail'

        # Build candidates
        candidates = self._build_candidates(
            img, icon, icon_path,
            emb_module, embedder_model,
            icon_embeddings, icon_siglip_embeddings,
        )

        # Rotation augmentation
        if self.config.get('rotation_augmentation_enabled', True):
            self._apply_rotation_augmentation(
                candidates, icon_path,
                emb_module, embedder_model,
                icon_embeddings, icon_siglip_embeddings,
            )

        # Score and decide
        best_entry, _ = self._score_candidates(
            candidates, icon, icon_path, icon_siglip_embeddings
        )
        decision, score = self._decide(best_entry)

        # Debug overlay
        vis_path = self._save_debug_overlay(img, candidates, input_path, icon_path)

        # Build ResultRecord
        result = models.ResultRecord(
            input_path=input_path,
            page_index=page_index,
            icon_name=os.path.basename(icon_path),
            icon_snip=icon,
            match_snip=best_entry.get('crop') if best_entry is not None else None,
            decision=decision,
            score=score,
            debug_image_path=vis_path,
            good_matches=len(candidates),
            sim=_safe_float(best_entry.get('sim'))           if best_entry else None,
            siglip=_safe_float(best_entry.get('siglip_sim')) if best_entry else None,
            combined_score=_safe_float(best_entry.get('combined_score')) if best_entry else None,
            pattern_details=best_entry.get('pattern_details') if best_entry else None,
        )

        # Failure report
        if decision == 'Fail':
            self._save_failure_report(icon, candidates, input_path,
                                      icon_path, output_path)

        return result, decision

    def _resolve_report_path(self, output_path: str) -> str:
        """Determine where to write the PDF report."""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        try:
            if output_path:
                if os.path.isdir(output_path):
                    out_dir = output_path
                else:
                    out_dir  = os.path.dirname(os.path.abspath(output_path)) or os.getcwd()
                    base, ext = os.path.splitext(os.path.basename(output_path))
                    ext       = ext or '.pdf'
                    return os.path.join(out_dir, f"{base}_{ts}{ext}")
            else:
                out_dir = self.config.get('debug_image_dir') or 'output'
            os.makedirs(out_dir, exist_ok=True)
            return os.path.join(out_dir, f"report_{ts}.pdf")
        except Exception:
            logger.exception('Failed to resolve report path')
            return f"report_{ts}.pdf"

def _load_config_from_ini(config_path):
    import configparser

    def _coerce(value):
        low = value.strip().lower()
        if low in ('true', 'yes', '1'):
            return True
        if low in ('false', 'no', '0'):
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value.strip()

    parser = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    parser.read(config_path, encoding='utf-8')
    flat = {}
    for section in parser.sections():
        for key, value in parser.items(section):
            flat[key] = _coerce(value)
    return flat


class VerificationController:
    """GUI-friendly wrapper around Controller.

    Loads the INI config file automatically and exposes a simpler run()
    signature that accepts a single label path + icon directory.
    """

    def __init__(self, config_path=None, progress=None):
        config = {}
        if config_path and os.path.isfile(config_path):
            try:
                config = _load_config_from_ini(config_path)
                logger.info('Config loaded from %s (%d keys)', config_path, len(config))
            except Exception:
                logger.exception('Failed to load config from %s — using defaults', config_path)
        elif config_path:
            logger.warning('Config file not found: %s — using defaults', config_path)
        self._inner = Controller(config=config, progress=progress)

    def cancel(self):
        self._inner.cancel()

    def run(self, label_path, icon_dir, output_path=None):
        """Run verification for one label image against all icons in icon_dir.

        Returns (results, summary) — same as Controller.run.
        """
        import glob as _glob

        if not os.path.exists(label_path):
            raise FileNotFoundError('Label image not found: ' + label_path)
        if not os.path.isdir(icon_dir):
            raise NotADirectoryError('Icon directory not found: ' + icon_dir)

        icon_exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif')
        icon_paths = []
        for pattern in icon_exts:
            icon_paths.extend(_glob.glob(os.path.join(icon_dir, pattern)))
        icon_paths.sort()

        if not icon_paths:
            raise ValueError('No icon images found in: ' + icon_dir)

        logger.info('VerificationController: label=%s  icons=%d  config_keys=%d',
                    os.path.basename(label_path), len(icon_paths),
                    len(self._inner.config))

        return self._inner.run(
            input_paths=[label_path],
            icon_paths=icon_paths,
            output_path=output_path,
        )