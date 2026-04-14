import os
from ifu_verifier.ifu_verifier_window import (
    extract_pdf_text, detect_sections, 
    search_ifu_for_keywords, compute_verdict_suggestion
)

class IFUController:
    @staticmethod
    def verify_prd(ifu_path: str, keywords: list, req_sections: list, progress_cb=None) -> dict:
        if not ifu_path or not os.path.isfile(ifu_path):
            raise FileNotFoundError(f"IFU document not found: {ifu_path}")

        if progress_cb: progress_cb(2, "📄  Opening PDF…")
        
        def _ext_cb(cur, total):
            if progress_cb: progress_cb(int((cur / total) * 45) if total > 0 else 0, f"📄  Extracting page {cur} of {total}…")
        
        pages = extract_pdf_text(ifu_path, page_callback=_ext_cb)
        if not pages:
            raise ValueError(f"Could not extract text from {os.path.basename(ifu_path)}")

        if progress_cb: progress_cb(48, "🔎  Detecting document sections…")
        secs = detect_sections(pages)

        if progress_cb: progress_cb(50, f"🔍  Searching {len(pages)} pages…")
        
        def _src_cb(cur, total, found, n_kw):
            rem = n_kw - found
            pct = 50 + int((cur / total) * 50) if total > 0 else 50
            stat = f"🔍  Searching page {cur}/{total}  ·  {found}/{n_kw} keywords found"
            if rem > 0: stat += f"  ·  {rem} still pending…"
            else: stat += "  ·  All found — stopping early!"
            if progress_cb: progress_cb(pct, stat)

        hits = search_ifu_for_keywords(pages, keywords, progress_callback=_src_cb)
        verdict, score, reason = compute_verdict_suggestion(hits, secs, req_sections, 0.75)
        
        return {
            "hits": hits,
            "sections": secs,
            "verdict": verdict,
            "score": score,
            "reason": reason
        }