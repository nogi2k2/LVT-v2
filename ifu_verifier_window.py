"""
ifu_verifier_window.py
=======================
IFU Verification Window — EVO MDD Verification Tool

Workflow:
  1. Loaded with IFU PRDs from the parsed V&V Plan (passed in as ifu_prds)
  2. Engineer uploads the IFU PDF (one shared file for all PRDs)
  3. Click "Run All Verifications" — Ollama processes each PRD:
       - Narrows to 3-5 candidate pages using section-heading detection
       - Sends requirement + candidate pages to the configured model
       - Returns: satisfied / confidence / reason / evidence_text / evidence_page
  4. Results shown as cards — engineer can confirm or override:
       Pass / Fail / Manual Review buttons + free-text Notes field
  5. "Export PDF Report" builds the full report in a background thread:
       - Cover summary table
       - Per-PRD page with IFU page image + yellow paragraph highlight
       - Evidence callout box

Helper functions (also imported by main_gui.py for its inline IFU tab):
  extract_keywords, extract_pdf_text, detect_sections,
  search_ifu_for_keywords, compute_verdict_suggestion,
  infer_required_sections, build_pdf_report
"""

import os
import re
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime

try:
    from app import (COLORS, FONT_TITLE, FONT_HEADING, FONT_BODY,
                     FONT_MONO, FONT_SMALL, _lighten)
except ImportError:
    COLORS = {
        "bg": "#F5F7FA", "bg_card": "#FFFFFF", "bg_input": "#FFFFFF",
        "accent": "#0B54A4", "accent_light": "#00A0DC",
        "accent2": "#16A34A", "accent3": "#D97706",
        "text": "#1A1A2E", "text_muted": "#6B7280",
        "border": "#D1D5DB", "header_bg": "#0B54A4",
        "danger": "#DC2626", "success": "#16A34A", "warn": "#D97706",
        "log_bg": "#F8FAFC", "log_text": "#1E3A5F",
    }
    FONT_TITLE   = ("Segoe UI", 16, "bold")
    FONT_HEADING = ("Segoe UI", 11, "bold")
    FONT_BODY    = ("Segoe UI",  9)
    FONT_MONO    = ("Consolas",  9)
    FONT_SMALL   = ("Segoe UI",  8)
    def _lighten(h):
        h = h.lstrip("#"); r, g, b = [int(h[i:i+2], 16) for i in (0, 2, 4)]
        return f"#{min(255,r+30):02x}{min(255,g+30):02x}{min(255,b+30):02x}"


# ═══════════════════════════════════════════════════════════════════════════════
# Core keyword-search helpers (also used by main_gui.py IFU tab)
# ═══════════════════════════════════════════════════════════════════════════════

_STOP_WORDS = {
    "the","a","an","and","or","of","for","in","to","shall","include",
    "information","related","this","is","with","be","will","as","per",
    "that","it","its","are","have","has","not","from","on","at","by",
    "if","all","any","each","which","was","were","been","into","about",
    "also","both","but","can","do","does","did","had","may","more",
    "no","nor","our","out","so","than","their","them","then","there",
    "these","they","this","those","through","under","up","use","very",
    "when","where","who","will","with","would","you","your",
}


def _word_stem(word: str) -> str:
    w = word.lower()
    for suffix in (
        "ifications", "ication", "izations", "ization",
        "tions", "tion",
        "nesses", "ness",
        "ments", "ment",
        "ings", "ing",
        "ances", "ance",
        "ences", "ence",
        "ables", "able",
        "ibles", "ible",
        "ities", "ity",
        "ious", "ous",
        "iers", "ier",
        "ies", "ied",
        "ers", "er",
        "ed", "es",
        "s",
    ):
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            return w[: -len(suffix)]
    return w


def _kw_search_patterns(kw: str) -> list:
    kw_lower = kw.lower()
    stem     = _word_stem(kw_lower)
    results  = []
    exact_body = re.escape(kw_lower)
    results.append((
        re.compile(r"(?<!\w)" + exact_body + r"(?!\w)", re.IGNORECASE),
        re.compile(r".{0,80}" + exact_body + r".{0,80}",  re.IGNORECASE),
        "exact",
    ))
    if stem != kw_lower and len(stem) >= 4:
        stem_body = re.escape(stem)
        results.append((
            re.compile(r"(?<!\w)" + stem_body + r"\w*", re.IGNORECASE),
            re.compile(r".{0,80}" + stem_body + r"\w*.{0,80}", re.IGNORECASE),
            "stem",
        ))
    return results


_SECTION_PATTERNS = [
    (re.compile(r"^\s*intended\s+use", re.I | re.M),            "Intended Use"),
    (re.compile(r"^\s*specifications?", re.I | re.M),            "Specifications"),
    (re.compile(r"^\s*performance\s+specifications?", re.I | re.M), "Performance Specifications"),
    (re.compile(r"^\s*symbols?\s+glossary", re.I | re.M),        "Symbols Glossary"),
    (re.compile(r"^\s*instructions?\s+for\s+use", re.I | re.M),  "Instructions for Use"),
    (re.compile(r"^\s*warnings?", re.I | re.M),                  "Warnings"),
    (re.compile(r"^\s*compatibility", re.I | re.M),              "Compatibility"),
    (re.compile(r"^\s*disposal", re.I | re.M),                   "Disposal"),
    (re.compile(r"^\s*materials?", re.I | re.M),                 "Materials"),
    (re.compile(r"^\s*contraindications?", re.I | re.M),         "Contraindications"),
    (re.compile(r"^\s*cleaning\s+and\s+maintenance", re.I | re.M), "Cleaning and Maintenance"),
    (re.compile(r"^\s*troubleshooting", re.I | re.M),            "Troubleshooting"),
    (re.compile(r"^\s*accessories", re.I | re.M),                "Accessories"),
]


def extract_keywords(requirement_text: str) -> list:
    phrases = re.findall(r"['\"]([^'\"]{3,30})['\"]", requirement_text)
    words   = re.findall(r"[a-zA-Z]{4,}", requirement_text.lower())
    kws     = [w for w in words if w not in _STOP_WORDS]
    seen, out = set(), []
    for p in phrases:
        key = p.lower().strip()
        if key not in seen:
            seen.add(key)
            out.append(p.strip())
    for w in kws:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out[:10]


def extract_pdf_text(pdf_path: str, page_callback=None) -> list:
    pages = []
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                t = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                pages.append({"page": i + 1, "text": t})
                if page_callback:
                    page_callback(i + 1, total)
        return pages
    except Exception:
        pass
    try:
        import fitz
        doc = fitz.open(pdf_path)
        total = len(doc)
        for i, page in enumerate(doc):
            t = page.get_text() or ""
            pages.append({"page": i + 1, "text": t})
            if page_callback:
                page_callback(i + 1, total)
        doc.close()
        return pages
    except Exception:
        pass
    return []


def detect_sections(pages: list) -> dict:
    found     = {}
    full_text = "\n".join(p["text"] for p in pages)
    for pattern, section_name in _SECTION_PATTERNS:
        m = pattern.search(full_text)
        if m:
            char_pos   = m.start()
            cumulative = 0
            page_num   = 1
            for p in pages:
                cumulative += len(p["text"]) + 1
                if char_pos < cumulative:
                    page_num = p["page"]
                    break
            snippet = full_text[m.start():m.start()+120].replace("\n", " ").strip()
            found[section_name] = {"page": page_num, "snippet": snippet}
    return found


def search_ifu_for_keywords(pages: list, keywords: list,
                             progress_callback=None) -> list:
    if not keywords:
        return []
    sections    = detect_sections(pages)
    total_pages = len(pages)
    kw_patterns = [_kw_search_patterns(kw) for kw in keywords]
    hits        = [None] * len(keywords)
    pending     = set(range(len(keywords)))

    for pg_idx, pg in enumerate(pages):
        if not pending:
            break
        if progress_callback:
            progress_callback(pg_idx + 1, total_pages,
                              len(keywords) - len(pending), len(keywords))
        section_label = ""
        best_sec_page = -1
        for sec_name, sec_info in sections.items():
            if sec_info["page"] <= pg["page"] and sec_info["page"] > best_sec_page:
                section_label = sec_name
                best_sec_page = sec_info["page"]

        for kw_idx in list(pending):
            kw_orig = keywords[kw_idx]
            for chk_pat, snip_pat, match_type in kw_patterns[kw_idx]:
                m = chk_pat.search(pg["text"])
                if not m:
                    continue
                if match_type == "stem":
                    matched_word = m.group(0)
                    min_len = max(len(kw_orig) - 2,
                                  len(_word_stem(kw_orig.lower())) + 2)
                    if len(matched_word) < min_len:
                        continue
                ms = snip_pat.findall(pg["text"])
                snippet = (ms[0].strip().replace("\n", " ") if ms
                           else pg["text"][:120].replace("\n", " "))
                hits[kw_idx] = {
                    "keyword":    keywords[kw_idx],
                    "page":       pg["page"],
                    "snippet":    snippet[:180],
                    "found":      True,
                    "match_type": match_type,
                    "section":    section_label,
                    "all_pages":  [pg["page"]],
                }
                pending.discard(kw_idx)
                break

    if pending and sections:
        for kw_idx in list(pending):
            kw   = keywords[kw_idx]
            stem = _word_stem(kw.lower())
            for sec_name, sec_info in sections.items():
                sec_lower = sec_name.lower()
                if (stem in sec_lower or kw.lower() in sec_lower or
                        _word_stem(sec_lower.replace(" ", "")) in stem):
                    hits[kw_idx] = {
                        "keyword":    kw,
                        "page":       sec_info["page"],
                        "snippet":    f'Section heading: "{sec_name}"',
                        "found":      True,
                        "match_type": "section",
                        "section":    sec_name,
                        "all_pages":  [sec_info["page"]],
                    }
                    pending.discard(kw_idx)
                    break

    return [
        hits[i] if hits[i] is not None else {
            "keyword":    keywords[i],
            "page":       None,
            "snippet":    "",
            "found":      False,
            "match_type": None,
            "section":    "",
            "all_pages":  [],
        }
        for i in range(len(keywords))
    ]


def compute_verdict_suggestion(hits: list, sections_found: dict,
                                required_sections: list,
                                threshold: float = 0.75) -> tuple:
    if not hits:
        return "REVIEW", 0.0, "No keywords to evaluate"
    _WEIGHTS      = {"exact": 1.0, "stem": 0.9, "section": 0.75}
    weighted_sum  = sum(_WEIGHTS.get(h.get("match_type"), 0.0)
                        for h in hits if h["found"])
    max_possible  = len(hits) * 1.0
    score         = weighted_sum / max_possible if max_possible > 0 else 0.0
    found_count   = sum(1 for h in hits if h["found"])
    exact_n       = sum(1 for h in hits if h.get("match_type") == "exact")
    stem_n        = sum(1 for h in hits if h.get("match_type") == "stem")
    section_n     = sum(1 for h in hits if h.get("match_type") == "section")
    missing_sections = [s for s in required_sections if s not in sections_found]
    parts = []
    if exact_n:
        parts.append(f"{exact_n} exact")
    if stem_n:
        parts.append(f"{stem_n} stem-matched")
    if section_n:
        parts.append(f"{section_n} via section heading")
    breakdown = " + ".join(parts) if parts else "none"
    if score >= threshold and not missing_sections:
        verdict = "PASS"
        reason  = f"{found_count}/{len(hits)} keywords found ({breakdown})"
    elif score == 0.0:
        verdict = "FAIL"
        reason  = "No keywords or concepts found in IFU"
    elif score < 0.5:
        verdict = "REVIEW"
        reason  = (f"Only {found_count}/{len(hits)} found ({breakdown}) "
                   f"— manual review needed")
    else:
        verdict = "REVIEW"
        reason  = f"{found_count}/{len(hits)} found ({breakdown})"
        if missing_sections:
            reason += f" · missing sections: {', '.join(missing_sections)}"
    return verdict, score, reason


def infer_required_sections(requirement_text: str) -> list:
    text   = requirement_text.lower()
    needed = []
    mapping = [
        (["performance spec", "specification", "tubing", "circuit length",
          "resistance", "compliance", "filter efficiency", "technical"], "Specifications"),
        (["instructions for use", "assembly", "how to use"],             "Instructions for Use"),
        (["symbol", "glossary", "symbols glossary"],                     "Symbols Glossary"),
        (["intended use", "indication"],                                  "Intended Use"),
        (["warning", "caution"],                                          "Warnings"),
        (["disposal", "dispose"],                                         "Disposal"),
        (["material", "latex", "phthalate"],                              "Materials"),
    ]
    for keywords_list, section_name in mapping:
        if any(k in text for k in keywords_list):
            needed.append(section_name)
    return needed


def render_hit_page(pdf_path: str, keyword: str, page_num: int,
                    scale: float = 1.5) -> bytes:
    """Render a PDF page with yellow highlights on keyword occurrences.
    Returns PNG bytes or None on failure. Used by build_pdf_report."""
    try:
        import fitz
        doc  = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return None
        page = doc[page_num - 1]
        search_terms = [keyword]
        stem = _word_stem(keyword.lower())
        if stem != keyword.lower() and len(stem) >= 4:
            search_terms.append(stem)
        for term in search_terms:
            for rect in page.search_for(term, quads=False):
                hl = page.add_highlight_annot(rect)
                hl.set_colors(stroke=(1.0, 0.92, 0.0))
                hl.update()
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, annots=True, alpha=False)
        png_bytes = pix.tobytes("png")
        doc.close()
        return png_bytes
    except Exception:
        return None


def build_pdf_report(cards_data: list, plan_data, output_path: str,
                     ifu_pdf_path: str = "") -> str:
    """
    Build the keyword-based IFU verification PDF report (used by main_gui.py).
    Returns output_path on success or an error string.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors as rl_colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable,
                                        KeepTogether, Image as RLImage,
                                        PageBreak)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        import io

        W = A4[0] - 40 * mm

        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            leftMargin=20*mm, rightMargin=20*mm,
            topMargin=20*mm,  bottomMargin=20*mm)

        styles = getSampleStyleSheet()

        BLUE       = rl_colors.HexColor("#0B54A4")
        BLUE_LIGHT = rl_colors.HexColor("#EFF6FF")
        GREEN      = rl_colors.HexColor("#16A34A")
        GREEN_LT   = rl_colors.HexColor("#DCFCE7")
        RED        = rl_colors.HexColor("#DC2626")
        RED_LT     = rl_colors.HexColor("#FEE2E2")
        AMBER      = rl_colors.HexColor("#D97706")
        AMBER_LT   = rl_colors.HexColor("#FEF3C7")
        GREY       = rl_colors.HexColor("#6B7280")
        LGREY      = rl_colors.HexColor("#F5F7FA")
        WHITE      = rl_colors.white

        title_s = ParagraphStyle("rpt_title", parent=styles["Normal"],
                                 fontName="Helvetica-Bold", fontSize=16,
                                 textColor=WHITE, spaceAfter=0, leading=20)
        sub_s   = ParagraphStyle("rpt_sub", parent=styles["Normal"],
                                 fontName="Helvetica", fontSize=9,
                                 textColor=WHITE, spaceAfter=0)
        h2_s    = ParagraphStyle("rpt_h2", parent=styles["Normal"],
                                 fontName="Helvetica-Bold", fontSize=11,
                                 textColor=BLUE, spaceAfter=2, spaceBefore=6)
        body_s  = ParagraphStyle("rpt_body", parent=styles["Normal"],
                                 fontName="Helvetica", fontSize=8,
                                 textColor=rl_colors.HexColor("#1A1A2E"),
                                 leading=12)
        snip_s  = ParagraphStyle("rpt_snip", parent=styles["Normal"],
                                 fontName="Helvetica", fontSize=7,
                                 textColor=rl_colors.HexColor("#374151"),
                                 leading=11)
        note_s  = ParagraphStyle("rpt_note", parent=styles["Normal"],
                                 fontName="Helvetica", fontSize=8,
                                 textColor=rl_colors.HexColor("#374151"),
                                 leading=11)

        story = []

        # ── Cover header ──────────────────────────────────────────────────
        ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plan_title = (getattr(plan_data, "doc_title", "") or "—") if plan_data else "—"
        er_num     = (getattr(plan_data, "er_number", "")  or "—") if plan_data else "—"

        hdr_tbl = Table([[
            Paragraph("IFU Verification Report", title_s),
            Paragraph(f"{plan_title}<br/>"
                      f"ER: {er_num}  |  Generated: {ts}", sub_s),
        ]], colWidths=[W * 0.45, W * 0.55])
        hdr_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), BLUE),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story += [hdr_tbl, Spacer(1, 8)]

        # ── Summary bar ────────────────────────────────────────────────────
        total   = len(cards_data)
        passed  = sum(1 for c in cards_data if c.get("verdict_var") and
                      c["verdict_var"].get() == "PASS")
        failed  = sum(1 for c in cards_data if c.get("verdict_var") and
                      c["verdict_var"].get() == "FAIL")
        review  = sum(1 for c in cards_data if c.get("verdict_var") and
                      c["verdict_var"].get() == "REVIEW")
        pending = total - passed - failed - review

        sum_row = Table([[
            Paragraph(f"<b>Total: {total}</b>", body_s),
            Paragraph(f"<font color='#16A34A'><b>✔ Pass: {passed}</b></font>",
                      body_s),
            Paragraph(f"<font color='#DC2626'><b>✗ Fail: {failed}</b></font>",
                      body_s),
            Paragraph(f"<font color='#D97706'>⚠ Review: {review}</font>",
                      body_s),
            Paragraph(f"○ Pending: {pending}", body_s),
        ]], colWidths=[W / 5] * 5)
        sum_row.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), LGREY),
            ("BOX",           (0, 0), (-1, -1), 0.5,
             rl_colors.HexColor("#D1D5DB")),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ]))
        story += [sum_row, Spacer(1, 10), HRFlowable(width="100%",
                                                      thickness=1,
                                                      color=BLUE),
                  Spacer(1, 6)]

        # ── Per-PRD detail sections ────────────────────────────────────────
        for cd in cards_data:
            prd    = cd.get("prd")
            if prd is None:
                continue
            prd_id = getattr(prd, "prd_id", "?")
            req    = getattr(prd, "requirement_text", "")
            vdict  = cd.get("verdict_var")
            verdict = vdict.get() if vdict else "PENDING"
            notes_w = cd.get("notes_text")
            notes   = notes_w.get("1.0", "end").strip() if notes_w else ""

            vc = {"PASS": GREEN, "FAIL": RED,
                  "REVIEW": AMBER}.get(verdict, GREY)
            vi = {"PASS": "✔", "FAIL": "✗",
                  "REVIEW": "⚠", "PENDING": "○"}.get(verdict, "○")

            # PRD header
            hd_st = ParagraphStyle("hd", parent=styles["Normal"],
                                   fontName="Helvetica-Bold", fontSize=10,
                                   textColor=WHITE)
            hd_tbl = Table([[
                Paragraph(f"{prd_id}  {vi}  {verdict}", hd_st)
            ]], colWidths=[W])
            hd_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), vc),
                ("TOPPADDING",    (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ]))
            story.append(KeepTogether([
                hd_tbl,
                Spacer(1, 3),
                Paragraph(req, body_s),
                Spacer(1, 4),
            ]))

            # Evidence hits (key is "hits" as set by main_gui._ifu_show_result)
            hits = cd.get("hits", [])
            if hits:
                ev_rows = [
                    [Paragraph("Keyword", snip_s),
                     Paragraph("Result",  snip_s),
                     Paragraph("Page",    snip_s),
                     Paragraph("Context", snip_s)]
                ]
                for h in hits:
                    colour = GREEN if h["found"] else RED
                    icon   = "✔" if h["found"] else "✗"
                    ev_rows.append([
                        Paragraph(h["keyword"], snip_s),
                        Paragraph(
                            f'<font color="{"#16A34A" if h["found"] else "#DC2626"}">'
                            f'{icon} {h.get("match_type","") or "not found"}</font>',
                            snip_s),
                        Paragraph(str(h["page"] or "—"), snip_s),
                        Paragraph((h.get("snippet") or "")[:100], snip_s),
                    ])
                ev_tbl = Table(ev_rows,
                               colWidths=[W*0.18, W*0.16, W*0.07, W*0.59],
                               repeatRows=1)
                ev_tbl.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, 0),  BLUE),
                    ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
                    ("GRID",          (0, 0), (-1, -1), 0.3,
                     rl_colors.HexColor("#D1D5DB")),
                    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
                    ("FONTSIZE",      (0, 0), (-1, -1), 7),
                    ("TOPPADDING",    (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 4),
                    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ]))
                story += [ev_tbl, Spacer(1, 4)]

                # Embed IFU page images for found hits (exact/stem only)
                if ifu_pdf_path:
                    for h in hits:
                        if h["found"] and h.get("match_type") in ("exact", "stem"):
                            png = render_hit_page(
                                ifu_pdf_path, h["keyword"], h["page"])
                            if png:
                                try:
                                    from PIL import Image as _PIL
                                    pil = _PIL.open(io.BytesIO(png))
                                    pw, ph = pil.size
                                    max_w = float(W)
                                    max_h = 200.0
                                    scale = min(max_w / pw, max_h / ph, 1.0)
                                    nw = int(pw * scale)
                                    nh = int(ph * scale)
                                    pil2 = pil.resize((nw, nh), _PIL.LANCZOS)
                                    bio2 = io.BytesIO()
                                    pil2.save(bio2, format="PNG")
                                    bio2.seek(0)
                                    img_rl = RLImage(bio2, width=nw, height=nh)
                                    story += [
                                        Paragraph(
                                            f"IFU page {h['page']} \u2014 "
                                            f"keyword: \u201c{h['keyword']}\u201d",
                                            snip_s),
                                        img_rl,
                                        Spacer(1, 4),
                                    ]
                                except Exception:
                                    pass

            if notes:
                story += [
                    Paragraph("Engineer notes:", h2_s),
                    Paragraph(notes, note_s),
                    Spacer(1, 4),
                ]

            story += [HRFlowable(width="100%", thickness=0.5,
                                  color=rl_colors.HexColor("#D1D5DB")),
                      Spacer(1, 6)]

        doc.build(story)
        return output_path

    except Exception as e:
        import traceback
        return f"Report error: {e}\n{traceback.format_exc()}"


# ═══════════════════════════════════════════════════════════════════════════════
# IFU Verifier Window — LLM-powered standalone window
# ═══════════════════════════════════════════════════════════════════════════════

class IFUVerifierWindow(tk.Toplevel):
    """
    Standalone IFU Verification window using Ollama LLM.

    Opened from the home screen after a V&V Plan is loaded.
    Parameters:
        parent    — Tkinter parent widget
        ifu_prds  — list of PRDEntry objects (IFU requirements from V&V Plan)
        plan_data — VVPlanData object (for header info and report metadata)
    """

    DEFAULT_MODEL     = "llama3.1:8b"
    AUTO_PASS_THRESHOLD = 0.75

    def __init__(self, parent, ifu_prds: list, plan_data=None):
        super().__init__(parent)
        self.title("IFU Verification — EVO MDD Verification Tool")
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)

        self._prds      = ifu_prds
        self._plan_data = plan_data
        self._cards     = []
        self._ifu_path  = tk.StringVar()
        self._model_var = tk.StringVar(value=self.DEFAULT_MODEL)
        self._running   = False

        # Check required packages
        self._missing = []
        try:
            import fitz
        except ImportError:
            self._missing.append("pymupdf  (pip install pymupdf)")
        try:
            import ollama
        except ImportError:
            self._missing.append("ollama  (pip install ollama)")

        if self._missing:
            self._build_missing_deps_ui()
        else:
            self._build_ui()

        self._center()

    def _center(self):
        self.update_idletasks()
        w, h = 1140, 820
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self.minsize(820, 600)

    def _refocus(self):
        self.lift(); self.focus_force()
        self.attributes("-topmost", True)
        self.after(150, lambda: self.attributes("-topmost", False))

    # ── Missing deps notice ───────────────────────────────────────────────

    def _build_missing_deps_ui(self):
        f = tk.Frame(self, bg=COLORS["bg"], padx=30, pady=30)
        f.pack(fill="both", expand=True)
        tk.Label(f, text="Missing dependencies",
                 bg=COLORS["bg"], fg=COLORS["danger"],
                 font=FONT_HEADING).pack(anchor="w")
        tk.Label(f,
                 text="Install the following packages then restart the tool:",
                 bg=COLORS["bg"], fg=COLORS["text"],
                 font=FONT_BODY).pack(anchor="w", pady=(8, 4))
        for pkg in self._missing:
            tk.Label(f, text=f"  pip install {pkg}",
                     bg=COLORS["bg"], fg=COLORS["text"],
                     font=FONT_MONO).pack(anchor="w")
        tk.Label(f,
                 text=f"\nAlso ensure Ollama is running:\n"
                      f"  ollama pull {self.DEFAULT_MODEL}",
                 bg=COLORS["bg"], fg=COLORS["text_muted"],
                 font=FONT_SMALL).pack(anchor="w", pady=(12, 0))

    # ── Main UI ───────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=COLORS["header_bg"], padx=20, pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text="IFU Verification",
                 bg=COLORS["header_bg"], fg="white",
                 font=FONT_TITLE).pack(side="left")
        meta = ""
        if self._plan_data:
            title = getattr(self._plan_data, "doc_title", "") or ""
            er    = getattr(self._plan_data, "er_number", "") or ""
            meta  = (f"  ·  {title[:50]}  |  ER: {er}"
                     f"  |  {len(self._prds)} IFU PRD(s)")
        tk.Label(hdr, text=meta,
                 bg=COLORS["header_bg"], fg="#BFDBFE",
                 font=FONT_SMALL).pack(side="left", padx=(8, 0))

        # ── Input bar ─────────────────────────────────────────────────────
        bar = tk.Frame(self, bg=COLORS["bg_card"],
                       highlightthickness=1,
                       highlightbackground=COLORS["border"],
                       padx=16, pady=10)
        bar.pack(fill="x")
        bar.columnconfigure(1, weight=1)

        # IFU file
        tk.Label(bar, text="IFU Document (PDF):",
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONT_SMALL).grid(row=0, column=0, sticky="w", padx=(0, 8))
        tk.Entry(bar, textvariable=self._ifu_path, state="readonly",
                 bg=COLORS["bg_input"], fg=COLORS["text"],
                 readonlybackground=COLORS["bg_input"],
                 relief="flat", font=FONT_MONO,
                 highlightthickness=1,
                 highlightbackground=COLORS["border"]).grid(
            row=0, column=1, sticky="ew", padx=(0, 8))
        tk.Button(bar, text="Browse…",
                  command=self._browse_ifu,
                  bg=COLORS["bg_card"], fg=COLORS["text"],
                  activebackground=COLORS["border"],
                  relief="flat", cursor="hand2",
                  font=FONT_SMALL, padx=10, pady=4, bd=0,
                  highlightthickness=1,
                  highlightbackground=COLORS["border"]).grid(row=0, column=2)

        # Ollama model
        tk.Label(bar, text="Ollama model:",
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONT_SMALL).grid(row=1, column=0, sticky="w",
                                       padx=(0, 8), pady=(6, 0))
        tk.Entry(bar, textvariable=self._model_var,
                 bg=COLORS["bg_input"], fg=COLORS["text"],
                 relief="flat", font=FONT_MONO,
                 highlightthickness=1,
                 highlightbackground=COLORS["border"],
                 width=24).grid(row=1, column=1, sticky="w", pady=(6, 0))

        self._ollama_status_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self._ollama_status_var,
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONT_SMALL).grid(row=1, column=2, sticky="w",
                                       padx=(8, 0), pady=(6, 0))
        tk.Button(bar, text="Check Ollama",
                  command=self._check_ollama,
                  bg=COLORS["bg_card"], fg=COLORS["text"],
                  activebackground=COLORS["border"],
                  relief="flat", cursor="hand2",
                  font=FONT_SMALL, padx=8, pady=3, bd=0,
                  highlightthickness=1,
                  highlightbackground=COLORS["border"]).grid(
            row=1, column=3, padx=(8, 0), pady=(6, 0))

        # ── Paned: cards left, log right ──────────────────────────────────
        paned = tk.PanedWindow(self, orient="horizontal",
                               bg=COLORS["border"],
                               sashwidth=5, sashrelief="flat")
        paned.pack(fill="both", expand=True)
        left  = self._build_cards_panel(paned)
        right = self._build_log_panel(paned)
        paned.add(left,  minsize=600, stretch="always")
        paned.add(right, minsize=260, stretch="never")

        # ── Footer ────────────────────────────────────────────────────────
        footer = tk.Frame(self, bg=COLORS["bg_card"],
                          highlightthickness=1,
                          highlightbackground=COLORS["border"],
                          padx=16, pady=8)
        footer.pack(fill="x", side="bottom")

        self._summary_var = tk.StringVar(value="")
        tk.Label(footer, textvariable=self._summary_var,
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONT_SMALL).pack(side="left")

        self._progress_var = tk.DoubleVar(value=0)
        self._pb = ttk.Progressbar(footer, variable=self._progress_var,
                                   maximum=100, length=180,
                                   mode="determinate")
        self._pb.pack(side="left", padx=(16, 0))

        tk.Button(footer, text="Export PDF Report",
                  command=self._export_pdf,
                  bg=COLORS["accent2"], fg="white",
                  activebackground=_lighten(COLORS["accent2"]),
                  relief="flat", cursor="hand2",
                  font=FONT_BODY, padx=14, pady=6, bd=0).pack(side="right")

        self._run_btn = tk.Button(
            footer, text="▶  Run All Verifications",
            command=self._run_all,
            bg=COLORS["accent"], fg="white",
            activebackground=_lighten(COLORS["accent"]),
            relief="flat", cursor="hand2",
            font=(FONT_BODY[0], FONT_BODY[1], "bold"),
            padx=14, pady=6, bd=0)
        self._run_btn.pack(side="right", padx=(0, 8))

        self._update_summary()

    # ── Cards panel ───────────────────────────────────────────────────────

    def _build_cards_panel(self, parent):
        frame  = tk.Frame(parent, bg=COLORS["bg"])
        canvas = tk.Canvas(frame, bg=COLORS["bg"], highlightthickness=0)
        vsb    = tk.Scrollbar(frame, orient="vertical",
                              command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=COLORS["bg"])
        win   = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(
                       scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win, width=e.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(
                            -1 * (e.delta // 120), "units"))

        if not self._prds:
            tk.Label(inner,
                     text="No IFU PRDs found in the V&V Plan.\n"
                          "Load a V&V Plan from the home screen first.",
                     bg=COLORS["bg"], fg=COLORS["text_muted"],
                     font=FONT_BODY, justify="center").pack(
                pady=40, padx=20)
        else:
            for prd in self._prds:
                card = self._make_prd_card(inner, prd)
                self._cards.append(card)

        return frame

    # ── Log panel ─────────────────────────────────────────────────────────

    def _build_log_panel(self, parent):
        frame = tk.Frame(parent, bg=COLORS["bg"])
        tk.Label(frame, text="Verification Log",
                 bg=COLORS["bg"], fg=COLORS["text_muted"],
                 font=FONT_SMALL).pack(anchor="w", padx=8, pady=(8, 2))
        self._log_text = scrolledtext.ScrolledText(
            frame, bg=COLORS["log_bg"], fg=COLORS["log_text"],
            font=FONT_MONO, state="disabled", relief="flat", wrap="word")
        self._log_text.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self._log_text.tag_config("OK",    foreground=COLORS["success"])
        self._log_text.tag_config("WARN",  foreground=COLORS["warn"])
        self._log_text.tag_config("ERROR", foreground=COLORS["danger"])
        return frame

    # ── PRD card ──────────────────────────────────────────────────────────

    def _make_prd_card(self, parent, prd) -> dict:
        card = tk.Frame(parent, bg=COLORS["bg_card"],
                        highlightthickness=1,
                        highlightbackground=COLORS["border"])
        card.pack(fill="x", padx=12, pady=6)

        # Amber accent bar (IFU colour)
        tk.Frame(card, bg=COLORS["accent3"], height=3).pack(fill="x")

        body = tk.Frame(card, bg=COLORS["bg_card"], padx=14, pady=10)
        body.pack(fill="x")
        body.columnconfigure(2, weight=1)

        # Row 0: badge + method
        badge = tk.Frame(body, bg=COLORS["accent3"])
        badge.grid(row=0, column=0, sticky="nw", padx=(0, 10), pady=(0, 4))
        tk.Label(badge, text=prd.prd_id,
                 bg=COLORS["accent3"], fg="#0F172A",
                 font=("Segoe UI", 8, "bold"), padx=8, pady=3).pack()

        method_text = getattr(prd, "vv_method", "") or "Inspection"
        tk.Label(body, text=f"Method: {method_text}  |  IFU requirement",
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONT_SMALL).grid(row=0, column=1, sticky="w")

        # Row 1: requirement text
        tk.Label(body, text=prd.requirement_text,
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONT_SMALL, anchor="w",
                 wraplength=540, justify="left").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(0, 6))

        # Row 2: verdict area (populated after LLM runs)
        verdict_frame = tk.Frame(body, bg=COLORS["bg_card"])
        verdict_frame.grid(row=2, column=0, columnspan=3, sticky="ew",
                           pady=(0, 4))

        verdict_var  = tk.StringVar(value="PENDING")
        conf_var     = tk.StringVar(value="")
        reason_var   = tk.StringVar(value="")
        evidence_var = tk.StringVar(value="")
        ev_page_var  = tk.IntVar(value=0)
        ev_para_var  = tk.StringVar(value="")

        verdict_lbl = tk.Label(verdict_frame, text="PENDING",
                               bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                               font=("Segoe UI", 9, "bold"))
        verdict_lbl.pack(side="left", padx=(0, 10))

        conf_lbl = tk.Label(verdict_frame, textvariable=conf_var,
                            bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                            font=FONT_SMALL)
        conf_lbl.pack(side="left")

        # Row 3: reason text
        reason_lbl = tk.Label(body, textvariable=reason_var,
                              bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                              font=FONT_SMALL, anchor="w",
                              wraplength=520, justify="left")
        reason_lbl.grid(row=3, column=0, columnspan=3, sticky="w",
                        pady=(0, 4))

        # Row 4: evidence text (italic)
        evidence_lbl = tk.Label(body, textvariable=evidence_var,
                                bg=COLORS["bg_card"],
                                fg=COLORS["accent"],
                                font=(FONT_SMALL[0], FONT_SMALL[1], "italic"),
                                anchor="w", wraplength=520, justify="left")
        evidence_lbl.grid(row=4, column=0, columnspan=3, sticky="w",
                          pady=(0, 6))

        # Row 5: override buttons + notes
        action_row = tk.Frame(body, bg=COLORS["bg_card"])
        action_row.grid(row=5, column=0, columnspan=3, sticky="ew",
                        pady=(4, 0))

        def _mark(v, label_text, colour):
            verdict_var.set(v)
            verdict_lbl.configure(text=label_text, fg=colour)
            self._update_summary()

        tk.Button(action_row, text="✔  Pass",
                  command=lambda: _mark("PASS", "✔  PASS",
                                        COLORS["success"]),
                  bg=COLORS["success"], fg="white",
                  activebackground=_lighten(COLORS["success"]),
                  relief="flat", cursor="hand2",
                  font=FONT_SMALL, padx=10, pady=5, bd=0).pack(
            side="left", padx=(0, 4))
        tk.Button(action_row, text="✗  Fail",
                  command=lambda: _mark("FAIL", "✗  FAIL",
                                        COLORS["danger"]),
                  bg=COLORS["danger"], fg="white",
                  activebackground=_lighten(COLORS["danger"]),
                  relief="flat", cursor="hand2",
                  font=FONT_SMALL, padx=10, pady=5, bd=0).pack(
            side="left", padx=(0, 4))
        tk.Button(action_row, text="○  Manual Review",
                  command=lambda: _mark("MANUAL", "○  MANUAL REVIEW",
                                        COLORS["warn"]),
                  bg=COLORS["accent3"], fg="white",
                  activebackground=_lighten(COLORS["accent3"]),
                  relief="flat", cursor="hand2",
                  font=FONT_SMALL, padx=10, pady=5, bd=0).pack(
            side="left")

        # Row 6: notes
        notes_row = tk.Frame(body, bg=COLORS["bg_card"])
        notes_row.grid(row=6, column=0, columnspan=3, sticky="ew",
                       pady=(6, 0))
        tk.Label(notes_row, text="Engineer notes:",
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONT_SMALL).pack(anchor="w")
        notes_text = tk.Text(notes_row, height=2,
                             bg=COLORS["bg_input"], fg=COLORS["text"],
                             font=FONT_SMALL, relief="flat", wrap="word",
                             highlightthickness=1,
                             highlightbackground=COLORS["border"])
        notes_text.pack(fill="x")

        return {
            "prd":         prd,
            "verdict_var": verdict_var,
            "conf_var":    conf_var,
            "reason_var":  reason_var,
            "evidence_var": evidence_var,
            "ev_page_var": ev_page_var,
            "ev_para_var": ev_para_var,
            "verdict_lbl": verdict_lbl,
            "conf_lbl":    conf_lbl,
            "notes_text":  notes_text,
        }

    # ── Ollama check ──────────────────────────────────────────────────────

    def _check_ollama(self):
        model = self._model_var.get().strip() or self.DEFAULT_MODEL
        self._ollama_status_var.set("Checking…")
        self.update_idletasks()

        def _thread():
            try:
                from ifu_llm_checker import IFULLMChecker
                checker  = IFULLMChecker(model=model)
                ok, msg  = checker.is_available()
                level    = "OK" if ok else "ERROR"
                self.after(0, self._ollama_status_var.set, msg)
                self.after(0, self._log, msg, level)
            except Exception as e:
                err = f"Ollama check error: {e}"
                self.after(0, self._ollama_status_var.set, err)
                self.after(0, self._log, err, "ERROR")

        threading.Thread(target=_thread, daemon=True).start()

    # ── Browse IFU ────────────────────────────────────────────────────────

    def _browse_ifu(self):
        p = filedialog.askopenfilename(
            parent=self,
            title="Select IFU Document (PDF)",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if p:
            self._ifu_path.set(p)
            self._log(f"IFU document: {os.path.basename(p)}", "OK")
        self._refocus()

    # ── Verification ──────────────────────────────────────────────────────

    def _run_all(self):
        ifu_path = self._ifu_path.get().strip()
        if not ifu_path:
            messagebox.showwarning(
                "No IFU Document",
                "Please browse for the IFU PDF first.",
                parent=self)
            return
        if not os.path.isfile(ifu_path):
            messagebox.showerror("File Not Found",
                                 f"IFU file not found:\n{ifu_path}",
                                 parent=self)
            return
        if self._running:
            return

        model = self._model_var.get().strip() or self.DEFAULT_MODEL
        self._run_btn.configure(state="disabled",
                                text="Running…")
        self._running = True
        self._progress_var.set(0)

        def _thread():
            try:
                self._log(f"Extracting text from IFU…")
                pages = extract_pdf_text(ifu_path)
                if not pages:
                    self.after(0, self._log,
                               "Could not extract text from IFU PDF.", "ERROR")
                    return

                self._log(f"IFU: {len(pages)} pages extracted")

                from ifu_llm_checker import IFULLMChecker
                checker = IFULLMChecker(model=model)

                total = len(self._cards)
                for idx, card in enumerate(self._cards):
                    prd = card["prd"]
                    self._log(f"[{idx+1}/{total}] {prd.prd_id} — checking…")
                    try:
                        result = checker.check(prd.requirement_text, pages)
                        self.after(0, self._apply_llm_result, card, result)
                    except Exception as e:
                        self.after(0, self._log,
                                   f"{prd.prd_id}: error — {e}", "ERROR")

                    pct = int((idx + 1) / total * 100)
                    self.after(0, self._progress_var.set, pct)

                self.after(0, self._log,
                           f"All {total} PRD(s) verified.", "OK")
                self.after(0, self._update_summary)

            except Exception as e:
                import traceback
                self.after(0, self._log,
                           f"Run error: {e}\n{traceback.format_exc()}",
                           "ERROR")
            finally:
                self.after(0, lambda: self._run_btn.configure(
                    state="normal", text="▶  Run All Verifications"))
                self.after(0, lambda: setattr(self, "_running", False))

        threading.Thread(target=_thread, daemon=True).start()

    def _apply_llm_result(self, card: dict, result):
        """Update a card's UI with the LLM result (called on main thread)."""
        satisfied  = getattr(result, "satisfied", False)
        confidence = getattr(result, "confidence", "low")
        reason     = getattr(result, "reason", "")
        ev_text    = getattr(result, "evidence_text", "")
        ev_page    = getattr(result, "evidence_page", 0)
        ev_para    = getattr(result, "evidence_para", "")
        error      = getattr(result, "error", "")

        if error:
            verdict   = "MANUAL"
            lbl_text  = "○  ERROR — manual review"
            lbl_colour = COLORS["warn"]
        elif satisfied and confidence in ("high", "medium"):
            verdict    = "PASS"
            lbl_text   = "✔  PASS (LLM)"
            lbl_colour  = COLORS["success"]
        elif not satisfied:
            verdict    = "FAIL"
            lbl_text   = "✗  FAIL (LLM)"
            lbl_colour  = COLORS["danger"]
        else:
            verdict    = "MANUAL"
            lbl_text   = "○  REVIEW (low confidence)"
            lbl_colour  = COLORS["warn"]

        card["verdict_var"].set(verdict)
        card["verdict_lbl"].configure(text=lbl_text, fg=lbl_colour)
        card["conf_var"].set(f"Confidence: {confidence}")
        card["reason_var"].set(reason)
        card["ev_page_var"].set(ev_page)
        card["ev_para_var"].set(ev_para)

        ev_display = ""
        if ev_text:
            ev_display = f'Evidence (p.{ev_page}): "{ev_text[:120]}"'
        card["evidence_var"].set(ev_display)

        prd_id = getattr(card["prd"], "prd_id", "?")
        self._log(
            f"{prd_id}: {lbl_text}  confidence={confidence}",
            "OK" if verdict == "PASS"
            else ("ERROR" if verdict == "FAIL" else "WARN"))
        self._update_summary()

    # ── Summary ───────────────────────────────────────────────────────────

    def _update_summary(self):
        total   = len(self._cards)
        passed  = sum(1 for c in self._cards
                      if c["verdict_var"].get() == "PASS")
        failed  = sum(1 for c in self._cards
                      if c["verdict_var"].get() == "FAIL")
        manual  = sum(1 for c in self._cards
                      if c["verdict_var"].get() == "MANUAL")
        pending = total - passed - failed - manual
        self._summary_var.set(
            f"{total} IFU PRD(s)  ·  "
            f"✔ {passed} Pass  ✗ {failed} Fail  "
            f"○ {manual} Manual  · {pending} Pending"
        )

    # ── Log ───────────────────────────────────────────────────────────────

    def _log(self, msg: str, level: str = "INFO"):
        try:
            self._log_text.configure(state="normal")
            prefix = {"OK": "✔ ", "WARN": "⚠ ", "ERROR": "✗ "}.get(level, "› ")
            self._log_text.insert("end", prefix + msg + "\n",
                                  level if level in ("OK", "WARN", "ERROR") else "")
            self._log_text.see("end")
            self._log_text.configure(state="disabled")
        except Exception:
            pass

    # ── Export PDF ────────────────────────────────────────────────────────

    def _export_pdf(self):
        if not self._cards:
            messagebox.showwarning("Nothing to Export",
                                   "No PRD cards to export.",
                                   parent=self)
            return

        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        default = f"IFU_LLM_Report_{ts}.pdf"
        path    = filedialog.asksaveasfilename(
            parent=self,
            title="Save IFU Verification Report",
            initialfile=default,
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if not path:
            self._refocus(); return

        self._log(f"Exporting report to {os.path.basename(path)}…")

        def _build():
            try:
                from ifu_pdf_reporter import (
                    build_ifu_report, IFUVerificationResult)

                ifu_path = self._ifu_path.get()
                model    = self._model_var.get().strip() or self.DEFAULT_MODEL
                results  = []
                for cd in self._cards:
                    prd = cd["prd"]
                    results.append(IFUVerificationResult(
                        prd_id        = getattr(prd, "prd_id", "?"),
                        requirement   = getattr(prd, "requirement_text", ""),
                        ifu_path      = ifu_path,
                        verdict       = cd["verdict_var"].get(),
                        confidence    = cd["conf_var"].get().replace(
                            "Confidence: ", "") or "low",
                        reason        = cd["reason_var"].get(),
                        evidence_text = cd["evidence_var"].get(),
                        evidence_page = cd["ev_page_var"].get(),
                        evidence_para = cd["ev_para_var"].get(),
                        notes         = cd["notes_text"].get(
                            "1.0", "end").strip(),
                        model_used    = model,
                    ))

                out = build_ifu_report(results, self._plan_data, path)
                if out == path:
                    self.after(0, self._log,
                               f"Report saved: {path}", "OK")
                    self.after(0, messagebox.showinfo,
                               "Report Saved",
                               f"IFU Verification Report saved to:\n{path}")
                else:
                    self.after(0, self._log, f"Export error: {out}", "ERROR")
                    self.after(0, messagebox.showerror, "Export Error", out)

            except Exception as e:
                import traceback
                err = f"Export failed: {e}\n{traceback.format_exc()}"
                self.after(0, self._log, err, "ERROR")
                self.after(0, messagebox.showerror, "Export Error", str(e))

        threading.Thread(target=_build, daemon=True).start()
        self._refocus()
