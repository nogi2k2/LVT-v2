"""
ifu_pdf_reporter.py
====================
Builds the IFU verification PDF report.

For each PRD requirement:
  - Cover page: summary table of all PRDs with PASS/FAIL
  - Detail page per PRD:
      • Requirement text
      • LLM verdict (PASS/FAIL, confidence, reason)
      • Actual IFU page image with yellow paragraph highlight
      • Evidence text in a callout box

Uses PyMuPDF (fitz) to render IFU pages as images and draw highlights.
Uses ReportLab to build the final PDF report.

Public API
----------
build_ifu_report(results, plan_data, output_path)

results    : list[IFUVerificationResult]
plan_data  : VVPlanData (or None)
output_path: str
"""

from __future__ import annotations
import io
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, KeepTogether
)
from PIL import Image as PILImage

# ── Philips palette ─────────────────────────────────────────────────────────
PHILIPS_BLUE = colors.HexColor('#0B54A4')
HEADER_BG    = colors.HexColor('#E8EEF6')
PASS_GREEN   = colors.HexColor('#1A7F4B')
FAIL_RED     = colors.HexColor('#C0392B')
WARN_AMBER   = colors.HexColor('#D97706')
BORDER_GREY  = colors.HexColor('#B0B8C4')
YELLOW_HL    = colors.HexColor('#FFF176')
ROW_ALT      = colors.HexColor('#F7F9FC')

PAGE_W, PAGE_H  = A4
LEFT_M = RIGHT_M = 40
USABLE_W = PAGE_W - LEFT_M - RIGHT_M


# ── Data model ─────────────────────────────────────────────────────────────

@dataclass
class IFUVerificationResult:
    """One result per PRD requirement."""
    prd_id:          str
    requirement:     str
    ifu_path:        str
    verdict:         str           # "PASS" | "FAIL" | "MANUAL"
    confidence:      str           # "high" | "medium" | "low"
    reason:          str
    evidence_text:   str
    evidence_page:   int           # 1-based; 0 = not found
    evidence_para:   str           # paragraph text for highlight
    notes:           str = ""
    model_used:      str = ""


# ── Page image renderer with highlight ─────────────────────────────────────

def render_page_with_highlight(pdf_path: str,
                                page_num: int,
                                paragraph_text: str,
                                dpi: int = 150) -> Optional[PILImage.Image]:
    """
    Render IFU page page_num (1-based) as a PIL image.
    Draw a yellow rectangle behind the paragraph containing paragraph_text.
    Returns PIL Image (RGB) or None on failure.
    """
    try:
        import fitz
    except ImportError:
        return None

    try:
        doc  = fitz.open(pdf_path)
        page = doc[page_num - 1]

        if paragraph_text and len(paragraph_text) > 4:
            search_text = paragraph_text[:40].strip()
            rects = page.search_for(search_text)

            if rects:
                r = rects[0]
                highlight_rect = fitz.Rect(
                    r.x0 - 2,
                    r.y0 - 4,
                    page.rect.width - 2,
                    r.y1 + 4,
                )
                page.draw_rect(highlight_rect,
                               color=(1, 0.95, 0.2),
                               fill=(1, 0.98, 0.4),
                               fill_opacity=0.45,
                               overlay=True)
            else:
                words = paragraph_text.split()[:6]
                for word in words:
                    if len(word) > 4:
                        rects = page.search_for(word)
                        if rects:
                            r = rects[0]
                            highlight_rect = fitz.Rect(
                                r.x0 - 2, r.y0 - 4,
                                page.rect.width - 2, r.y1 + 4,
                            )
                            page.draw_rect(highlight_rect,
                                           color=(1, 0.95, 0.2),
                                           fill=(1, 0.98, 0.4),
                                           fill_opacity=0.45,
                                           overlay=True)
                            break

        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img

    except Exception:
        return None


def _pil_to_rl_image(pil_img: PILImage.Image,
                      max_w: float, max_h: float) -> Optional[Image]:
    """Convert PIL image to ReportLab Image, scaled to fit box."""
    if pil_img is None:
        return None
    try:
        w, h  = pil_img.size
        scale = min(max_w / w, max_h / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        img   = pil_img.resize((nw, nh), PILImage.LANCZOS)
        bio   = io.BytesIO()
        img.save(bio, format="PNG")
        bio.seek(0)
        return Image(bio, width=nw, height=nh)
    except Exception:
        return None


# ── Header / footer ─────────────────────────────────────────────────────────

def _header_footer(canvas, doc, title, timestamp, summary_text):
    canvas.saveState()
    canvas.setFillColor(PHILIPS_BLUE)
    canvas.rect(0, PAGE_H - 52, PAGE_W, 52, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawString(LEFT_M, PAGE_H - 26, title)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(LEFT_M, PAGE_H - 42,
                      f"Generated: {timestamp}   |   {summary_text}")
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(PAGE_W - RIGHT_M, 22, f"Page {doc.page}")
    canvas.restoreState()


# ── Verdict paragraph ────────────────────────────────────────────────────────

def _verdict_para(verdict: str, styles) -> Paragraph:
    colour = (PASS_GREEN if verdict == "PASS"
              else FAIL_RED if verdict == "FAIL"
              else WARN_AMBER)
    st = ParagraphStyle("V", parent=styles["Normal"],
                        textColor=colour,
                        fontName="Helvetica-Bold", fontSize=11)
    icon = {"PASS": "✔", "FAIL": "✗", "MANUAL": "○"}.get(verdict, "?")
    return Paragraph(f"{icon}  {verdict}", st)


# ── Main report builder ──────────────────────────────────────────────────────

def build_ifu_report(results: list,
                     plan_data=None,
                     output_path: str = "ifu_verification_report.pdf") -> str:
    """
    Build the IFU verification PDF report.

    Each PRD gets:
      1. A row in the cover summary table
      2. A detail page with:
         - Requirement text
         - LLM verdict + reason
         - IFU page image with yellow highlight
         - Evidence callout

    Returns output_path on success or error string on failure.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            rightMargin=RIGHT_M, leftMargin=LEFT_M,
            topMargin=70, bottomMargin=50,
        )
        styles = getSampleStyleSheet()
        story  = []

        title     = "IFU Verification Report — EVO MDD"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        total  = len(results)
        passed = sum(1 for r in results if r.verdict == "PASS")
        failed = sum(1 for r in results if r.verdict == "FAIL")
        manual = total - passed - failed
        summary_text = (f"Total: {total}   Pass: {passed}   "
                        f"Fail: {failed}   Manual: {manual}")

        # ── Cover page ────────────────────────────────────────────────────

        title_st = ParagraphStyle("RT", parent=styles["Title"],
                                   fontSize=20, textColor=PHILIPS_BLUE,
                                   spaceAfter=4)
        story.append(Paragraph(title, title_st))
        story.append(Spacer(1, 4))

        if plan_data:
            info_st = ParagraphStyle("PI", parent=styles["Normal"],
                                      fontSize=9,
                                      textColor=colors.HexColor("#444444"))
            story.append(Paragraph(
                f"V&V Plan: {getattr(plan_data, 'doc_title', '')}", info_st))
            story.append(Paragraph(
                f"ER: {getattr(plan_data, 'er_number', '')}   "
                f"Sample size: {getattr(plan_data, 'sample_size', '')}",
                info_st))
            story.append(Spacer(1, 6))

        # Summary pills
        pill_data = [[
            Paragraph(f"<b>Total</b>: {total}", styles["Normal"]),
            Paragraph(f"<b>Pass</b>: {passed}",
                      ParagraphStyle("P", parent=styles["Normal"],
                                     textColor=PASS_GREEN,
                                     fontName="Helvetica-Bold")),
            Paragraph(f"<b>Fail</b>: {failed}",
                      ParagraphStyle("F", parent=styles["Normal"],
                                     textColor=FAIL_RED,
                                     fontName="Helvetica-Bold")),
            Paragraph(f"<b>Manual review</b>: {manual}",
                      ParagraphStyle("M", parent=styles["Normal"],
                                     textColor=WARN_AMBER,
                                     fontName="Helvetica-Bold")),
        ]]
        pill_tbl = Table(pill_data, colWidths=[1.6 * inch] * 4)
        pill_tbl.setStyle(TableStyle([
            ("BOX",           (0, 0), (-1, -1), 1,   PHILIPS_BLUE),
            ("INNERGRID",     (0, 0), (-1, -1), 0.5, PHILIPS_BLUE),
            ("BACKGROUND",    (0, 0), (-1, -1), HEADER_BG),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ]))
        story.append(pill_tbl)
        story.append(Spacer(1, 12))
        story.append(HRFlowable(width="100%", thickness=1, color=PHILIPS_BLUE))
        story.append(Spacer(1, 8))

        # Summary table
        hdr_st = ParagraphStyle("H", parent=styles["Normal"],
                                  fontName="Helvetica-Bold",
                                  textColor=colors.white, fontSize=8)
        sm_st  = ParagraphStyle("SM", parent=styles["Normal"],
                                  fontSize=8, leading=10, wordWrap="LTR")

        COL_PRD  = 1.4 * inch
        COL_VERD = 0.7 * inch
        COL_CONF = 0.7 * inch
        COL_REQ  = USABLE_W - COL_PRD - COL_VERD - COL_CONF

        hdr_row = [Paragraph(t, hdr_st) for t in
                   ["PRD ID", "Requirement", "Verdict", "Confidence"]]
        data = [hdr_row]

        for r in results:
            vcolour = (PASS_GREEN if r.verdict == "PASS"
                       else FAIL_RED if r.verdict == "FAIL"
                       else WARN_AMBER)
            v_st = ParagraphStyle("VS", parent=styles["Normal"],
                                   textColor=vcolour,
                                   fontName="Helvetica-Bold", fontSize=8)
            data.append([
                Paragraph(r.prd_id, sm_st),
                Paragraph(r.requirement[:120] +
                          ("…" if len(r.requirement) > 120 else ""), sm_st),
                Paragraph(r.verdict, v_st),
                Paragraph(r.confidence, sm_st),
            ])

        sum_tbl = Table(data,
                        colWidths=[COL_PRD, COL_REQ, COL_VERD, COL_CONF],
                        repeatRows=1)
        row_styles = [
            ("BACKGROUND",    (0, 0), (-1, 0),  PHILIPS_BLUE),
            ("GRID",          (0, 0), (-1, -1), 0.4, BORDER_GREY),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ]
        # Alternating row colours
        for i in range(1, len(data)):
            if i % 2 == 0:
                row_styles.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))

        sum_tbl.setStyle(TableStyle(row_styles))
        story.append(sum_tbl)

        # ── Detail pages (one per PRD) ────────────────────────────────────

        req_st = ParagraphStyle("REQ", parent=styles["Normal"],
                                 fontSize=9, leading=13,
                                 textColor=colors.HexColor("#1A1A2E"))
        lbl_st = ParagraphStyle("LBL", parent=styles["Normal"],
                                 fontSize=8, fontName="Helvetica-Bold",
                                 textColor=colors.HexColor("#555555"))
        ev_st  = ParagraphStyle("EV", parent=styles["Normal"],
                                 fontSize=8, leading=12, italic=True,
                                 textColor=colors.HexColor("#1A3A5C"))
        note_st = ParagraphStyle("NT", parent=styles["Normal"],
                                  fontSize=8, leading=12,
                                  textColor=colors.HexColor("#374151"))

        for r in results:
            story.append(PageBreak())

            # PRD ID header bar
            prd_colour = (PASS_GREEN if r.verdict == "PASS"
                          else FAIL_RED if r.verdict == "FAIL"
                          else WARN_AMBER)
            id_st = ParagraphStyle("ID", parent=styles["Normal"],
                                    fontSize=13, fontName="Helvetica-Bold",
                                    textColor=colors.white)
            id_tbl = Table([[Paragraph(r.prd_id, id_st)]],
                           colWidths=[USABLE_W])
            id_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), prd_colour),
                ("TOPPADDING",    (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ]))
            story.append(id_tbl)
            story.append(Spacer(1, 6))

            # Requirement box
            req_tbl = Table([
                [Paragraph("Requirement", lbl_st)],
                [Paragraph(r.requirement, req_st)],
            ], colWidths=[USABLE_W])
            req_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (0, 0), HEADER_BG),
                ("BACKGROUND",    (0, 1), (0, 1), colors.white),
                ("BOX",           (0, 0), (-1, -1), 0.5, BORDER_GREY),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ]))
            story.append(req_tbl)
            story.append(Spacer(1, 6))

            # Verdict + reason row
            conf_colour = (PASS_GREEN if r.confidence == "high"
                           else WARN_AMBER if r.confidence == "medium"
                           else FAIL_RED)
            conf_st = ParagraphStyle("CONF", parent=styles["Normal"],
                                      fontSize=9, textColor=conf_colour,
                                      fontName="Helvetica-Bold")

            verdict_row = [
                [Paragraph("Verdict", lbl_st),
                 Paragraph("Confidence", lbl_st),
                 Paragraph("Reason", lbl_st)],
                [_verdict_para(r.verdict, styles),
                 Paragraph(r.confidence.upper(), conf_st),
                 Paragraph(r.reason or "—", req_st)],
            ]
            vrd_tbl = Table(verdict_row,
                            colWidths=[1.1 * inch, 1.0 * inch,
                                       USABLE_W - 2.1 * inch])
            vrd_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), HEADER_BG),
                ("BACKGROUND",    (0, 1), (-1, 1), colors.white),
                ("BOX",           (0, 0), (-1, -1), 0.5, BORDER_GREY),
                ("INNERGRID",     (0, 0), (-1, -1), 0.3, BORDER_GREY),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(vrd_tbl)
            story.append(Spacer(1, 8))

            # IFU page image (left) + evidence callout (right)
            IFU_IMG_W = USABLE_W * 0.60
            EV_BOX_W  = USABLE_W * 0.36
            GAP       = USABLE_W - IFU_IMG_W - EV_BOX_W

            page_img = None
            if r.ifu_path and r.evidence_page > 0:
                pil_img  = render_page_with_highlight(
                    r.ifu_path, r.evidence_page, r.evidence_para)
                page_img = _pil_to_rl_image(pil_img, IFU_IMG_W, 5.5 * inch)

            if page_img is None:
                page_img = Paragraph(
                    f"[IFU page {r.evidence_page or '?'} — not rendered]",
                    ParagraphStyle("NA", parent=styles["Normal"],
                                   fontSize=8,
                                   textColor=colors.HexColor("#888888")))

            # Evidence callout
            ev_lines = []
            if r.evidence_page > 0:
                ev_lines.append(Paragraph(
                    f"Evidence found on page {r.evidence_page}", lbl_st))
                ev_lines.append(Spacer(1, 4))
            if r.evidence_text:
                ev_lines.append(Paragraph(
                    f'"{r.evidence_text}"', ev_st))
                ev_lines.append(Spacer(1, 6))
            if r.model_used:
                ev_lines.append(Paragraph(
                    f"Model: {r.model_used}", note_st))
            if r.notes:
                ev_lines.append(Spacer(1, 6))
                ev_lines.append(Paragraph("Engineer notes:", lbl_st))
                ev_lines.append(Paragraph(r.notes, note_st))
            if not ev_lines:
                ev_lines = [Paragraph("No evidence found.", note_st)]

            side_tbl = Table(
                [[page_img, Spacer(GAP, 1),
                  Table([[e] for e in ev_lines],
                        colWidths=[EV_BOX_W])]],
                colWidths=[IFU_IMG_W, GAP, EV_BOX_W]
            )
            side_tbl.setStyle(TableStyle([
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING",    (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
                ("BOX",           (2, 0), (2, 0), 0.5, BORDER_GREY),
                ("BACKGROUND",    (2, 0), (2, 0), YELLOW_HL),
                ("TOPPADDING",    (2, 0), (2, 0), 8),
                ("LEFTPADDING",   (2, 0), (2, 0), 8),
                ("RIGHTPADDING",  (2, 0), (2, 0), 8),
                ("BOTTOMPADDING", (2, 0), (2, 0), 8),
            ]))
            story.append(side_tbl)

        # ── Build PDF ──────────────────────────────────────────────────────

        def _on_page(canvas, doc_obj):
            _header_footer(canvas, doc_obj, title, timestamp, summary_text)

        doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
        return output_path

    except Exception as e:
        import traceback
        return f"Report generation failed: {e}\n{traceback.format_exc()}"
