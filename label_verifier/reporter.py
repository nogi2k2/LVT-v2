from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import io
import os
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# A4 usable width = 210mm - 40mm left - 40mm right = 130mm ≈ 7.28 inch
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = A4
LEFT_MARGIN = RIGHT_MARGIN = 40
USABLE_W = PAGE_W - LEFT_MARGIN - RIGHT_MARGIN   # points (~519 pt ≈ 7.26 inch)

# Philips-brand colour palette
PHILIPS_BLUE   = colors.HexColor('#0B54A4')
HEADER_BG      = colors.HexColor('#E8EEF6')
ROW_ALT        = colors.HexColor('#F7F9FC')
PASS_GREEN     = colors.HexColor('#1A7F4B')
FAIL_RED       = colors.HexColor('#C0392B')
BORDER_GREY    = colors.HexColor('#B0B8C4')


def _pil_from_array(arr):
    """Convert a numpy BGR / grayscale array OR PIL Image to a PIL Image (RGB)."""
    if arr is None:
        return None
    try:
        import numpy as np
        if isinstance(arr, PILImage.Image):
            return arr.convert('RGB')
        if hasattr(arr, 'ndim') and arr.ndim == 3 and arr.shape[2] == 3:
            return PILImage.fromarray(arr[:, :, ::-1])   # BGR → RGB
        return PILImage.fromarray(arr)
    except Exception:
        return None


def _img_bytesio(pil_img, max_w_pt, max_h_pt):
    """
    Resize *proportionally* so the image fits within (max_w_pt × max_h_pt) points
    and return (BytesIO, actual_w_pt, actual_h_pt).
    Returns (None, 0, 0) on failure.
    """
    if pil_img is None:
        return None, 0, 0
    try:
        img = pil_img.copy().convert('RGB')
        orig_w, orig_h = img.size
        if orig_w == 0 or orig_h == 0:
            return None, 0, 0
        # Scale to fit box while preserving aspect ratio
        scale = min(max_w_pt / orig_w, max_h_pt / orig_h, 1.0)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = img.resize((new_w, new_h), PILImage.LANCZOS)
        bio = io.BytesIO()
        img.save(bio, format='PNG')
        bio.seek(0)
        return bio, new_w, new_h
    except Exception:
        return None, 0, 0


def _rl_image(pil_img, max_w_pt, max_h_pt, fallback_text='N/A', styles=None):
    """Return a ReportLab Image (aspect-correct) or a Paragraph fallback."""
    bio, w, h = _img_bytesio(pil_img, max_w_pt, max_h_pt)
    if bio:
        return Image(bio, width=w, height=h)
    st = styles or getSampleStyleSheet()
    return Paragraph(fallback_text, st['Normal'])


def _header_footer(canvas, doc, title, timestamp, summary_text):
    canvas.saveState()
    # Top bar
    canvas.setFillColor(PHILIPS_BLUE)
    canvas.rect(0, PAGE_H - 52, PAGE_W, 52, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont('Helvetica-Bold', 13)
    canvas.drawString(LEFT_MARGIN, PAGE_H - 26, title)
    canvas.setFont('Helvetica', 8)
    canvas.drawString(LEFT_MARGIN, PAGE_H - 42, f"Run time: {timestamp}   |   {summary_text}")
    # Page number footer
    canvas.setFillColor(colors.HexColor('#555555'))
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(PAGE_W - RIGHT_MARGIN, 22, f"Page {doc.page}")
    canvas.restoreState()


def _decision_para(decision, styles):
    colour = PASS_GREEN if str(decision).lower() == 'pass' else FAIL_RED
    style = ParagraphStyle('DecCell', parent=styles['Normal'],
                           textColor=colour, fontName='Helvetica-Bold', fontSize=9)
    return Paragraph(str(decision), style)


def _score_colour(score):
    try:
        v = float(score)
        if v >= 0.75:
            return PASS_GREEN
        if v >= 0.55:
            return colors.HexColor('#D4860A')
        return FAIL_RED
    except Exception:
        return colors.black


def build_report(results, summary, output_path, config=None):
    """
    Build an enhanced PDF report summarising verification results.

    Args:
        results  (list)       : List of ResultRecord objects.
        summary  (RunSummary) : Summary of the run.
        output_path (str)     : Path to save the PDF report.
        config   (dict)       : Optional config snapshot.
    """
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=RIGHT_MARGIN, leftMargin=LEFT_MARGIN,
        topMargin=70, bottomMargin=50
    )
    styles = getSampleStyleSheet()
    story  = []

    title     = 'Label Verification Report'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    summary_text = (f"Total: {summary.total}   Passed: {summary.passed}   "
                    f"Failed: {summary.failed}")

    # ── Cover summary block ────────────────────────────────────────────────
    title_style = ParagraphStyle('ReportTitle', parent=styles['Title'],
                                 fontSize=20, textColor=PHILIPS_BLUE, spaceAfter=4)
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 4))

    # Coloured summary pills: Total / Passed / Failed
    pill_data = [[
        Paragraph(f"<b>Total</b>: {summary.total}", styles['Normal']),
        Paragraph(f"<b>Passed</b>: {summary.passed}",
                  ParagraphStyle('P', parent=styles['Normal'], textColor=PASS_GREEN)),
        Paragraph(f"<b>Failed</b>: {summary.failed}",
                  ParagraphStyle('F', parent=styles['Normal'], textColor=FAIL_RED)),
    ]]
    pill_tbl = Table(pill_data, colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
    pill_tbl.setStyle(TableStyle([
        ('BOX',        (0,0), (-1,-1), 1,   PHILIPS_BLUE),
        ('INNERGRID',  (0,0), (-1,-1), 0.5, PHILIPS_BLUE),
        ('BACKGROUND', (0,0), (-1,-1), HEADER_BG),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
    ]))
    story.append(pill_tbl)
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width='100%', thickness=1, color=PHILIPS_BLUE))
    story.append(Spacer(1, 10))

    # ── Summary table ──────────────────────────────────────────────────────
    # Column widths must sum to ≤ USABLE_W (519 pt).
    # Thumb columns: 80 pt each.  Path: fills remaining space.
    THUMB = 80        # points
    COL_DECISION = 48
    COL_SCORE    = 48
    COL_PATH     = USABLE_W - 2*THUMB - COL_DECISION - COL_SCORE  # ~263 pt

    small_style = ParagraphStyle('Small', parent=styles['Normal'],
                                 fontSize=7, leading=9, wordWrap='LTR')
    hdr_style   = ParagraphStyle('Hdr', parent=styles['Normal'],
                                 fontSize=8, fontName='Helvetica-Bold',
                                 textColor=colors.white)

    header_row = [
        Paragraph('Test Label Path', hdr_style),
        Paragraph('Reference Icon', hdr_style),
        Paragraph('Match Found',    hdr_style),
        Paragraph('Decision',       hdr_style),
        Paragraph('Score',          hdr_style),
    ]
    data = [header_row]

    for idx, r in enumerate(results):
        icon_pil  = _pil_from_array(r.icon_snip)
        match_pil = _pil_from_array(r.match_snip)

        icon_cell  = _rl_image(icon_pil,  THUMB, THUMB, 'N/A', styles)
        match_cell = _rl_image(match_pil, THUMB, THUMB, 'N/A', styles)

        path_para = Paragraph(r.input_path, small_style)
        dec_para  = _decision_para(getattr(r, 'decision', ''), styles)

        score_val = getattr(r, 'score', None)
        try:
            score_text = f"{float(score_val):.3f}" if score_val is not None else '—'
        except Exception:
            score_text = str(score_val)
        score_style = ParagraphStyle('Sc', parent=styles['Normal'],
                                     fontSize=9, fontName='Helvetica-Bold',
                                     textColor=_score_colour(score_val),
                                     alignment=1)   # centre
        score_para = Paragraph(score_text, score_style)

        data.append([path_para, icon_cell, match_cell, dec_para, score_para])

    col_widths = [COL_PATH, THUMB, THUMB, COL_DECISION, COL_SCORE]
    tbl = Table(data, colWidths=col_widths, repeatRows=1)

    row_styles = [
        ('BACKGROUND',    (0, 0), (-1, 0),  PHILIPS_BLUE),
        ('GRID',          (0, 0), (-1, -1), 0.4, BORDER_GREY),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN',         (3, 1), (4, -1),  'CENTER'),
        ('TOPPADDING',    (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING',   (0, 0), (-1, -1), 4),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 4),
    ]
    # Alternating row shading
    for i in range(1, len(data)):
        if i % 2 == 0:
            row_styles.append(('BACKGROUND', (0, i), (-1, i), ROW_ALT))

    tbl.setStyle(TableStyle(row_styles))
    story.append(tbl)
    story.append(PageBreak())

    # ── Per-result detail pages ────────────────────────────────────────────
    for r in results:
        heading_style = ParagraphStyle('DH', parent=styles['Heading2'],
                                       textColor=PHILIPS_BLUE)
        story.append(Paragraph(
            f"{r.icon_name} — {os.path.basename(r.input_path)}", heading_style))
        story.append(Paragraph(r.input_path,
                                ParagraphStyle('DP', parent=styles['Normal'],
                                               fontSize=7, textColor=colors.HexColor('#666666'))))
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width='100%', thickness=0.5, color=BORDER_GREY))
        story.append(Spacer(1, 8))

        # Large match image
        match_pil_d = None
        try:
            if getattr(r, 'debug_image_path', None) and os.path.exists(r.debug_image_path):
                match_pil_d = PILImage.open(r.debug_image_path).convert('RGB')
            else:
                match_pil_d = _pil_from_array(r.match_snip)
        except Exception:
            match_pil_d = _pil_from_array(r.match_snip)
        icon_pil_d = _pil_from_array(r.icon_snip)

        MAX_MATCH_W = 4.2 * inch
        MAX_MATCH_H = 5.0 * inch
        MAX_ICON_W  = 2.0 * inch
        MAX_ICON_H  = 2.0 * inch

        match_cell_d = _rl_image(match_pil_d, MAX_MATCH_W, MAX_MATCH_H,
                                  'Match image not available', styles)
        icon_cell_d  = _rl_image(icon_pil_d,  MAX_ICON_W,  MAX_ICON_H,
                                  'Icon not available', styles)

        # Right-column details
        def _safe_float(v):
            try:
                return f"{float(v):.4f}"
            except Exception:
                return str(v) if v is not None else 'N/A'

        details_items = []

        # Decision badge
        details_items.append(_decision_para(getattr(r, 'decision', ''), styles))
        details_items.append(Spacer(1, 6))

        # Icon image
        details_items.append(icon_cell_d)
        details_items.append(Spacer(1, 6))

        # Score breakdown
        score_val  = getattr(r, 'score',          None)
        sim_v      = getattr(r, 'sim',             None)
        sig_v      = getattr(r, 'siglip',          None)
        comb_v     = getattr(r, 'combined_score',  None)

        score_rows = [['Metric', 'Value']]
        score_rows.append(['Display Score', _safe_float(score_val)])
        if sim_v   is not None: score_rows.append(['SIM',      _safe_float(sim_v)])
        if sig_v   is not None: score_rows.append(['SigLIP',   _safe_float(sig_v)])
        if comb_v  is not None: score_rows.append(['Combined', _safe_float(comb_v)])

        sc_tbl = Table(score_rows, colWidths=[1.1*inch, 1.0*inch])
        sc_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, 0),  PHILIPS_BLUE),
            ('TEXTCOLOR',     (0, 0), (-1, 0),  colors.white),
            ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
            ('FONTSIZE',      (0, 0), (-1, -1), 8),
            ('GRID',          (0, 0), (-1, -1), 0.4, BORDER_GREY),
            ('ALIGN',         (1, 1), (1, -1),  'RIGHT'),
            ('TOPPADDING',    (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING',   (0, 0), (-1, -1), 4),
        ]))
        details_items.append(sc_tbl)
        details_items.append(Spacer(1, 6))

        # Pattern verifier details
        if getattr(r, 'pattern_details', None):
            pd = r.pattern_details
            kv_rows = [['Key', 'Value']]
            for k in ('topology_score', 'sift_good', 'akaze_good', 'brisk_good'):
                if k in pd:
                    v = pd[k]
                    # Format floats to 4 dp, leave integers as-is
                    try:
                        fmt_v = f"{float(v):.4f}" if isinstance(v, float) else str(v)
                    except Exception:
                        fmt_v = str(v)
                    # Friendlier display name
                    display_key = k.replace('_', ' ').title()
                    kv_rows.append([display_key, fmt_v])
            if len(kv_rows) > 1:
                # Wider columns: key=1.4", value=0.8" → fits right column cleanly
                kv_tbl = Table(kv_rows, colWidths=[1.4*inch, 0.8*inch])
                kv_tbl.setStyle(TableStyle([
                    ('BACKGROUND',    (0, 0), (-1, 0), HEADER_BG),
                    ('GRID',          (0, 0), (-1,-1), 0.4, BORDER_GREY),
                    ('FONTSIZE',      (0, 0), (-1,-1), 8),
                    ('ALIGN',         (1, 1), (1, -1), 'RIGHT'),
                    ('TOPPADDING',    (0, 0), (-1,-1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1,-1), 3),
                    ('LEFTPADDING',   (0, 0), (-1,-1), 4),
                    ('RIGHTPADDING',  (0, 0), (-1,-1), 4),
                ]))
                details_items.append(Paragraph('Pattern Details:', styles['Normal']))
                details_items.append(kv_tbl)
                details_items.append(Spacer(1, 4))

        # Keypoint counts
        if isinstance(getattr(r, 'kp_counts', None), dict):
            kc = r.kp_counts
            kp_rows = [
                ['KP Icon',   str(kc.get('kp_icon',       'N/A'))],
                ['KP Image',  str(kc.get('kp_image',      'N/A'))],
                ['Good',      str(kc.get('good_matches',  'N/A'))],
                ['Inliers',   str(kc.get('inliers',       'N/A'))],
            ]
            kp_tbl = Table(kp_rows, colWidths=[1.1*inch, 1.0*inch])
            kp_tbl.setStyle(TableStyle([
                ('GRID',       (0,0), (-1,-1), 0.4, BORDER_GREY),
                ('FONTSIZE',   (0,0), (-1,-1), 8),
                ('TOPPADDING', (0,0), (-1,-1), 3),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ]))
            details_items.append(Paragraph('Keypoints:', styles['Normal']))
            details_items.append(kp_tbl)
            details_items.append(Spacer(1, 4))

        # Comment
        if getattr(r, 'comment', None):
            details_items.append(
                Paragraph(f"<i>Comment: {r.comment}</i>",
                          ParagraphStyle('Cm', parent=styles['Normal'], fontSize=8)))

        # Pack detail items into a single-column table so they flow vertically
        RIGHT_COL_W = 2.3 * inch
        details_tbl = Table([[d] for d in details_items], colWidths=[RIGHT_COL_W])
        details_tbl.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'),
                                          ('LEFTPADDING', (0,0), (-1,-1), 0),
                                          ('RIGHTPADDING', (0,0), (-1,-1), 0)]))

        # Side-by-side layout: match image | details column
        MATCH_COL_W = USABLE_W - RIGHT_COL_W - 6
        side_row = Table([[match_cell_d, details_tbl]],
                          colWidths=[MATCH_COL_W, RIGHT_COL_W])
        side_row.setStyle(TableStyle([
            ('VALIGN',       (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING',  (0,0), (-1,-1), 2),
            ('RIGHTPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(side_row)
        story.append(Spacer(1, 12))
        story.append(PageBreak())

    # ── Build with header/footer ───────────────────────────────────────────
    _hf = lambda c, d: _header_footer(c, d, title, timestamp, summary_text)
    try:
        doc.build(story, onFirstPage=_hf, onLaterPages=_hf)
    except Exception as e:
        # Emergency fallback: plain text-only report
        try:
            fb = SimpleDocTemplate(output_path, pagesize=A4)
            fb.build([
                Paragraph(title, getSampleStyleSheet()['Title']),
                Paragraph(f'Report generation error: {e}', getSampleStyleSheet()['Normal'])
            ])
        except Exception:
            pass


def build_icon_report(icon_img, candidates, output_path, title=None):
    """Build a small PDF report for one reference icon and a list of candidate dicts."""
    title = title or 'Icon Candidate Report'
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                             rightMargin=RIGHT_MARGIN, leftMargin=LEFT_MARGIN,
                             topMargin=70, bottomMargin=50)
    styles    = getSampleStyleSheet()
    story     = []
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    story.append(Paragraph(title, ParagraphStyle('T', parent=styles['Title'],
                                                  textColor=PHILIPS_BLUE)))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Generated: {timestamp}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Reference icon
    icon_pil = _pil_from_array(icon_img)
    icon_cell = _rl_image(icon_pil, 3.5*inch, 3.5*inch, 'Icon not available', styles)
    story.append(icon_cell)
    story.append(Spacer(1, 12))

    # Summary table
    hdr_s = ParagraphStyle('H', parent=styles['Normal'],
                            fontName='Helvetica-Bold', textColor=colors.white, fontSize=8)
    data = [[Paragraph(h, hdr_s) for h in ['Rank', 'SIM', 'SigLIP', 'Combined', 'BBox']]]
    for i, c in enumerate(candidates, start=1):
        def _sf(v):
            try: return f"{float(v):.4f}" if v is not None else 'N/A'
            except Exception: return str(v)
        data.append([str(i), _sf(c.get('sim')), _sf(c.get('siglip_sim')),
                     _sf(c.get('combined_score')),
                     str(c.get('bbox') or c.get('bbox_raw') or '')])

    cand_tbl = Table(data, colWidths=[0.6*inch, 0.9*inch, 0.9*inch, 0.9*inch, 3.0*inch])
    cand_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), PHILIPS_BLUE),
        ('GRID',          (0,0), (-1,-1), 0.4, BORDER_GREY),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',         (1,1), (3,-1),  'RIGHT'),
        ('FONTSIZE',      (0,0), (-1,-1), 8),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(cand_tbl)
    story.append(PageBreak())

    # Per-candidate pages
    for i, c in enumerate(candidates, start=1):
        story.append(Paragraph(f"Candidate {i}", styles['Heading2']))
        story.append(Spacer(1, 6))

        crop_pil  = _pil_from_array(c.get('crop'))
        crop_cell = _rl_image(crop_pil, 4.5*inch, 4.5*inch, 'Crop not available', styles)

        def _sf(v):
            try: return f"{float(v):.4f}" if v is not None else 'N/A'
            except Exception: return str(v)

        details = [
            Paragraph(f"SIM: {_sf(c.get('sim'))}", styles['Normal']),
            Paragraph(f"SigLIP: {_sf(c.get('siglip_sim'))}", styles['Normal']),
            Paragraph(f"Combined: {_sf(c.get('combined_score'))}", styles['Normal']),
        ]
        cd = c.get('combined_details', {})
        if cd:
            details.append(Paragraph('Details:', styles['Normal']))
            for k, v in cd.items():
                details.append(Paragraph(f"  {k}: {v}", styles['Normal']))

        RIGHT_COL_W = 2.3 * inch
        det_tbl  = Table([[d] for d in details], colWidths=[RIGHT_COL_W])
        det_tbl.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))

        MATCH_COL_W = USABLE_W - RIGHT_COL_W - 6
        pg_tbl = Table([[crop_cell, det_tbl]], colWidths=[MATCH_COL_W, RIGHT_COL_W])
        pg_tbl.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
        story.append(pg_tbl)
        story.append(PageBreak())

    try:
        doc.build(story)
    except Exception as e:
        try:
            fb = SimpleDocTemplate(output_path, pagesize=A4)
            fb.build([Paragraph(title, getSampleStyleSheet()['Title']),
                      Paragraph(f'Icon report failed: {e}', getSampleStyleSheet()['Normal'])])
        except Exception:
            pass