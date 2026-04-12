"""
ISO 15223 Symbol Extractor — Window 2
========================================
Extracts all symbols from an ISO 15223 standard PDF.
Core extraction logic preserved from original extract_symbols.py.
Enhanced with:
  • Progress pop-up with per-page feedback
  • Modern dark-themed GUI
  • Output folder picker
  • Live preview of extracted symbols
  • JSON/CSV export
"""

import os
import re
import json
import csv
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Default output = "Icon Library" folder next to this file (project root)
_ICON_LIBRARY_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Icon Library")
)

try:
    from app import (
        COLORS, FONT_TITLE, FONT_HEADING, FONT_BODY, FONT_MONO, FONT_SMALL,
        ProgressPopup, _lighten, APP_STATE, fire_event
    )
except ImportError:
    APP_STATE = None
    def fire_event(event, *args): pass
    COLORS = {
        "bg": "#0F172A", "bg_card": "#1E293B", "bg_input": "#0F172A",
        "accent": "#3B82F6", "accent2": "#10B981", "accent3": "#F59E0B",
        "danger": "#EF4444", "text": "#F1F5F9", "text_muted": "#94A3B8",
        "border": "#334155", "header_bg": "#1E3A8A", "success": "#22C55E",
        "warn": "#F59E0B", "log_bg": "#020617", "log_text": "#10B981",
    }
    FONT_TITLE   = ("Segoe UI", 16, "bold")
    FONT_HEADING = ("Segoe UI", 11, "bold")
    FONT_BODY    = ("Segoe UI", 9)
    FONT_MONO    = ("Consolas", 9)
    FONT_SMALL   = ("Segoe UI", 8)
    def _lighten(h):
        h = h.lstrip("#")
        r,g,b = [int(h[i:i+2],16) for i in (0,2,4)]
        return f"#{min(255,r+30):02x}{min(255,g+30):02x}{min(255,b+30):02x}"


# ══════════════════════════════════════════════════════════════════════════════
# CORE EXTRACTION LOGIC (preserved from original extract_symbols.py)
# ══════════════════════════════════════════════════════════════════════════════

ROW_GRAPHIC_TOP    = 699
ROW_GRAPHIC_BOTTOM = 787
ROW_REF_TOP        = 744
ROW_REF_BOTTOM     = 789
ROW_TITLE_TOP      = 621
ROW_TITLE_BOTTOM   = 699
ROW_DESC_TOP       = 529
ROW_DESC_BOTTOM    = 621
ROW_REQ_TOP        = 437
ROW_REQ_BOTTOM     = 529
ROW_NOTES_TOP      = 254
ROW_NOTES_BOTTOM   = 437
ROW_RESTRICT_TOP   = 177
ROW_RESTRICT_BOTTOM = 254
ROW_ISO_TOP        = 86
ROW_ISO_BOTTOM     = 177

ZOOM       = 5
IMAGE_SIZE = 256


def get_textbox(page, x0, y0, x1, y1):
    import fitz
    rect = fitz.Rect(x0, y0, x1, y1)
    text = page.get_textbox(rect)
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_symbol_columns(page):
    import fitz
    page_width = page.rect.width
    paths  = page.get_drawings()
    v_lines = sorted(set(
        round(p["rect"].x0)
        for p in paths
        if (p.get("rect") and
            abs(p["rect"].width) < 3 and
            p["rect"].height > 50)
    ))

    refs = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t = span["text"].strip()
                if re.match(r"^5\.\d+\.\d+$", t):
                    bbox = span["bbox"]
                    if bbox[1] >= ROW_REF_TOP - 10:
                        refs.append((t, float(bbox[0]), bbox))

    if not refs:
        return []

    refs.sort(key=lambda x: x[1])

    result = []
    for i, (ref, x0_ref, ref_bbox) in enumerate(refs):
        left_divs = [v for v in v_lines if v <= x0_ref + 8]
        col_x0    = max(left_divs) if left_divs else x0_ref - 5

        right_divs = [v for v in v_lines if v > x0_ref + 10]
        if right_divs:
            col_x1 = right_divs[0]
        elif i + 1 < len(refs):
            col_x1 = refs[i + 1][1] - 2
        else:
            col_x1 = page_width - 35

        result.append((col_x0, col_x1, ref, ref_bbox))

    return result


def smart_trim(img):
    import cv2
    import numpy as np
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    row_sum = np.sum(th, axis=1)
    col_sum = np.sum(th, axis=0)
    row_thresh = max(np.max(row_sum) * 0.15, 1)
    col_thresh = max(np.max(col_sum) * 0.15, 1)
    rows = np.where(row_sum > row_thresh)[0]
    cols = np.where(col_sum > col_thresh)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    pad  = 4
    y_min = max(0, rows[0] - pad)
    y_max = min(img.shape[0], rows[-1] + pad)
    x_min = max(0, cols[0] - pad)
    x_max = min(img.shape[1], cols[-1] + pad)
    return img[y_min:y_max, x_min:x_max]


def make_square(img, size=IMAGE_SIZE):
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.ones((size, size, 3), dtype=np.uint8) * 255
    scale  = (size - 20) / max(h, w)
    new_w  = int(w * scale)
    new_h  = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas  = np.ones((size, size, 3), dtype=np.uint8) * 255
    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


def extract_symbols_from_pdf(pdf_path: str, output_dir: str, log_fn, progress_fn=None) -> list[dict]:
    """
    Core extraction — unchanged logic from original extract_symbols.py.
    log_fn(msg, level): callable for progress messages.
    progress_fn(msg): lightweight detail update for popup.
    """
    import fitz
    import cv2
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    doc      = fitz.open(pdf_path)
    mat      = fitz.Matrix(ZOOM, ZOOM)
    all_data = []
    seen_refs = set()

    total_pages = len(doc)
    log_fn(f"PDF has {total_pages} pages", "INFO")

    for page_idx in range(total_pages):
        page = doc[page_idx]
        if progress_fn:
            progress_fn(f"Page {page_idx+1}/{total_pages}…")

        columns = get_symbol_columns(page)
        if not columns:
            continue

        log_fn(f"Page {page_idx+1} — {len(columns)} symbol(s) found", "INFO")

        for col_x0, col_x1, ref, ref_bbox in columns:
            if ref in seen_refs:
                log_fn(f"   Duplicate ref {ref}, skipping", "WARN")
                continue
            seen_refs.add(ref)

            # Extract text fields
            title        = get_textbox(page, col_x0, ROW_TITLE_TOP,     col_x1, ROW_TITLE_BOTTOM)
            description  = get_textbox(page, col_x0, ROW_DESC_TOP,      col_x1, ROW_DESC_BOTTOM)
            requirements = get_textbox(page, col_x0, ROW_REQ_TOP,       col_x1, ROW_REQ_BOTTOM)
            notes        = get_textbox(page, col_x0, ROW_NOTES_TOP,     col_x1, ROW_NOTES_BOTTOM)
            restrictions = get_textbox(page, col_x0, ROW_RESTRICT_TOP,  col_x1, ROW_RESTRICT_BOTTOM)
            iso_number   = get_textbox(page, col_x0, ROW_ISO_TOP,       col_x1, ROW_ISO_BOTTOM)

            # Extract symbol image
            clip_x0 = col_x0 + 1
            clip_y0 = ROW_GRAPHIC_TOP + 1
            clip_x1 = col_x1 - 1
            clip_y1 = ROW_GRAPHIC_BOTTOM

            clip = fitz.Rect(clip_x0, clip_y0, clip_x1, clip_y1)
            pix  = page.get_pixmap(matrix=mat, clip=clip)

            img_array = __import__("numpy").frombuffer(pix.samples, dtype=__import__("numpy").uint8).reshape(
                pix.height, pix.width, pix.n)
            if pix.n == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
            img_array = img_array.copy()

            # Mask out reference text
            rx0, ry0, rx1, ry1 = ref_bbox
            px0 = max(0, int((rx0 - clip_x0) * ZOOM) - 2)
            px1 = min(img_array.shape[1], int((rx1 - clip_x0) * ZOOM) + 2)
            py0 = max(0, int((ry0 - clip_y0) * ZOOM) - 2)
            py1 = min(img_array.shape[0], int((ry1 - clip_y0) * ZOOM) + 2)
            if py0 < py1 and px0 < px1:
                img_array[py0:py1, px0:px1] = 255

            # Smart L-mark removal (preserved from original)
            gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                ink_x0b = min(cv2.boundingRect(c)[0] for c in contours)
                ink_y0b = min(cv2.boundingRect(c)[1] for c in contours)
                ink_x1b = max(cv2.boundingRect(c)[0]+cv2.boundingRect(c)[2] for c in contours)
                ink_y1b = max(cv2.boundingRect(c)[1]+cv2.boundingRect(c)[3] for c in contours)

                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    is_small = (w < img_array.shape[1]*0.15) and (h < img_array.shape[0]*0.15)
                    near_L = abs(x - ink_x0b) < 10
                    near_R = abs((x+w) - ink_x1b) < 10
                    near_T = abs(y - ink_y0b) < 10
                    near_B = abs((y+h) - ink_y1b) < 10
                    in_corner = ((near_L and near_T) or (near_R and near_T) or
                                 (near_L and near_B) or (near_R and near_B))
                    if is_small and in_corner:
                        cv2.drawContours(img_array, [cnt], -1, (255,255,255), thickness=cv2.FILLED)

            img_trimmed = smart_trim(img_array)
            img_rotated = cv2.rotate(img_trimmed, cv2.ROTATE_90_CLOCKWISE)
            img_square  = make_square(img_rotated)

            safe_title = re.sub(r'[^a-zA-Z0-9]+', '_', title).strip('_') or "Unknown"
            filename   = f"{ref.replace('.', '_')}_{safe_title}.png"
            out_path   = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, img_square)

            log_fn(f"   ✔ {ref:12s} | {title[:40]}", "INFO")
            if progress_fn:
                progress_fn(f"Saved: {filename}")

            all_data.append({
                "reference":    ref,
                "title":        title,
                "description":  description,
                "requirements": requirements,
                "notes":        notes,
                "restrictions": restrictions,
                "iso_number":   iso_number,
                "image":        filename,
            })

    doc.close()
    return all_data


def save_json_csv(data: list[dict], output_dir: str, log_fn):
    if not data:
        log_fn("No data to save.", "WARN")
        return
    json_path = os.path.join(output_dir, "iso_symbols.json")
    csv_path  = os.path.join(output_dir, "iso_symbols.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log_fn(f"JSON saved: {json_path}", "INFO")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    log_fn(f"CSV saved: {csv_path}", "INFO")


# ══════════════════════════════════════════════════════════════════════════════
# GUI — Extractor Window
# ══════════════════════════════════════════════════════════════════════════════

class ExtractorWindow(tk.Toplevel):

    PROCESS_STEPS = [
        "Open ISO 15223 PDF",
        "Detect page columns",
        "Extract symbol images",
        "Extract text metadata",
        "Save JSON + CSV index",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title("ISO 15223 Symbol Extractor — Layer 2")
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)

        self.pdf_path     = tk.StringVar()
        self.output_dir   = tk.StringVar(value=_ICON_LIBRARY_DIR)
        self.extracted    = []
        self._popup       = None

        self._build_ui()
        self._center()

    def _center(self):
        self.update_idletasks()
        w, h = 1100, 750
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _build_ui(self):
        self._setup_styles()

        # Header
        hdr = tk.Frame(self, bg="#1C4A2A", padx=16, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="ISO 15223 Symbol Extractor", bg="#1C4A2A",
                 fg="white", font=FONT_TITLE).pack(side="left")
        tk.Label(hdr, text="  ·  Layer 2 — Symbol Library Builder", bg="#1C4A2A",
                 fg="#86EFAC", font=FONT_BODY).pack(side="left")

        # Main paned
        paned = tk.PanedWindow(self, orient="horizontal", bg=COLORS["bg"],
                               sashwidth=6, sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=10, pady=10)

        left  = tk.Frame(paned, bg=COLORS["bg"])
        right = tk.Frame(paned, bg=COLORS["bg"])
        paned.add(left,  minsize=380)
        paned.add(right, minsize=340)

        self._build_left(left)
        self._build_right(right)

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Ext.TLabelframe",
                         background=COLORS["bg_card"], relief="flat",
                         borderwidth=1, bordercolor=COLORS["border"])
        style.configure("Ext.TLabelframe.Label",
                         background=COLORS["bg_card"],
                         foreground=COLORS["accent2"], font=FONT_HEADING)
        style.configure("Ext.TNotebook", background=COLORS["bg"])
        style.configure("Ext.TNotebook.Tab",
                         background=COLORS["bg_card"], foreground=COLORS["text_muted"],
                         padding=[10, 5], font=FONT_BODY)
        style.map("Ext.TNotebook.Tab",
                  background=[("selected", COLORS["accent2"])],
                  foreground=[("selected", "white")])

    def _section(self, parent, title):
        f = ttk.LabelFrame(parent, text=f"  {title}  ", style="Ext.TLabelframe", padding=(12, 8))
        f.configure(labelanchor="n")
        return f

    def _build_left(self, parent):
        # PDF input
        pdf_sec = self._section(parent, "ISO 15223 Standard PDF")
        pdf_sec.pack(fill="x", pady=(0, 8))

        row = tk.Frame(pdf_sec, bg=COLORS["bg_card"])
        row.pack(fill="x")
        entry = tk.Entry(row, textvariable=self.pdf_path, state="readonly",
                         bg=COLORS["bg_input"], fg=COLORS["text"],
                         readonlybackground=COLORS["bg_input"],
                         relief="flat", font=FONT_MONO,
                         highlightthickness=1, highlightbackground=COLORS["border"])
        entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(row, text="Browse…", command=self._browse_pdf,
                  bg=COLORS["bg_card"], fg=COLORS["text"],
                  activebackground=COLORS["border"], activeforeground=COLORS["text"],
                  relief="flat", cursor="hand2", font=FONT_SMALL,
                  padx=10, pady=5, bd=0,
                  highlightthickness=1, highlightbackground=COLORS["border"]).pack(side="left")
        tk.Label(pdf_sec, text="Must be the ISO 15223-1 standard document with the symbol table layout.",
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"], font=FONT_SMALL).pack(anchor="w", pady=(2,0))

        # Output dir
        out_sec = self._section(parent, "Output Folder (Icon Library/)")
        out_sec.pack(fill="x", pady=(0, 8))

        row2 = tk.Frame(out_sec, bg=COLORS["bg_card"])
        row2.pack(fill="x")
        entry2 = tk.Entry(row2, textvariable=self.output_dir, state="readonly",
                          bg=COLORS["bg_input"], fg=COLORS["text"],
                          readonlybackground=COLORS["bg_input"],
                          relief="flat", font=FONT_MONO,
                          highlightthickness=1, highlightbackground=COLORS["border"])
        entry2.pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(row2, text="Choose…", command=self._browse_outdir,
                  bg=COLORS["bg_card"], fg=COLORS["text"],
                  activebackground=COLORS["border"], activeforeground=COLORS["text"],
                  relief="flat", cursor="hand2", font=FONT_SMALL,
                  padx=10, pady=5, bd=0,
                  highlightthickness=1, highlightbackground=COLORS["border"]).pack(side="left")
        tk.Label(out_sec, text="Symbols saved here replace the Icon Library used by Label Verification. You can change this path if needed.",
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"], font=FONT_SMALL).pack(anchor="w", pady=(2,0))

        # Info box about row coordinates
        info_sec = self._section(parent, "Row Coordinate Settings")
        info_sec.pack(fill="x", pady=(0, 8))

        info_text = (
            f"Graphic rows:     {ROW_GRAPHIC_TOP} – {ROW_GRAPHIC_BOTTOM}\n"
            f"Reference rows:   {ROW_REF_TOP} – {ROW_REF_BOTTOM}\n"
            f"Title rows:       {ROW_TITLE_TOP} – {ROW_TITLE_BOTTOM}\n"
            f"Description rows: {ROW_DESC_TOP} – {ROW_DESC_BOTTOM}\n"
            f"Requirement rows: {ROW_REQ_TOP} – {ROW_REQ_BOTTOM}\n"
            f"Notes rows:       {ROW_NOTES_TOP} – {ROW_NOTES_BOTTOM}\n"
            f"Image zoom:       {ZOOM}x  →  {IMAGE_SIZE}×{IMAGE_SIZE} px output"
        )
        tk.Label(info_sec, text=info_text, bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONT_MONO, justify="left", anchor="w").pack(anchor="w")
        tk.Label(info_sec,
                 text="These match the ISO 15223-1 table layout. Modify source code if your PDF differs.",
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"], font=FONT_SMALL).pack(anchor="w", pady=(4,0))

        # Action buttons
        btn_row = tk.Frame(parent, bg=COLORS["bg"], pady=8)
        btn_row.pack(fill="x")

        run_btn = tk.Button(btn_row, text="▶  Extract All Symbols",
                            command=self._run,
                            bg=COLORS["accent2"], fg="white",
                            activebackground=_lighten(COLORS["accent2"]),
                            activeforeground="white", relief="flat",
                            cursor="hand2", font=FONT_HEADING,
                            padx=18, pady=10, bd=0)
        run_btn.pack(side="left", padx=(0, 10))

        self.open_btn = tk.Button(btn_row, text="📂 Open Output Folder",
                                   command=self._open_folder,
                                   bg=COLORS["border"], fg=COLORS["text"],
                                   activebackground=_lighten(COLORS["border"]),
                                   activeforeground=COLORS["text"], relief="flat",
                                   cursor="hand2", font=FONT_BODY,
                                   padx=12, pady=10, bd=0,
                                   state="disabled")
        self.open_btn.pack(side="left")

        # Stats bar
        self._stats_var = tk.StringVar(value="Ready — select PDF and output folder.")
        tk.Label(btn_row, textvariable=self._stats_var,
                 bg=COLORS["bg"], fg=COLORS["text_muted"], font=FONT_SMALL).pack(side="left", padx=14)

        # Mini progress bar
        self._mini_pb = ttk.Progressbar(parent, orient="horizontal", mode="indeterminate")
        self._mini_pb.pack(fill="x", pady=(0, 2))

        # Log
        log_sec = self._section(parent, "Extraction Log")
        log_sec.pack(fill="both", expand=True)
        self.log_text = scrolledtext.ScrolledText(
            log_sec, bg=COLORS["log_bg"], fg=COLORS["log_text"],
            insertbackground=COLORS["text"],
            font=FONT_MONO, state="disabled", relief="flat", wrap="word", height=10)
        self.log_text.pack(fill="both", expand=True)
        self.log_text.tag_config("INFO",  foreground=COLORS["log_text"])
        self.log_text.tag_config("WARN",  foreground=COLORS["warn"])
        self.log_text.tag_config("ERROR", foreground=COLORS["danger"])

    def _build_right(self, parent):
        """Right side — live symbol preview list."""
        hdr = tk.Frame(parent, bg=COLORS["bg_card"], padx=12, pady=8)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Extracted Symbols Preview", bg=COLORS["bg_card"],
                 fg=COLORS["text"], font=FONT_HEADING).pack(side="left")
        self._count_lbl = tk.Label(hdr, text="0 symbols", bg=COLORS["bg_card"],
                                    fg=COLORS["text_muted"], font=FONT_SMALL)
        self._count_lbl.pack(side="right")

        # Search bar
        search_row = tk.Frame(parent, bg=COLORS["bg"], pady=4)
        search_row.pack(fill="x", padx=4)
        tk.Label(search_row, text="Search:", bg=COLORS["bg"], fg=COLORS["text_muted"],
                 font=FONT_SMALL).pack(side="left", padx=(0, 4))
        self._search_var = tk.StringVar()
        self._search_var.trace("w", self._filter_preview)
        search_entry = tk.Entry(search_row, textvariable=self._search_var,
                                bg=COLORS["bg_input"], fg=COLORS["text"],
                                insertbackground=COLORS["text"], relief="flat",
                                font=FONT_MONO, width=24,
                                highlightthickness=1, highlightbackground=COLORS["border"])
        search_entry.pack(side="left", fill="x", expand=True)

        # Symbol list
        list_frame = tk.Frame(parent, bg=COLORS["bg"])
        list_frame.pack(fill="both", expand=True, padx=4, pady=4)

        cols = ("ref", "title", "image")
        self.sym_tree = ttk.Treeview(list_frame, columns=cols, show="headings", selectmode="browse")
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self.sym_tree.yview)
        self.sym_tree.configure(yscrollcommand=vsb.set)
        self.sym_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        self.sym_tree.heading("ref",   text="Reference")
        self.sym_tree.heading("title", text="Title")
        self.sym_tree.heading("image", text="Filename")
        self.sym_tree.column("ref",   width=80,  minwidth=60)
        self.sym_tree.column("title", width=180, minwidth=80)
        self.sym_tree.column("image", width=200, minwidth=80)

        # Detail panel
        detail_frame = tk.Frame(parent, bg=COLORS["bg_card"], padx=10, pady=8)
        detail_frame.pack(fill="x", padx=4, pady=(0, 4))
        tk.Label(detail_frame, text="Description", bg=COLORS["bg_card"],
                 fg=COLORS["accent2"], font=FONT_SMALL).pack(anchor="w")
        self._detail_text = tk.Text(detail_frame, bg=COLORS["bg_input"], fg=COLORS["text"],
                                     font=FONT_SMALL, relief="flat", height=4, wrap="word",
                                     state="disabled",
                                     highlightthickness=1, highlightbackground=COLORS["border"])
        self._detail_text.pack(fill="x")

        self.sym_tree.bind("<<TreeviewSelect>>", self._on_sym_select)
        self._all_symbols_display = []

    # ── Browsing ──────────────────────────────────────────────────────────

    def _refocus(self):
        """Re-raise this window after a file dialog closes on Windows."""
        self.lift()
        self.focus_force()
        self.attributes("-topmost", True)
        self.after(150, lambda: self.attributes("-topmost", False))

    def _browse_pdf(self):
        p = filedialog.askopenfilename(parent=self, 
            title="Select ISO 15223 PDF",
            filetypes=[("PDF files", "*.pdf *.PDF")])
        if p:
            self.pdf_path.set(p)
            # Auto-set output dir to Icon Library at project root if not yet set
            if not self.output_dir.get():
                self.output_dir.set(_ICON_LIBRARY_DIR)
            self._log("INFO", f"PDF: {os.path.basename(p)}")

        self._refocus()
    def _browse_outdir(self):
        d = filedialog.askdirectory(parent=self, title="Select Output Folder")
        if d:
            self.output_dir.set(d)
            self._log("INFO", f"Output folder: {d}")
        self._refocus()
    def _open_folder(self):
        d = self.output_dir.get()
        if d and os.path.exists(d):
            import subprocess
            import sys as _sys
            if _sys.platform == "win32":
                os.startfile(d)
            elif _sys.platform == "darwin":
                subprocess.Popen(["open", d])
            else:
                subprocess.Popen(["xdg-open", d])

    # ── Logging ───────────────────────────────────────────────────────────
    def _log(self, level: str, msg: str):
        self.log_text.configure(state="normal")
        prefix = {"INFO": "› ", "WARN": "⚠ ", "ERROR": "✗ "}.get(level, "  ")
        self.log_text.insert("end", prefix + msg + "\n", level)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.update_idletasks()

    # ── Run ───────────────────────────────────────────────────────────────
    def _run(self):
        if not self.pdf_path.get():
            messagebox.showerror("Missing PDF", "Please select the ISO 15223 PDF.", parent=self)
            return
        if not self.output_dir.get():
            messagebox.showerror("Missing Folder", "Please choose an output folder.", parent=self)
            return

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        self._stats_var.set("Extracting…")
        self._mini_pb.start(15)
        self.open_btn.configure(state="disabled")

        self._popup = ProgressPopup(self, title="ISO Symbol Extraction", steps=self.PROCESS_STEPS)
        self._popup.set_step(0)

        threading.Thread(target=self._extract_thread, daemon=True).start()

    def _extract_thread(self):
        try:
            def log(msg, level="INFO"):
                self.after(0, self._log, level, msg)

            def progress_fn(detail):
                if self._popup and self._popup.winfo_exists():
                    self.after(0, self._popup.set_detail, detail)

            self.after(0, self._popup.set_step, 0, "Opening PDF…")
            log(f"Opening: {self.pdf_path.get()}", "INFO")

            self.after(0, self._popup.set_step, 1, "Scanning page columns…")
            self.after(0, self._popup.set_step, 2, "Extracting images…")

            data = extract_symbols_from_pdf(
                self.pdf_path.get(),
                self.output_dir.get(),
                log,
                progress_fn,
            )

            self.after(0, self._popup.set_step, 3, "Saving metadata…")
            save_json_csv(data, self.output_dir.get(), log)

            self.after(0, self._popup.set_step, 4, "Done!")
            self.extracted = data
            self.after(0, self._on_done, data)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.after(0, self._log, "ERROR", str(e))
            self.after(0, self._log, "ERROR", tb)
            self.after(0, self._stats_var.set, "Error — see log.")
            self.after(0, self._mini_pb.stop)
            if self._popup:
                self.after(0, self._popup.error, str(e)[:80])

    def _on_done(self, data):
        self._mini_pb.stop()
        n = len(data)
        out_dir = self.output_dir.get()
        self._stats_var.set(f"Extracted {n} symbols → {out_dir}")
        self.open_btn.configure(state="normal")
        self._populate_preview(data)
        if self._popup:
            self._popup.finish(f"{n} symbols extracted successfully!")

        # ── Automation: notify Label Verifier of new icon library path ──────
        if APP_STATE is not None:
            APP_STATE["icon_library_path"] = out_dir
        fire_event("icon_library_ready", out_dir)

    def _populate_preview(self, data):
        for item in self.sym_tree.get_children():
            self.sym_tree.delete(item)
        self._all_symbols_display = data
        for d in data:
            self.sym_tree.insert("", "end", values=(d["reference"], d["title"], d["image"]))
        self._count_lbl.config(text=f"{len(data)} symbols")

    def _filter_preview(self, *_):
        q = self._search_var.get().lower().strip()
        for item in self.sym_tree.get_children():
            self.sym_tree.delete(item)
        for d in self._all_symbols_display:
            if (q in d["reference"].lower() or q in d["title"].lower() or
                    q in d["description"].lower()):
                self.sym_tree.insert("", "end", values=(d["reference"], d["title"], d["image"]))

    def _on_sym_select(self, _event):
        sel = self.sym_tree.selection()
        if not sel:
            return
        vals = self.sym_tree.item(sel[0], "values")
        if not vals:
            return
        ref = vals[0]
        detail = next((d for d in self._all_symbols_display if d["reference"] == ref), None)
        if not detail:
            return
        self._detail_text.configure(state="normal")
        self._detail_text.delete("1.0", "end")
        self._detail_text.insert("end",
            f"Ref: {detail['reference']}  |  ISO: {detail['iso_number']}\n"
            f"Title: {detail['title']}\n"
            f"Description: {detail['description']}\n"
            f"Requirements: {detail['requirements'][:200]}…"
        )
        self._detail_text.configure(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    win = ExtractorWindow(root)
    win.mainloop()