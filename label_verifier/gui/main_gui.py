"""
Label Verification — Main GUI
==============================
Dark-themed, fully responsive Tkinter window.
Wraps the label_verifier controller pipeline with a modern UI.

Tabs (right panel):
  1. Label Verification       — Required Symbols · Results · Preview sub-tabs
  2. IFU / Addendum Verification — inline PRD card verification

V&V Plan upload lives in the left sidebar; parsed state is kept in hidden
widgets and auto-populates both tabs when a plan is loaded.

Multi-label batch mode:
  The label picker supports selecting multiple files at once.
  A deterministic progress bar shows "Label X / N — filename" as
  each file is processed in sequence.
"""

import os
import threading
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

logger = logging.getLogger(__name__)

try:
    from app import (
        COLORS, FONT_TITLE, FONT_HEADING, FONT_BODY, FONT_MONO, FONT_SMALL,
        ProgressPopup, _lighten, APP_STATE, register_callback
    )
except ImportError:
    APP_STATE = None
    def register_callback(event, fn): pass
    COLORS = {
        # Philips light theme
        "bg":           "#F5F7FA",
        "bg_card":      "#FFFFFF",
        "bg_input":     "#FFFFFF",
        "accent":       "#0B54A4",
        "accent2":      "#00A0DC",
        "accent3":      "#E87722",
        "danger":       "#DC2626",
        "text":         "#1A1A2E",
        "text_muted":   "#6B7280",
        "border":       "#D1D5DB",
        "header_bg":    "#0B54A4",
        "success":      "#16A34A",
        "warn":         "#D97706",
        "log_bg":       "#F8FAFC",
        "log_text":     "#1E3A5F",
        "sidebar":      "#0B2B5C",
        "sidebar_hover":"#0E3870",
    }
    FONT_TITLE   = ("Segoe UI", 16, "bold")
    FONT_HEADING = ("Segoe UI", 11, "bold")
    FONT_BODY    = ("Segoe UI",  9)
    FONT_MONO    = ("Consolas",  9)
    FONT_SMALL   = ("Segoe UI",  8)
    ProgressPopup = None
    def _lighten(h):
        h = h.lstrip("#")
        r, g, b = [int(h[i:i+2], 16) for i in (0, 2, 4)]
        return f"#{min(255,r+30):02x}{min(255,g+30):02x}{min(255,b+30):02x}"

_ICON_LIBRARY_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "Icon Library")
)
_DEFAULT_CONFIG = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "configs", "default_config.ini")
)

_LABEL_FILETYPES = [
    ("Images & PDFs", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.pdf"),
    ("All files", "*.*"),
]


def _browse_dir(var, title, parent):
    d = filedialog.askdirectory(title=title, parent=parent)
    if d:
        var.set(d)


def _browse_file_single(var, title, filetypes, parent):
    p = filedialog.askopenfilename(title=title, filetypes=filetypes, parent=parent)
    if p:
        var.set(p)


def _open_image_from_path(path: str) -> Image.Image:
    """Open any supported image file, including PDFs (renders page 1 via PyMuPDF).

    Returns a PIL Image, or raises an exception on failure.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc  = fitz.open(path)
            page = doc[0]
            mat  = fitz.Matrix(1.5, 1.5)      # 1.5× zoom ≈ 108 dpi
            pix  = page.get_pixmap(matrix=mat, alpha=False)
            doc.close()
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        except ImportError:
            raise RuntimeError(
                "PDF preview requires PyMuPDF (pip install pymupdf). "
                "Install it or select an image file instead."
            )
    return Image.open(path)


class MainGUI:
    """Dark-themed, responsive Label Verification window."""

    HEADER_BG = "#0B54A4"
    ACCENT    = "#0B54A4"

    def __init__(self, root: tk.Misc, mode: str = "label"):
        """
        mode: "label" — only Label Verification tab (no IFU)
              "ifu"   — only IFU / Addendum Verification tab (no label)
        """
        self.root  = root
        self._mode = mode

        _titles = {
            "label": "Label Verification — EVO MDD",
            "ifu":   "IFU / Addendum Verification — EVO MDD",
        }
        self.root.title(_titles.get(mode, "EVO MDD Verification Tool"))
        self.root.configure(bg=COLORS["bg"])
        self.root.resizable(True, True)

        # ── Internal state ────────────────────────────────────────────────
        # List of selected label file paths (supports multi-select)
        self._label_paths: list[str] = []

        # V&V Plan state — widgets assigned in _build_left_panel
        self._vv_plan_path       = tk.StringVar()
        self._vv_plan_data       = None
        self._vv_plan_banner     = None
        self._vv_plan_banner_var = None
        self._ifu_btn            = None  # kept for compat; not shown in sidebar

        # IFU inline verification state (all lives inside the IFU tab)
        self._ifu_shared_var   = tk.StringVar()
        self._ifu_cards: list  = []
        self._ifu_summary_var  = tk.StringVar(value="")
        self._ifu_complete_var = tk.StringVar(value="")   # completion banner

        # StringVar bound to the entry widget — shows path or "N files selected"
        self._label_path_display = tk.StringVar()

        self._icon_dir    = tk.StringVar(value=_ICON_LIBRARY_DIR)
        self._config_path = tk.StringVar(
            value=_DEFAULT_CONFIG if os.path.exists(_DEFAULT_CONFIG) else ""
        )
        self._status_var  = tk.StringVar(value="Ready.")
        self._results     = []
        self._photo_cache = []
        self._running     = False

        # Icon checklist state
        self._icon_check_vars: list  = []  # list of tk.BooleanVar
        self._icon_list_paths: list  = []  # matching list of absolute paths
        self._icon_list_names: list  = []  # display names (no extension)

        # Stubs for label-specific sidebar widgets (created only in label mode;
        # set to None here so IFU mode methods can guard against them)
        self._file_count_var        = tk.StringVar()
        self._file_count_lbl        = None
        self._icon_list_canvas      = None
        self._icon_list_inner       = None
        self._icon_list_win         = None
        self._icon_list_placeholder = None
        self._sel_rec_btn           = None
        self._icon_hint_lbl         = None
        # Label sub-notebook tabs (created only in label mode)
        self._label_sub_nb          = None
        self._req_tab               = None
        self._grid_tab              = None
        self._label_tab             = None

        # Hint banner below the icon checklist (updated when V&V Plan is loaded)
        self._icon_hint_var = tk.StringVar(
            value="Upload a V&V Plan (.docx) to see which icons are required."
        )

        # Batch-run progress (determinate)
        self._pb_var          = tk.DoubleVar(value=0.0)
        self._batch_lbl_var   = tk.StringVar(value="")

        # Cumulative totals across a batch run
        self._batch_total   = 0
        self._batch_found   = 0
        self._batch_missing = 0

        # Symbols from V&V Plan — list of {"symbol", "gtin_count", "gtins"}
        self._required_symbols: list = []

        self._build_ui()
        self._center()
        self._wire_automations()

    # ── Event bus wiring ──────────────────────────────────────────────────

    def _wire_automations(self):
        register_callback("icon_library_ready", self._on_icon_library_ready)
        register_callback("symbols_ready",      self._on_symbols_ready)
        register_callback("vv_plan_ready",      self._on_vv_plan_ready)

        if self._mode == "label":
            # Auto-refresh the icon checklist whenever the folder path changes
            self._icon_dir.trace_add("write", self._populate_icon_checklist)
            self._populate_icon_checklist()

        if APP_STATE:
            if APP_STATE.get("icon_library_path") and self._mode == "label":
                self._apply_icon_library(APP_STATE["icon_library_path"])
            if APP_STATE.get("verify_list") and self._mode == "label":
                self._apply_symbols_ready(APP_STATE["verify_list"])
            # Auto-load V&V plan shared from home screen
            if APP_STATE.get("vv_plan_data"):
                self.root.after(
                    100, self._apply_vv_plan_from_state,
                    APP_STATE["vv_plan_path"], APP_STATE["vv_plan_data"])

    def _on_vv_plan_ready(self, data):
        """Called via event bus when the home screen loads a new V&V Plan."""
        path = APP_STATE.get("vv_plan_path", "") if APP_STATE else ""
        self.root.after(0, self._apply_vv_plan_from_state, path, data)

    def _on_icon_library_ready(self, path: str):
        self.root.after(0, self._apply_icon_library, path)

    def _apply_icon_library(self, path: str):
        self._icon_dir.set(path)
        self._log(f"[Auto] Symbol Reference Library updated → {path}", "OK")

    def _on_symbols_ready(self, summary: list):
        self.root.after(0, self._apply_symbols_ready, summary)

    def _apply_symbols_ready(self, summary: list):
        self._required_symbols = [s for s in summary if s.get("gtin_count", 0) > 0]
        needed_names = [s["symbol"] for s in self._required_symbols]

        # Count total PRD references across all symbols
        total_prds = sum(s.get("gtin_count", 0) for s in self._required_symbols)

        if self._mode != "label":
            return  # label sidebar widgets don't exist in IFU mode

        if needed_names:
            self._log(
                f"[V&V Plan] {total_prds} PRD reference(s) · "
                f"{len(needed_names)} symbol type(s): {', '.join(needed_names)}",
                "OK",
            )
            names_str = ", ".join(needed_names)
            self._icon_hint_var.set(
                f"★  {len(needed_names)} symbol(s) required by V&V Plan:\n"
                f"{names_str}\n"
                f"Press '★ Required Symbols' to select only these icons."
            )
            self._icon_hint_lbl.configure(
                fg=COLORS["accent3"],
                bg=COLORS["bg_card"],
                highlightbackground=COLORS["accent3"],
            )
            if self._sel_rec_btn:
                self._sel_rec_btn.configure(state="normal")
        else:
            self._icon_hint_var.set(
                "No symbols required.  Upload a V&V Plan to populate required symbols."
            )
            self._icon_hint_lbl.configure(
                fg=COLORS["text_muted"],
                bg=COLORS["bg_card"],
                highlightbackground=COLORS["border"],
            )
            if self._sel_rec_btn:
                self._sel_rec_btn.configure(state="disabled")

        # Rebuild checklist so required icons get their badges
        self._populate_icon_checklist()
        self._populate_required_symbols_tab()
        self._nb.select(self._label_verification_tab)
        self._label_sub_nb.select(self._req_tab)

    # ── Layout ────────────────────────────────────────────────────────────

    def _center(self):
        self.root.update_idletasks()
        w, h = 1280, 760
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x  = max(0, (sw - w) // 2)
        y  = max(0, (sh - h) // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.minsize(900, 580)

    def _build_ui(self):
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self._build_header()
        self._build_main()
        self._build_statusbar()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=self.HEADER_BG, padx=20, pady=14)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.columnconfigure(1, weight=1)

        tk.Label(hdr, text="EVO MDD Verification Tool",
                 bg=self.HEADER_BG, fg="white",
                 font=FONT_TITLE).grid(row=0, column=0, sticky="w")
        tk.Label(hdr, text="  ·  Automated medical symbol verification",
                 bg=self.HEADER_BG, fg="#93C5FD",
                 font=FONT_BODY).grid(row=0, column=1, sticky="w")
        tk.Label(hdr, text="EVO MDD Verification Tool",
                 bg=self.HEADER_BG, fg="#64748B",
                 font=FONT_SMALL).grid(row=0, column=2, sticky="e")

    def _build_main(self):
        pane = tk.PanedWindow(
            self.root, orient="horizontal",
            bg=COLORS["border"], sashwidth=5, sashrelief="flat",
            handlesize=0,
        )
        pane.grid(row=1, column=0, sticky="nsew")

        left  = self._build_left_panel(pane)
        right = self._build_right_panel(pane)

        pane.add(left,  minsize=300, width=340, stretch="never")
        pane.add(right, minsize=460,             stretch="always")

    # ── Left panel ────────────────────────────────────────────────────────

    def _build_left_panel(self, parent):
        frame = tk.Frame(parent, bg=COLORS["bg"], padx=12, pady=12)
        frame.rowconfigure(99, weight=1)
        frame.columnconfigure(0, weight=1)

        row = 0

        # ── Label-only sidebar sections ────────────────────────────────
        # Skipped entirely for IFU mode; stubs are pre-set in __init__.
        if self._mode != "label":
            return frame   # sidebar done for IFU mode

        row = self._section_header(frame, "Label Verification", row)

        tk.Label(frame, text="Upload Label (PDF or Image)",
                 bg=COLORS["bg"], fg=COLORS["text_muted"],
                 font=FONT_SMALL, anchor="w").grid(
            row=row, column=0, columnspan=3, sticky="w")
        row += 1

        # Read-only entry showing path / "N files selected"
        tk.Entry(
            frame, textvariable=self._label_path_display, state="readonly",
            bg=COLORS["bg_input"], fg=COLORS["text"],
            readonlybackground=COLORS["bg_input"],
            relief="flat", font=FONT_MONO,
            highlightthickness=1, highlightbackground=COLORS["border"],
        ).grid(row=row, column=0, sticky="ew", padx=(0, 4))
        frame.columnconfigure(0, weight=1)

        tk.Button(
            frame, text="Browse…",
            command=self._browse_labels,
            bg=COLORS["bg_card"], fg=COLORS["text"],
            activebackground=COLORS["border"], activeforeground=COLORS["text"],
            relief="flat", cursor="hand2",
            font=FONT_SMALL, padx=8, pady=4, bd=0,
            highlightthickness=1, highlightbackground=COLORS["border"],
        ).grid(row=row, column=1, sticky="e", pady=(0, 4))
        row += 1

        # File-count chip — shows how many files are queued
        self._file_count_var = tk.StringVar(value="")
        self._file_count_lbl = tk.Label(
            frame, textvariable=self._file_count_var,
            bg=COLORS["bg"], fg=COLORS["text_muted"],
            font=FONT_SMALL, anchor="w",
        )
        self._file_count_lbl.grid(row=row, column=0, columnspan=3,
                                  sticky="w", pady=(0, 8))
        row += 1

        # ── Reference Library ─────────────────────────────────────────
        row = self._section_header(frame, "Reference Library", row, top=4)
        row = self._path_row(frame, "Symbol Reference Library", self._icon_dir,
                             "Select Symbol Reference Library Folder",
                             [], is_dir=True, row=row)

        # ── Icons to Verify ───────────────────────────────────────────
        row = self._section_header(frame, "Icons to Verify", row, top=4)

        # Hint banner — tells the user which symbols are required
        self._icon_hint_lbl = tk.Label(
            frame, textvariable=self._icon_hint_var,
            bg=COLORS["bg_card"],
            fg=COLORS["text_muted"],
            font=FONT_SMALL, anchor="w",
            wraplength=260, justify="left",
            padx=8, pady=6,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        self._icon_hint_lbl.grid(row=row, column=0, columnspan=3,
                                  sticky="ew", pady=(0, 6))
        row += 1

        # Select All / Deselect All / Select Recommended buttons
        btn_row = tk.Frame(frame, bg=COLORS["bg"])
        btn_row.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)
        btn_row.columnconfigure(2, weight=1)

        def _select_all():
            for v in self._icon_check_vars:
                v.set(True)

        def _deselect_all():
            for v in self._icon_check_vars:
                v.set(False)

        def _select_recommended():
            req = self._get_required_name_set()
            if not req:
                return
            for name, v in zip(self._icon_list_names, self._icon_check_vars):
                v.set(name.lower() in req)

        tk.Button(
            btn_row, text="All", command=_select_all,
            bg=COLORS["bg_card"], fg=COLORS["text"],
            activebackground=COLORS["border"], activeforeground=COLORS["text"],
            relief="flat", cursor="hand2",
            font=FONT_SMALL, padx=4, pady=3, bd=0,
            highlightthickness=1, highlightbackground=COLORS["border"],
        ).grid(row=0, column=0, sticky="ew", padx=(0, 2))
        tk.Button(
            btn_row, text="None", command=_deselect_all,
            bg=COLORS["bg_card"], fg=COLORS["text"],
            activebackground=COLORS["border"], activeforeground=COLORS["text"],
            relief="flat", cursor="hand2",
            font=FONT_SMALL, padx=4, pady=3, bd=0,
            highlightthickness=1, highlightbackground=COLORS["border"],
        ).grid(row=0, column=1, sticky="ew", padx=2)

        self._sel_rec_btn = tk.Button(
            btn_row, text="★ Required Symbols", command=_select_recommended,
            bg=COLORS["accent3"], fg="#0F172A",
            activebackground=COLORS["accent3"], activeforeground="#0F172A",
            relief="flat", cursor="hand2",
            font=("Segoe UI", 8, "bold"), padx=4, pady=3, bd=0,
            highlightthickness=1, highlightbackground=COLORS["accent3"],
            state="disabled",
        )
        self._sel_rec_btn.grid(row=0, column=2, sticky="ew", padx=(2, 0))
        row += 1

        # Scrollable checklist
        list_outer = tk.Frame(frame, bg=COLORS["border"],
                              highlightthickness=1,
                              highlightbackground=COLORS["border"])
        list_outer.grid(row=row, column=0, columnspan=3, sticky="nsew",
                        pady=(0, 6))
        list_outer.rowconfigure(0, weight=1)
        list_outer.columnconfigure(0, weight=1)
        frame.rowconfigure(row, weight=1)
        row += 1

        self._icon_list_canvas = tk.Canvas(
            list_outer, bg=COLORS["bg_input"], highlightthickness=0,
            width=200, height=130,
        )
        icon_vsb = tk.Scrollbar(list_outer, orient="vertical",
                                command=self._icon_list_canvas.yview,
                                bg=COLORS["bg_card"], troughcolor=COLORS["bg"])
        self._icon_list_canvas.configure(yscrollcommand=icon_vsb.set)
        self._icon_list_canvas.grid(row=0, column=0, sticky="nsew")
        icon_vsb.grid(row=0, column=1, sticky="ns")

        self._icon_list_inner = tk.Frame(self._icon_list_canvas,
                                         bg=COLORS["bg_input"])
        self._icon_list_win = self._icon_list_canvas.create_window(
            (0, 0), window=self._icon_list_inner, anchor="nw"
        )
        self._icon_list_inner.bind(
            "<Configure>",
            lambda e: self._icon_list_canvas.configure(
                scrollregion=self._icon_list_canvas.bbox("all"))
        )
        self._icon_list_canvas.bind(
            "<Configure>",
            lambda e: self._icon_list_canvas.itemconfig(
                self._icon_list_win, width=e.width)
        )
        self._icon_list_canvas.bind_all(
            "<MouseWheel>",
            lambda e: self._icon_list_canvas.yview_scroll(
                -1 * (e.delta // 120), "units")
        )

        # Placeholder text shown before the folder is populated
        self._icon_list_placeholder = tk.Label(
            self._icon_list_inner,
            text="Browse an Icon Library Folder\nto see icons here.",
            bg=COLORS["bg_input"], fg=COLORS["text_muted"],
            font=FONT_SMALL, justify="center",
        )
        self._icon_list_placeholder.pack(pady=16)

        # ── Configuration ─────────────────────────────────────────────
        row = self._section_header(frame, "Configuration", row, top=4)
        row = self._path_row(frame, "Verification Configuration", self._config_path,
                             "Select Config File",
                             [("INI files", "*.ini *.cfg"), ("All files", "*.*")],
                             is_dir=False, row=row)

        # ── Run button ────────────────────────────────────────────────
        self._verify_btn = tk.Button(
            frame, text="▶  Start Verification",
            command=self._run,
            bg=self.ACCENT, fg="white",
            activebackground=_lighten(self.ACCENT),
            activeforeground="white",
            relief="flat", cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            padx=0, pady=10, bd=0,
        )
        self._verify_btn.grid(row=row, column=0, columnspan=3,
                              sticky="ew", pady=(14, 2))
        row += 1

        # ── Batch progress label ("Label 2 / 5 — filename.pdf") ───────
        self._batch_lbl_widget = tk.Label(
            frame, textvariable=self._batch_lbl_var,
            bg=COLORS["bg"], fg=COLORS["accent3"],
            font=FONT_SMALL, anchor="w",
        )
        self._batch_lbl_widget.grid(row=row, column=0, columnspan=3,
                                    sticky="w", pady=(2, 0))
        row += 1

        # ── Determinate progress bar ──────────────────────────────────
        style = ttk.Style()
        style.configure("Batch.Horizontal.TProgressbar",
                        troughcolor=COLORS["bg_card"],
                        background=COLORS["accent"],
                        bordercolor=COLORS["border"],
                        lightcolor=COLORS["accent"],
                        darkcolor=COLORS["accent"])

        self._mini_pb = ttk.Progressbar(
            frame, mode="determinate",
            variable=self._pb_var, maximum=100.0,
            style="Batch.Horizontal.TProgressbar",
        )
        self._mini_pb.grid(row=row, column=0, columnspan=3,
                           sticky="ew", pady=(2, 10))
        row += 1

        # ── Log ───────────────────────────────────────────────────────
        tk.Label(frame, text="Log", bg=COLORS["bg"],
                 fg=COLORS["text_muted"], font=FONT_SMALL,
                 anchor="w").grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        log_frame = tk.Frame(frame, bg=COLORS["log_bg"],
                             highlightthickness=1,
                             highlightbackground=COLORS["border"])
        log_frame.grid(row=row, column=0, columnspan=3, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        frame.rowconfigure(row, weight=1)
        row += 1

        self.log_text = tk.Text(
            log_frame, wrap="word", state="disabled",
            bg=COLORS["log_bg"], fg=COLORS["log_text"],
            font=FONT_MONO, relief="flat", padx=6, pady=4,
            insertbackground=COLORS["text"], selectbackground=COLORS["accent"],
        )
        log_sb = tk.Scrollbar(log_frame, command=self.log_text.yview,
                              bg=COLORS["bg_card"], troughcolor=COLORS["bg"])
        self.log_text.configure(yscrollcommand=log_sb.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_sb.grid(row=0, column=1, sticky="ns")

        self.log_text.tag_configure("INFO",  foreground=COLORS["log_text"])
        self.log_text.tag_configure("WARN",  foreground=COLORS["warn"])
        self.log_text.tag_configure("ERROR", foreground=COLORS["danger"])
        self.log_text.tag_configure("OK",    foreground=COLORS["success"])

        return frame


    def _refocus(self):
        """Bring this window back to front after a file dialog closes."""
        self.root.lift()
        self.root.focus_force()
        self.root.attributes("-topmost", True)
        self.root.after(100, lambda: self.root.attributes("-topmost", False))

    def _browse_vv_plan(self):
        """Browse for a V&V Plan .docx and parse it immediately."""
        p = filedialog.askopenfilename(
            parent=self.root,
            title="Select V&V Plan Document",
            filetypes=[("Word documents", "*.docx *.doc"),
                       ("All files", "*.*")])
        if not p:
            self._refocus(); return

        self._vv_plan_path.set(p)
        self._vv_plan_banner_var.set("Parsing V&V Plan...")
        self._vv_plan_banner.configure(fg=COLORS["accent"])
        self.root.update_idletasks()

        try:
            import sys, os as _os
            _root = _os.path.normpath(
                _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
            if _root not in sys.path:
                sys.path.insert(0, _root)

            from vv_plan_parser import parse_vv_plan
            data = parse_vv_plan(p)

            if data.raw_error:
                self._vv_plan_banner_var.set(f"Error: {data.raw_error}")
                self._vv_plan_banner.configure(fg=COLORS["danger"])
                self._refocus(); return

            self._vv_plan_data = data
            n_lbl = len(data.label_prds)
            n_ifu = len(data.ifu_prds)
            n_tot = len(data.prd_entries)
            summary = (
                f"Loaded: {_os.path.basename(p)}\n"
                f"{data.er_number}  |  {n_tot} PRD(s)  |  Sample: {data.sample_size}\n"
                f"Label PRDs: {n_lbl}   IFU PRDs: {n_ifu}"
            )
            self._vv_plan_banner_var.set(summary)
            self._vv_plan_banner.configure(
                fg=COLORS["success"],
                highlightbackground=COLORS["success"])

            self._populate_vv_plan_tab()
            if self._mode in ("ifu", "full"):
                self._populate_ifu_tab()

            # Auto-populate required symbols for label verification
            if data.label_prds and self._mode in ("label", "full"):
                sym_summary = []
                sym_seen = {}
                for entry in data.label_prds:
                    for sym in entry.symbols:
                        if sym not in sym_seen:
                            sym_seen[sym] = {"symbol": sym, "gtin_count": 0, "gtins": []}
                            sym_summary.append(sym_seen[sym])
                        sym_seen[sym]["gtin_count"] += 1
                        sym_seen[sym]["gtins"].append(entry.prd_id)
                if sym_summary:
                    self._apply_symbols_ready(sym_summary)
                    self._log(
                        f"[V&V Plan] {n_lbl} label PRD(s) — required symbols auto-selected",
                        "OK")

            self._log(
                f"[V&V Plan] {data.doc_title or _os.path.basename(p)}\n"
                f"  ER: {data.er_number}  Req doc: {data.req_doc}  Sample: {data.sample_size}\n"
                f"  {n_lbl} label PRD(s), {n_ifu} IFU PRD(s)",
                "OK")

        except Exception as e:
            import traceback
            self._vv_plan_banner_var.set(f"Parse error: {e}")
            self._vv_plan_banner.configure(fg=COLORS["danger"])
            self._log(f"V&V Plan parse error: {e}\n{traceback.format_exc()}", "ERROR")

        self._refocus()

    def _apply_vv_plan_from_state(self, path: str, data):
        """Apply a V&V Plan already parsed and stored in APP_STATE."""
        if data is None:
            return
        self._vv_plan_data = data
        self._vv_plan_path.set(path)
        n_lbl = len(getattr(data, "label_prds", []))
        n_ifu = len(getattr(data, "ifu_prds",   []))
        import os as _os
        summary = (
            f"Auto-loaded: {_os.path.basename(path)}\n"
            f"Label PRDs: {n_lbl}   IFU PRDs: {n_ifu}"
        )
        if self._vv_plan_banner_var:
            self._vv_plan_banner_var.set(summary)
        if self._vv_plan_banner:
            self._vv_plan_banner.configure(
                fg=COLORS["success"],
                highlightbackground=COLORS["success"])
        self._populate_vv_plan_tab()
        if self._mode in ("ifu", "full"):
            self._populate_ifu_tab()
        if getattr(data, "label_prds", None) and self._mode in ("label", "full"):
            sym_summary = []
            sym_seen = {}
            for entry in data.label_prds:
                for sym in getattr(entry, "symbols", []):
                    if sym not in sym_seen:
                        sym_seen[sym] = {"symbol": sym, "gtin_count": 0, "gtins": []}
                        sym_summary.append(sym_seen[sym])
                    sym_seen[sym]["gtin_count"] += 1
            if sym_summary:
                self._apply_symbols_ready(sym_summary)
        self._log(f"[Auto] V&V Plan loaded from shared state: {_os.path.basename(path)}", "OK")

    # ── IFU inline verification helpers ──────────────────────────────────

    def _ifu_browse_shared(self):
        p = filedialog.askopenfilename(
            parent=self.root,
            title="Select IFU Document (PDF)",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if p:
            self._ifu_shared_var.set(p)
            self._log(f"IFU document set: {os.path.basename(p)}", "INFO")
        self._refocus()

    def _ifu_run_all(self):
        shared = self._ifu_shared_var.get()
        if not self._ifu_cards:
            return
        any_run = False
        for cd in self._ifu_cards:
            path = shared
            if path and os.path.isfile(path):
                any_run = True
                self._ifu_verify_single(cd)
        if not any_run:
            messagebox.showwarning(
                "No IFU Document",
                "Please upload an IFU document using the Browse button above.",
                parent=self.root)

    def _ifu_set_progress(self, card: dict, pct: int, label: str):
        """Set the card progress bar value and label text (main-thread safe)."""
        card["prog_bar"]["value"] = min(max(pct, 0), 100)
        card["prog_lbl_var"].set(label)

    def _ifu_verify_single(self, card: dict):
        import sys, os as _os, threading as _th
        _root_dir = _os.path.normpath(
            _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
        if _root_dir not in sys.path:
            sys.path.insert(0, _root_dir)
        from ifu_verifier_window import (extract_pdf_text, detect_sections,
                                         search_ifu_for_keywords,
                                         compute_verdict_suggestion)
        ifu_path = self._ifu_shared_var.get()
        if not ifu_path or not _os.path.isfile(ifu_path):
            return

        prd = card["prd"]
        self._log(f"Verifying {prd.prd_id} — {_os.path.basename(ifu_path)}…")

        rt = card["results_text"]
        rt.configure(state="normal"); rt.delete("1.0", "end")
        rt.insert("end", "Extracting text from IFU document…\n")
        rt.configure(state="disabled")
        card["score_var"].set("Working…")
        self.root.after(0, self._ifu_set_progress, card, 0, "Reading PDF…")
        self.root.update_idletasks()

        # ── Phase 1: extraction progress (0 → 50%) ────────────────────
        def _extract_cb(cur, total):
            pct = int(cur / total * 45) if total > 0 else 0   # 0-45%
            self.root.after(0, self._ifu_set_progress, card, pct,
                            f"📄  Extracting page {cur} of {total}…")

        # ── Phase 2: search progress (50 → 100%) ─────────────────────
        def _search_cb(cur, total, found, n_kw):
            remaining = n_kw - found
            pct = 50 + int(cur / total * 50) if total > 0 else 50
            status = (f"🔍  Searching page {cur}/{total}  ·  "
                      f"{found}/{n_kw} keywords found")
            if remaining > 0:
                status += f"  ·  {remaining} still pending…"
            else:
                status += "  ·  All found — stopping early!"
            self.root.after(0, self._ifu_set_progress, card, pct, status)

        def _run():
            try:
                # Phase 1 — extract
                self.root.after(0, self._ifu_set_progress, card, 2,
                                "📄  Opening PDF…")
                pages = extract_pdf_text(ifu_path, page_callback=_extract_cb)
                if not pages:
                    self.root.after(0, self._ifu_show_error, card,
                                    f"Could not extract text from "
                                    f"{_os.path.basename(ifu_path)}")
                    return

                # Phase 1.5 — section detection
                self.root.after(0, self._ifu_set_progress, card, 48,
                                "🔎  Detecting document sections…")
                secs = detect_sections(pages)

                # Phase 2 — keyword search
                self.root.after(0, self._ifu_set_progress, card, 50,
                                f"🔍  Searching {len(pages)} pages…")
                hits = search_ifu_for_keywords(
                    pages, card["keywords"], progress_callback=_search_cb)

                v, sc, reason = compute_verdict_suggestion(
                    hits, secs, card["req_sections"], 0.75)
                self.root.after(0, self._ifu_show_result,
                                card, hits, secs, v, sc, reason)
            except Exception as e:
                import traceback
                self.root.after(0, self._log,
                                f"IFU error: {e}\n{traceback.format_exc()}", "ERROR")

        _th.Thread(target=_run, daemon=True).start()

    def _ifu_show_error(self, card: dict, msg: str):
        rt = card["results_text"]
        rt.configure(state="normal"); rt.delete("1.0", "end")
        rt.insert("end", f"Error: {msg}\n", "missing")
        rt.configure(state="disabled")
        card["score_var"].set(f"Error: {msg}")
        self._log(f"{card['prd'].prd_id}: {msg}", "ERROR")

    def _ifu_show_result(self, card: dict, hits: list, sections: dict,
                         verdict: str, score: float, reason: str):
        # ── Store hits in card for the PDF report ──────────────────
        card["hits"] = hits

        # ── Finalise progress bar ───────────────────────────────────
        found_count = sum(1 for h in hits if h["found"])
        n_kw        = len(hits)
        stopped_pg  = ""
        if found_count == n_kw and n_kw > 0:
            # all found — find which was the last page scanned
            max_pg = max((h["page"] for h in hits if h.get("page")), default="?")
            stopped_pg = f"  ·  Stopped at page {max_pg} (all found)"
        card["prog_bar"]["value"] = 100
        card["prog_lbl_var"].set(
            f"Done  ·  {found_count}/{n_kw} keywords found{stopped_pg}")

        # ── Results text ────────────────────────────────────────────
        rt = card["results_text"]
        rt.configure(state="normal"); rt.delete("1.0", "end")

        if sections:
            rt.insert("end", "Sections detected: ", "section")
            rt.insert("end", ", ".join(sections.keys()) + "\n", "section")

        _mt_badge = {
            "exact":   "",
            "stem":    " [stem]",
            "section": " [section heading]",
        }
        for h in hits:
            if h["found"]:
                badge = _mt_badge.get(h.get("match_type", "exact"), "")
                rt.insert("end", f"  ✔  '{h['keyword']}'{badge}", "found")
                sec = f" [{h['section']}]" if h.get("section") else ""
                rt.insert("end", f"  p.{h['page']}{sec}: ", "page")
                rt.insert("end", f"{h['snippet'][:90]}\n")
            else:
                rt.insert("end", f"  ✗  '{h['keyword']}' — not found in IFU\n", "missing")
        rt.configure(state="disabled")

        pct = int(score * 100)
        card["score_var"].set(
            f"Match: {found_count}/{n_kw} keywords ({pct}%)  ·  {reason}")

        vl = card["verdict_lbl"]
        vv = card["verdict_var"]
        if verdict == "PASS":
            vv.set("PASS");   vl.configure(text="✔  PASS",   fg=COLORS["success"])
        elif verdict == "FAIL":
            vv.set("FAIL");   vl.configure(text="✗  FAIL",   fg=COLORS["danger"])
        else:
            vv.set("REVIEW"); vl.configure(text="⚠  REVIEW", fg=COLORS["warn"])

        self._log(
            f"{card['prd'].prd_id}: {found_count}/{n_kw} ({pct}%) → {verdict}",
            "OK" if verdict == "PASS" else ("ERROR" if verdict == "FAIL" else "WARN"))
        self._ifu_update_summary()


    def _ifu_update_summary(self):
        total   = len(self._ifu_cards)
        passed  = sum(1 for c in self._ifu_cards if c["verdict_var"].get() == "PASS")
        failed  = sum(1 for c in self._ifu_cards if c["verdict_var"].get() == "FAIL")
        review  = sum(1 for c in self._ifu_cards if c["verdict_var"].get() == "REVIEW")
        pending = total - passed - failed - review
        self._ifu_summary_var.set(
            f"{total} IFU PRD(s)  ·  "
            f"✔ {passed} Pass  ✗ {failed} Fail  ⚠ {review} Review  ○ {pending} Pending"
        )
        # Completion banner
        if hasattr(self, "_ifu_complete_var"):
            if pending == 0 and total > 0:
                if failed == 0 and review == 0:
                    msg = f"✅  All {total} IFU requirement(s) verified — {passed} Pass"
                    fg  = COLORS["success"]
                elif failed > 0:
                    msg = (f"⚠  Verification complete — {passed} Pass · "
                           f"{failed} Fail · {review} Review")
                    fg  = COLORS["danger"]
                else:
                    msg = (f"⚠  Verification complete — {passed} Pass · "
                           f"{review} require review")
                    fg  = COLORS["warn"]
                self._ifu_complete_var.set(msg)
                if hasattr(self, "_ifu_complete_lbl"):
                    self._ifu_complete_lbl.configure(fg=fg)
            else:
                self._ifu_complete_var.set("")

    def _ifu_export(self):
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save IFU Verification Report",
            initialfile=f"IFU_Verification_{ts}.pdf",
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf"), ("All", "*.*")])
        if not path:
            self._refocus(); return

        import sys, os as _os
        _root_dir = _os.path.normpath(
            _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
        if _root_dir not in sys.path:
            sys.path.insert(0, _root_dir)
        from ifu_verifier_window import build_pdf_report
        ifu_path = self._ifu_shared_var.get()
        result = build_pdf_report(self._ifu_cards, self._vv_plan_data, path,
                                  ifu_pdf_path=ifu_path)
        if result == path:
            self._log(f"IFU report saved: {path}", "INFO")
            messagebox.showinfo("Report Saved",
                                f"IFU Verification Report saved:\n{path}",
                                parent=self.root)
        else:
            messagebox.showerror("Export Error", result, parent=self.root)
        self._refocus()

    def _browse_labels(self):
        """Open a multi-select file dialog for label images / PDFs."""
        paths = filedialog.askopenfilenames(
            title="Select Label Image(s) or PDF(s)",
            filetypes=_LABEL_FILETYPES,
            parent=self.root,
        )
        if not paths:
            return
        self._label_paths = list(paths)
        n = len(paths)
        if n == 1:
            self._label_path_display.set(paths[0])
            self._file_count_var.set("")
        else:
            self._label_path_display.set(f"{n} files selected")
            names = ", ".join(os.path.basename(p) for p in paths[:3])
            suffix = f"  … +{n - 3} more" if n > 3 else ""
            self._file_count_var.set(f"  {names}{suffix}")
        self._refocus()
    def _get_required_name_set(self) -> set:
        """Return a lower-case set of required symbol names from the V&V Plan."""
        return {s["symbol"].lower() for s in self._required_symbols
                if s.get("symbol")}

    def _populate_icon_checklist(self, *_args):
        """Scan the current icon library folder and rebuild the checklist.

        Icons that match a required symbol (from the V&V Plan) are
        highlighted with an amber '★ Required' badge and sorted to the top.
        Non-required icons appear dimmed below.
        """
        if self._icon_list_canvas is None:  # IFU-mode: no label sidebar
            return
        import glob as _glob

        for w in self._icon_list_inner.winfo_children():
            w.destroy()
        self._icon_check_vars.clear()
        self._icon_list_paths.clear()
        self._icon_list_names.clear()

        icon_dir = self._icon_dir.get().strip()
        if not icon_dir or not os.path.isdir(icon_dir):
            tk.Label(
                self._icon_list_inner,
                text="Browse an Icon Library Folder\nto see icons here.",
                bg=COLORS["bg_input"], fg=COLORS["text_muted"],
                font=FONT_SMALL, justify="center",
            ).pack(pady=16)
            self._icon_list_canvas.configure(
                scrollregion=self._icon_list_canvas.bbox("all"))
            return

        icon_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif")
        raw_paths = []
        for pattern in icon_exts:
            raw_paths.extend(_glob.glob(os.path.join(icon_dir, pattern)))
        raw_paths.sort()

        if not raw_paths:
            tk.Label(
                self._icon_list_inner,
                text="No icon images found in this folder.",
                bg=COLORS["bg_input"], fg=COLORS["text_muted"],
                font=FONT_SMALL, justify="center",
            ).pack(pady=16)
            self._icon_list_canvas.configure(
                scrollregion=self._icon_list_canvas.bbox("all"))
            return

        req_set = self._get_required_name_set()

        # Sort: required icons first, then the rest alphabetically
        def _sort_key(p):
            n = os.path.splitext(os.path.basename(p))[0].lower()
            return (0 if n in req_set else 1, n)

        sorted_paths = sorted(raw_paths, key=_sort_key)

        # Section divider if we have both required and optional icons
        has_divider = bool(req_set) and any(
            os.path.splitext(os.path.basename(p))[0].lower() not in req_set
            for p in sorted_paths
        )
        divider_shown = False

        for path in sorted_paths:
            name = os.path.splitext(os.path.basename(path))[0]
            is_req = name.lower() in req_set

            # Show divider between required and optional groups
            if has_divider and not is_req and not divider_shown:
                divider_shown = True
                div = tk.Frame(self._icon_list_inner, bg=COLORS["border"],
                               height=1)
                div.pack(fill="x", padx=6, pady=(4, 2))
                tk.Label(
                    self._icon_list_inner,
                    text="── other icons ──",
                    bg=COLORS["bg_input"], fg=COLORS["text_muted"],
                    font=("Segoe UI", 7), justify="center",
                ).pack()

            self._icon_list_paths.append(path)
            self._icon_list_names.append(name)
            # Pre-check required icons; leave others unchecked when mapper has run
            checked = True if not req_set else is_req
            var = tk.BooleanVar(value=checked)
            self._icon_check_vars.append(var)

            # Row frame so we can place checkbox + badge side by side
            row_frame = tk.Frame(self._icon_list_inner, bg=COLORS["bg_input"])
            row_frame.pack(fill="x", padx=4, pady=1)

            cb = tk.Checkbutton(
                row_frame, text=name, variable=var,
                bg=COLORS["bg_input"],
                fg=COLORS["accent3"] if is_req else COLORS["text_muted"],
                activebackground=COLORS["bg_input"],
                activeforeground=COLORS["text"],
                selectcolor=COLORS["bg_card"],
                font=("Segoe UI", 8, "bold") if is_req else FONT_SMALL,
                anchor="w", cursor="hand2",
            )
            cb.pack(side="left", fill="x", expand=True)

            if is_req:
                tk.Label(
                    row_frame, text="★ Required",
                    bg=COLORS["accent3"], fg="#0F172A",
                    font=("Segoe UI", 7, "bold"),
                    padx=5, pady=1,
                ).pack(side="right", padx=(2, 4))

        self._icon_list_canvas.update_idletasks()
        self._icon_list_canvas.configure(
            scrollregion=self._icon_list_canvas.bbox("all"))

    def _section_header(self, parent, text, row, top=0):
        tk.Label(parent, text=text, bg=COLORS["bg"],
                 fg=COLORS["accent"], font=FONT_HEADING,
                 anchor="w").grid(row=row, column=0, columnspan=3,
                                  sticky="w", pady=(top, 2))
        sep = tk.Frame(parent, bg=COLORS["border"], height=1)
        sep.grid(row=row + 1, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        return row + 2

    def _path_row(self, parent, label, var, title, filetypes, is_dir, row):
        tk.Label(parent, text=label, bg=COLORS["bg"],
                 fg=COLORS["text_muted"], font=FONT_SMALL,
                 anchor="w").grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        tk.Entry(
            parent, textvariable=var, state="readonly",
            bg=COLORS["bg_input"], fg=COLORS["text"],
            readonlybackground=COLORS["bg_input"],
            relief="flat", font=FONT_MONO,
            highlightthickness=1, highlightbackground=COLORS["border"],
        ).grid(row=row, column=0, sticky="ew", padx=(0, 4))
        parent.columnconfigure(0, weight=1)

        if is_dir:
            cmd = lambda v=var, t=title, p=parent: _browse_dir(v, t, p)
        else:
            cmd = lambda v=var, t=title, f=filetypes, p=parent: \
                _browse_file_single(v, t, f, p)

        tk.Button(
            parent, text="Browse…", command=cmd,
            bg=COLORS["bg_card"], fg=COLORS["text"],
            activebackground=COLORS["border"], activeforeground=COLORS["text"],
            relief="flat", cursor="hand2",
            font=FONT_SMALL, padx=8, pady=4, bd=0,
            highlightthickness=1, highlightbackground=COLORS["border"],
        ).grid(row=row, column=1, sticky="e", pady=(0, 8))
        return row + 1

    # ── Right panel ───────────────────────────────────────────────────────

    def _build_right_panel(self, parent):
        frame = tk.Frame(parent, bg=COLORS["bg"])
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        self._build_summary_bar(frame)
        self._build_results_area(frame)

        return frame

    def _build_summary_bar(self, parent):
        bar = tk.Frame(parent, bg=COLORS["bg_card"],
                       highlightthickness=1,
                       highlightbackground=COLORS["border"])
        bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        bar.columnconfigure(4, weight=1)

        self._verdict_var = tk.StringVar(value="—")
        self._total_var   = tk.StringVar(value="—")
        self._found_var   = tk.StringVar(value="—")
        self._missing_var = tk.StringVar(value="—")
        self._labels_var  = tk.StringVar(value="—")

        self._verdict_lbl = tk.Label(
            bar, textvariable=self._verdict_var,
            bg=COLORS["bg_card"], fg=COLORS["text_muted"],
            font=("Segoe UI", 13, "bold"), width=10, pady=10,
        )
        self._verdict_lbl.grid(row=0, column=0, padx=(14, 20))

        for col, (lbl_text, var) in enumerate([
            ("Labels",         self._labels_var),
            ("Symbol Checks",  self._total_var),
            ("Verified ✔",     self._found_var),
            ("Not Found ✗",    self._missing_var),
        ], start=1):
            cell = tk.Frame(bar, bg=COLORS["bg_card"])
            cell.grid(row=0, column=col, padx=14, pady=8)
            tk.Label(cell, textvariable=var,
                     bg=COLORS["bg_card"], fg=COLORS["text"],
                     font=("Segoe UI", 15, "bold")).pack()
            tk.Label(cell, text=lbl_text,
                     bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                     font=FONT_SMALL).pack()

    def _build_results_area(self, parent):
        nb_frame = tk.Frame(parent, bg=COLORS["bg"])
        nb_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=8)
        nb_frame.rowconfigure(0, weight=1)
        nb_frame.columnconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TNotebook",
                         background=COLORS["bg"], borderwidth=0)
        style.configure("Dark.TNotebook.Tab",
                         background=COLORS["bg_card"],
                         foreground=COLORS["text_muted"],
                         padding=[12, 6], font=FONT_BODY)
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", COLORS["accent"])],
                  foreground=[("selected", "white")])

        self._nb = ttk.Notebook(nb_frame, style="Dark.TNotebook")
        self._nb.grid(row=0, column=0, sticky="nsew")

        # Hidden frame for stub widgets that are never displayed
        self._vv_hidden = tk.Frame(nb_frame, bg=COLORS["bg"])

        # Requirements tab — always shown first in both label and IFU modes
        self._vv_tab = self._make_vv_plan_tab(self._nb)
        self._nb.add(self._vv_tab, text="  Extracted Requirements  ")

        # Build only the tab(s) relevant to the current mode
        if self._mode in ("label", "full"):
            self._label_verification_tab = self._make_label_verification_tab(self._nb)
            self._nb.add(self._label_verification_tab,
                         text="  Label Verification  ")
        else:
            self._label_verification_tab = tk.Frame(self._vv_hidden)

        if self._mode in ("ifu", "full"):
            self._ifu_tab = self._make_ifu_tab(self._nb)
            self._nb.add(self._ifu_tab,
                         text="  IFU / Addendum Verification  ")
        else:
            self._ifu_tab = tk.Frame(self._vv_hidden)

    # ── Required Symbols tab ──────────────────────────────────────────────

    def _make_vv_plan_tab(self, parent):
        outer = tk.Frame(parent, bg=COLORS["bg"])
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)
        bn = tk.Frame(outer, bg=COLORS["bg_card"],
                      highlightthickness=1, highlightbackground=COLORS["border"])
        bn.grid(row=0, column=0, sticky="ew", padx=10, pady=(10,0))
        bni = tk.Frame(bn, bg=COLORS["bg_card"], padx=14, pady=10)
        bni.pack(fill="x")
        self._vv_tab_title_var = tk.StringVar(
            value="No V&V Plan loaded.")
        self._vv_tab_er_var    = tk.StringVar(value="")
        self._vv_tab_stats_var = tk.StringVar(value="")
        tk.Label(bni, textvariable=self._vv_tab_title_var,
                 bg=COLORS["bg_card"], fg=COLORS["accent"],
                 font=FONT_HEADING, anchor="w",
                 wraplength=900, justify="left").pack(anchor="w")
        meta = tk.Frame(bni, bg=COLORS["bg_card"])
        meta.pack(anchor="w", pady=(4,0))
        tk.Label(meta, textvariable=self._vv_tab_er_var,
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONT_SMALL).pack(side="left", padx=(0,20))
        tk.Label(meta, textvariable=self._vv_tab_stats_var,
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONT_SMALL).pack(side="left")
        sf = tk.Frame(outer, bg=COLORS["bg"])
        sf.grid(row=1, column=0, sticky="nsew", padx=10, pady=8)
        sf.rowconfigure(0, weight=1)
        sf.columnconfigure(0, weight=1)
        cv = tk.Canvas(sf, bg=COLORS["bg"], highlightthickness=0)
        vsb = tk.Scrollbar(sf, orient="vertical", command=cv.yview,
                           bg=COLORS["bg_card"], troughcolor=COLORS["bg"])
        cv.configure(yscrollcommand=vsb.set)
        cv.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self._vv_tab_inner = tk.Frame(cv, bg=COLORS["bg"])
        wid = cv.create_window((0,0), window=self._vv_tab_inner, anchor="nw")
        self._vv_tab_inner.bind("<Configure>",
            lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig(wid, width=e.width))
        tk.Label(self._vv_tab_inner,
                 text="Open V&V Plan from the home screen to see extracted requirements here.",
                 bg=COLORS["bg"], fg=COLORS["text_muted"],
                 font=FONT_BODY, justify="center").pack(pady=60)
        return outer

    def _populate_vv_plan_tab(self):
        data = self._vv_plan_data
        if not data:
            return
        self._vv_tab_title_var.set(data.doc_title)
        self._vv_tab_er_var.set("ER: " + data.er_number +
                                "   Sample: " + data.sample_size)
        self._vv_tab_stats_var.set(
            str(len(data.prd_entries)) + " PRD(s)  |  " +
            str(len(data.label_prds)) + " Label  |  " +
            str(len(data.ifu_prds)) + " IFU")
        for w in self._vv_tab_inner.winfo_children():
            w.destroy()
        def _sec(title, color):
            h = tk.Frame(self._vv_tab_inner, bg=color)
            h.pack(fill="x", padx=12, pady=(12,0))
            tk.Label(h, text=title, bg=color, fg="white",
                     font=FONT_HEADING, padx=10, pady=5).pack(anchor="w")
        def _row(vals, header=False, bg=COLORS["bg_card"]):
            widths = [200, 80, 480, 100]
            rf = tk.Frame(self._vv_tab_inner, bg=bg,
                          highlightthickness=1,
                          highlightbackground=COLORS["border"])
            rf.pack(fill="x", padx=12)
            fnt = (FONT_SMALL[0], FONT_SMALL[1], "bold") if header else FONT_SMALL
            hdr_bg = COLORS["header_bg"] if header else bg
            for v, w in zip(vals, widths):
                tk.Label(rf, text=str(v), bg=hdr_bg,
                         fg="white" if header else COLORS["text"],
                         font=fnt, anchor="w", width=w//7,
                         wraplength=w, justify="left",
                         padx=6, pady=4).pack(side="left")
        if data.label_prds:
            _sec("Label Verification PRDs (" + str(len(data.label_prds)) + ")",
                 COLORS["accent"])
            _row(["PRD ID","Method","Requirement Text","Symbols"], header=True)
            for i, e in enumerate(data.label_prds):
                _row([e.prd_id, e.vv_method,
                      e.requirement_text[:200],
                      ", ".join(e.symbols) or "-"],
                     bg=COLORS["bg_card"] if i%2==0 else COLORS["bg"])
        if data.ifu_prds:
            _sec("IFU Verification PRDs (" + str(len(data.ifu_prds)) + ")",
                 COLORS["accent3"])
            _row(["PRD ID","Method","Requirement Text","Type"], header=True)
            for i, e in enumerate(data.ifu_prds):
                _row([e.prd_id, e.vv_method,
                      e.requirement_text[:200],
                      e.verification_type],
                     bg=COLORS["bg_card"] if i%2==0 else COLORS["bg"])
        nt = tk.Frame(self._vv_tab_inner, bg=COLORS["bg_card"],
                      highlightthickness=1, highlightbackground=COLORS["border"])
        nt.pack(fill="x", padx=12, pady=(14,10))
        tk.Label(nt,
                 text=("Label PRDs: switch to the Label Verification tab and "
                       "upload a label, then run Start Verification. "
                       "IFU PRDs: switch to the IFU / Addendum Verification tab."),
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONT_SMALL, justify="left",
                 padx=12, pady=8).pack(anchor="w")
        if self._mode in ("label", "full"):
            self._nb.select(self._label_verification_tab)
            self._label_sub_nb.select(self._req_tab)
        elif self._mode == "ifu" and hasattr(self, "_ifu_tab"):
            self._nb.select(self._ifu_tab)

    # ── Label Verification combined tab ───────────────────────────────────

    def _make_label_verification_tab(self, parent):
        """
        A combined tab that wraps Required Symbols, Label Results and Label
        Preview in their own sub-notebook, styled to match IFU Verification.
        """
        outer = tk.Frame(parent, bg=COLORS["bg"])
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)

        # ── Top bar (matching IFU Verification style) ─────────────────
        top_bar = tk.Frame(outer, bg=COLORS["bg_card"],
                           highlightthickness=1,
                           highlightbackground=COLORS["border"])
        top_bar.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 0))

        inner_top = tk.Frame(top_bar, bg=COLORS["bg_card"], padx=14, pady=10)
        inner_top.pack(fill="x")

        tk.Label(inner_top, text="Label Verification",
                 bg=COLORS["bg_card"], fg=COLORS["accent"],
                 font=FONT_HEADING).pack(side="left", padx=(0, 24))

        # Sub-notebook (Required Symbols | Label Results | Label Preview)
        style = ttk.Style()
        style.configure("Sub.TNotebook",
                        background=COLORS["bg"], borderwidth=0)
        style.configure("Sub.TNotebook.Tab",
                        background=COLORS["bg"],
                        foreground=COLORS["text_muted"],
                        padding=[10, 5], font=FONT_SMALL)
        style.map("Sub.TNotebook.Tab",
                  background=[("selected", COLORS["bg_card"])],
                  foreground=[("selected", COLORS["accent"])])

        sub_nb_frame = tk.Frame(outer, bg=COLORS["bg"])
        sub_nb_frame.grid(row=1, column=0, sticky="nsew")
        sub_nb_frame.rowconfigure(0, weight=1)
        sub_nb_frame.columnconfigure(0, weight=1)

        self._label_sub_nb = ttk.Notebook(sub_nb_frame, style="Sub.TNotebook")
        self._label_sub_nb.grid(row=0, column=0, sticky="nsew")

        # Create the three child tabs
        self._req_tab   = self._make_required_symbols_tab(self._label_sub_nb)
        self._grid_tab  = self._make_grid_tab(self._label_sub_nb)
        self._label_tab = self._make_label_tab(self._label_sub_nb)

        self._label_sub_nb.add(self._req_tab,   text="  Required Symbols  ")
        self._label_sub_nb.add(self._grid_tab,  text="  Label Results  ")
        self._label_sub_nb.add(self._label_tab, text="  Label Preview  ")

        return outer

    def _make_ifu_tab(self, parent):
        outer = tk.Frame(parent, bg=COLORS["bg"])
        outer.rowconfigure(2, weight=1)
        outer.columnconfigure(0, weight=1)

        # ── Top bar: title + shared IFU file upload ────────────────────
        top_bar = tk.Frame(outer, bg=COLORS["bg_card"],
                           highlightthickness=1,
                           highlightbackground=COLORS["accent3"],
                           padx=14, pady=10)
        top_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        top_bar.columnconfigure(1, weight=1)

        tk.Label(top_bar, text="IFU Verification",
                 bg=COLORS["bg_card"], fg=COLORS["accent3"],
                 font=FONT_HEADING).grid(row=0, column=0, sticky="w",
                                         padx=(0, 16), rowspan=2)

        tk.Label(top_bar, text="IFU Document (PDF):",
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONT_SMALL).grid(row=0, column=1, sticky="w")

        file_row = tk.Frame(top_bar, bg=COLORS["bg_card"])
        file_row.grid(row=1, column=1, sticky="ew", pady=(2, 0))
        file_row.columnconfigure(0, weight=1)

        tk.Entry(file_row, textvariable=self._ifu_shared_var, state="readonly",
                 bg=COLORS["bg_input"], fg=COLORS["text"],
                 readonlybackground=COLORS["bg_input"],
                 relief="flat", font=FONT_MONO,
                 highlightthickness=1,
                 highlightbackground=COLORS["border"]).grid(
            row=0, column=0, sticky="ew", padx=(0, 6))

        tk.Button(file_row, text="Browse IFU…",
                  command=self._ifu_browse_shared,
                  bg=COLORS["bg_card"], fg=COLORS["text"],
                  activebackground=COLORS["border"],
                  relief="flat", cursor="hand2",
                  font=FONT_SMALL, padx=10, pady=4, bd=0,
                  highlightthickness=1,
                  highlightbackground=COLORS["border"]).grid(row=0, column=1)

        # Action buttons on the right side of the top bar
        btn_frame = tk.Frame(top_bar, bg=COLORS["bg_card"])
        btn_frame.grid(row=0, column=2, rowspan=2, sticky="ne", padx=(16, 0))

        self._ifu_run_btn = tk.Button(
            btn_frame, text="Run All Verifications",
            command=self._ifu_run_all,
            bg=COLORS["accent"], fg="white",
            activebackground=_lighten(COLORS["accent"]),
            relief="flat", cursor="hand2",
            font=(FONT_BODY[0], FONT_BODY[1], "bold"),
            padx=12, pady=6, bd=0, state="disabled")
        self._ifu_run_btn.pack(side="left", padx=(0, 6))

        self._ifu_export_btn = tk.Button(
            btn_frame, text="Export Report",
            command=self._ifu_export,
            bg=COLORS["accent2"], fg="white",
            activebackground=_lighten(COLORS["accent2"]),
            relief="flat", cursor="hand2",
            font=FONT_BODY, padx=12, pady=6, bd=0, state="disabled")
        self._ifu_export_btn.pack(side="left")

        # ── Summary bar ───────────────────────────────────────────────
        sum_bar = tk.Frame(outer, bg=COLORS["bg_card"],
                           highlightthickness=1,
                           highlightbackground=COLORS["border"],
                           padx=14, pady=6)
        sum_bar.grid(row=1, column=0, sticky="ew", padx=10, pady=(4, 0))
        sum_bar.columnconfigure(1, weight=1)

        tk.Label(sum_bar, textvariable=self._ifu_summary_var,
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONT_SMALL).grid(row=0, column=0, sticky="w")

        # Completion banner — hidden until all PRDs are done
        self._ifu_complete_var = tk.StringVar(value="")
        self._ifu_complete_lbl = tk.Label(
            sum_bar, textvariable=self._ifu_complete_var,
            bg=COLORS["bg_card"], font=(FONT_BODY[0], FONT_BODY[1], "bold"),
            fg=COLORS["success"])
        self._ifu_complete_lbl.grid(row=0, column=1, sticky="e", padx=(16, 0))

        # ── Scrollable PRD cards area ─────────────────────────────────
        sf = tk.Frame(outer, bg=COLORS["bg"])
        sf.grid(row=2, column=0, sticky="nsew", padx=10, pady=8)
        sf.rowconfigure(0, weight=1)
        sf.columnconfigure(0, weight=1)

        cv = tk.Canvas(sf, bg=COLORS["bg"], highlightthickness=0)
        vsb = tk.Scrollbar(sf, orient="vertical", command=cv.yview,
                           bg=COLORS["bg_card"], troughcolor=COLORS["bg"])
        cv.configure(yscrollcommand=vsb.set)
        cv.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        self._ifu_tab_inner = tk.Frame(cv, bg=COLORS["bg"])
        wid = cv.create_window((0, 0), window=self._ifu_tab_inner, anchor="nw")
        self._ifu_tab_inner.bind("<Configure>",
            lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig(wid, width=e.width))
        cv.bind_all("<MouseWheel>",
                    lambda e: cv.yview_scroll(-1*(e.delta//120), "units"))

        tk.Label(self._ifu_tab_inner,
                 text="No IFU requirements yet.\n"
                      "Upload a V&V Plan that contains IFU PRDs.",
                 bg=COLORS["bg"], fg=COLORS["text_muted"],
                 font=FONT_BODY, justify="center").pack(pady=60)

        return outer

    def _populate_ifu_tab(self):
        data = self._vv_plan_data
        if not data or not data.ifu_prds:
            return
        for w in self._ifu_tab_inner.winfo_children():
            w.destroy()
        tk.Label(self._ifu_tab_inner,
                 text=str(len(data.ifu_prds)) + " IFU requirement(s) — " +
                      data.doc_title,
                 bg=COLORS["bg"], fg=COLORS["text"],
                 font=FONT_HEADING).pack(anchor="w", padx=14, pady=(12,4))
        tk.Frame(self._ifu_tab_inner, bg=COLORS["border"],
                 height=1).pack(fill="x", padx=14, pady=(0,8))

        import sys as _sys, os as _os2
        _root_dir = _os2.path.normpath(
            _os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)), "..", ".."))
        if _root_dir not in _sys.path:
            _sys.path.insert(0, _root_dir)
        try:
            from ifu_verifier_window import extract_keywords, infer_required_sections
        except ImportError:
            extract_keywords = lambda t: []
            infer_required_sections = lambda t: []

        self._ifu_cards.clear()

        for e in data.ifu_prds:
            card_frame = tk.Frame(self._ifu_tab_inner, bg=COLORS["bg_card"],
                                  highlightthickness=1,
                                  highlightbackground=COLORS["border"])
            card_frame.pack(fill="x", padx=12, pady=5)

            # Amber accent bar
            tk.Frame(card_frame, bg=COLORS["accent3"], height=3).pack(fill="x")

            body = tk.Frame(card_frame, bg=COLORS["bg_card"], padx=14, pady=10)
            body.pack(fill="x")

            # ── Header: PRD badge + V&V method ───────────────────────
            hdr_row = tk.Frame(body, bg=COLORS["bg_card"])
            hdr_row.pack(fill="x", pady=(0, 4))
            badge_f = tk.Frame(hdr_row, bg=COLORS["accent3"])
            badge_f.pack(side="left", padx=(0, 10))
            tk.Label(badge_f, text=e.prd_id,
                     bg=COLORS["accent3"], fg="#0F172A",
                     font=("Segoe UI", 8, "bold"), padx=8, pady=3).pack()
            method = getattr(e, "vv_method", "") or "Inspection"
            tk.Label(hdr_row, text=f"IFU requirement  |  {method}",
                     bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                     font=FONT_SMALL).pack(side="left")

            # ── Requirement text ──────────────────────────────────────
            tk.Label(body, text=e.requirement_text,
                     bg=COLORS["bg_card"], fg=COLORS["text"],
                     font=FONT_SMALL, anchor="w",
                     wraplength=820, justify="left").pack(anchor="w", pady=(0, 6))

            # ── Keywords + expected sections ──────────────────────────
            kws      = extract_keywords(e.requirement_text)
            req_secs = infer_required_sections(e.requirement_text)

            if kws:
                kw_row = tk.Frame(body, bg=COLORS["bg_card"])
                kw_row.pack(anchor="w", pady=(0, 2))
                tk.Label(kw_row, text="Search keywords:",
                         bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                         font=FONT_SMALL).pack(side="left", padx=(0, 6))
                for kw in kws[:8]:
                    tk.Label(kw_row, text=kw,
                             bg=COLORS["accent"], fg="white",
                             font=FONT_SMALL, padx=5, pady=1).pack(side="left", padx=2)

            if req_secs:
                sec_row = tk.Frame(body, bg=COLORS["bg_card"])
                sec_row.pack(anchor="w", pady=(0, 6))
                tk.Label(sec_row, text="Expected sections:",
                         bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                         font=FONT_SMALL).pack(side="left", padx=(0, 6))
                for sec in req_secs:
                    tk.Label(sec_row, text=sec,
                             bg="#FEF3C7", fg=COLORS["accent3"],
                             font=FONT_SMALL, padx=5, pady=1,
                             highlightthickness=1,
                             highlightbackground=COLORS["accent3"]).pack(
                        side="left", padx=2)

            # ── Progress bar (shows while scanning pages) ─────────────
            prog_outer = tk.Frame(body, bg=COLORS["bg_card"])
            prog_outer.pack(fill="x", pady=(0, 4))
            prog_outer.columnconfigure(1, weight=1)

            prog_lbl_var = tk.StringVar(value="")
            tk.Label(prog_outer, textvariable=prog_lbl_var,
                     bg=COLORS["bg_card"], fg=COLORS["accent"],
                     font=FONT_SMALL, anchor="w").grid(
                row=0, column=0, sticky="w", padx=(0, 8))

            prog_bar = ttk.Progressbar(prog_outer, mode="determinate",
                                       maximum=100, value=0)
            prog_bar.grid(row=0, column=1, sticky="ew")

            # ── Match score label ─────────────────────────────────────
            score_var = tk.StringVar(value="Not verified yet — click Verify or Run All.")
            tk.Label(body, textvariable=score_var,
                     bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                     font=FONT_SMALL, anchor="w").pack(anchor="w", pady=(0, 2))

            # ── Results text area ─────────────────────────────────────
            results_text = tk.Text(body, height=4,
                                   bg=COLORS["bg_input"], fg=COLORS["text"],
                                   font=FONT_SMALL, relief="flat",
                                   state="disabled", wrap="word",
                                   highlightthickness=1,
                                   highlightbackground=COLORS["border"])
            results_text.pack(fill="x", pady=(0, 6))
            results_text.tag_config("found",   foreground=COLORS["success"])
            results_text.tag_config("missing", foreground=COLORS["danger"])
            results_text.tag_config("page",    foreground=COLORS["accent"])
            results_text.tag_config("section", foreground=COLORS["accent3"])

            # ── Action row: Verify button + auto-verdict display ──────
            act_row = tk.Frame(body, bg=COLORS["bg_card"])
            act_row.pack(fill="x", pady=(0, 4))

            verdict_var = tk.StringVar(value="PENDING")
            verdict_lbl = tk.Label(act_row, text="Pending",
                                   bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                                   font=("Segoe UI", 9, "bold"))

            card_dict = {
                "prd":          e,
                "results_text": results_text,
                "verdict_var":  verdict_var,
                "verdict_lbl":  verdict_lbl,
                "score_var":    score_var,
                "prog_lbl_var": prog_lbl_var,
                "prog_bar":     prog_bar,
                "hits":         [],
                "notes_text":   None,
                "keywords":     kws,
                "req_sections": req_secs,
            }

            tk.Button(act_row, text="Verify",
                      command=lambda cd=card_dict: self._ifu_verify_single(cd),
                      bg=COLORS["accent"], fg="white",
                      activebackground=_lighten(COLORS["accent"]),
                      relief="flat", cursor="hand2",
                      font=FONT_BODY, padx=12, pady=5, bd=0).pack(side="left", padx=(0, 10))

            verdict_lbl.pack(side="left", padx=(0, 8))

            # ── Notes ─────────────────────────────────────────────────
            notes_lbl_row = tk.Frame(body, bg=COLORS["bg_card"])
            notes_lbl_row.pack(anchor="w")
            tk.Label(notes_lbl_row, text="Notes:",
                     bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                     font=FONT_SMALL).pack(side="left")
            notes_text = tk.Text(body, height=2,
                                 bg=COLORS["bg_input"], fg=COLORS["text"],
                                 font=FONT_SMALL, relief="flat", wrap="word",
                                 highlightthickness=1,
                                 highlightbackground=COLORS["border"])
            notes_text.pack(fill="x", pady=(2, 0))
            card_dict["notes_text"] = notes_text

            self._ifu_cards.append(card_dict)

        # Enable Run All and Export buttons now that cards are built
        self._ifu_run_btn.configure(state="normal")
        self._ifu_export_btn.configure(state="normal")
        self._ifu_update_summary()

    def _make_required_symbols_tab(self, parent):
        outer = tk.Frame(parent, bg=COLORS["bg"])
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)

        hdr = tk.Frame(outer, bg=COLORS["bg_card"],
                       highlightthickness=1,
                       highlightbackground=COLORS["border"])
        hdr.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self._req_banner_var = tk.StringVar(
            value="No symbols loaded yet.  Upload a V&V Plan to auto-populate required symbols."
        )
        # Prominent stat strip: total PRDs big, breakdown small
        banner_inner = tk.Frame(hdr, bg=COLORS["bg_card"])
        banner_inner.pack(fill="x", padx=12, pady=8)

        # Big PRD count on the left
        self._req_gtin_count_lbl = tk.Label(
            banner_inner, text="—",
            bg=COLORS["bg_card"], fg=COLORS["accent"],
            font=("Segoe UI", 26, "bold"),
        )
        self._req_gtin_count_lbl.pack(side="left", padx=(0, 8))
        tk.Label(banner_inner, text="PRDs to Verify",
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONT_BODY).pack(side="left", padx=(0, 24))

        # Breakdown text on the right
        tk.Label(banner_inner, textvariable=self._req_banner_var,
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONT_SMALL, anchor="w",
                 wraplength=700, justify="left").pack(side="left", fill="x", expand=True)

        scroll_frame = tk.Frame(outer, bg=COLORS["bg"])
        scroll_frame.grid(row=1, column=0, sticky="nsew")
        scroll_frame.rowconfigure(0, weight=1)
        scroll_frame.columnconfigure(0, weight=1)

        self._req_canvas = tk.Canvas(scroll_frame, bg=COLORS["bg"],
                                     highlightthickness=0)
        req_vsb = tk.Scrollbar(scroll_frame, orient="vertical",
                               command=self._req_canvas.yview,
                               bg=COLORS["bg_card"], troughcolor=COLORS["bg"])
        self._req_canvas.configure(yscrollcommand=req_vsb.set)
        self._req_canvas.grid(row=0, column=0, sticky="nsew")
        req_vsb.grid(row=0, column=1, sticky="ns")

        self._req_inner = tk.Frame(self._req_canvas, bg=COLORS["bg"])
        self._req_win = self._req_canvas.create_window(
            (0, 0), window=self._req_inner, anchor="nw"
        )
        self._req_inner.bind(
            "<Configure>",
            lambda e: self._req_canvas.configure(
                scrollregion=self._req_canvas.bbox("all"))
        )
        self._req_canvas.bind(
            "<Configure>",
            lambda e: self._req_canvas.itemconfig(self._req_win, width=e.width)
        )
        self._req_canvas.bind_all(
            "<MouseWheel>",
            lambda e: self._req_canvas.yview_scroll(-1*(e.delta//120), "units")
        )

        tk.Label(
            self._req_inner,
            text=(
                "Symbols will appear here once a V&V Plan has been uploaded.\n"
                "They update automatically when the plan is parsed."
            ),
            bg=COLORS["bg"], fg=COLORS["text_muted"],
            font=FONT_BODY, justify="center",
        ).pack(pady=80)

        return outer

    def _populate_required_symbols_tab(self):
        for w in self._req_inner.winfo_children():
            w.destroy()

        if not self._required_symbols:
            tk.Label(
                self._req_inner,
                text="No symbols required.\nUpload a V&V Plan to populate required symbols.",
                bg=COLORS["bg"], fg=COLORS["text_muted"],
                font=FONT_BODY, justify="center",
            ).pack(pady=80)
            self._req_banner_var.set("No symbols loaded.  Upload a V&V Plan first.")
            self._req_gtin_count_lbl.configure(text="—", fg=COLORS["text_muted"])
            return

        total_needed = len(self._required_symbols)
        # Total PRD references across all symbol types
        total_prd_refs = sum(s.get("gtin_count", 0) for s in self._required_symbols)
        _breakdown = "  ·  ".join(
            f"{s['symbol']}: {s.get('gtin_count', 0)} PRD(s)"
            for s in self._required_symbols
        )
        # Update the big count label (shows total PRD references)
        self._req_gtin_count_lbl.configure(
            text=str(total_prd_refs),
            fg=COLORS["accent"],
        )
        self._req_banner_var.set(
            f"{total_needed} symbol type(s) required   ·   {_breakdown}"
        )

        tk.Label(self._req_inner,
                 text="Symbols Required on This Device Label",
                 bg=COLORS["bg"], fg=COLORS["accent"],
                 font=FONT_HEADING, anchor="w").pack(fill="x", padx=14, pady=(12, 4))
        tk.Frame(self._req_inner, bg=COLORS["border"], height=1).pack(
            fill="x", padx=14, pady=(0, 10))

        grid_host = tk.Frame(self._req_inner, bg=COLORS["bg"])
        grid_host.pack(fill="both", expand=True, padx=10, pady=4)

        COLS = 3
        for idx, sym_entry in enumerate(self._required_symbols):
            col = idx % COLS
            row = idx // COLS
            grid_host.columnconfigure(col, weight=1)
            self._make_required_symbol_card(grid_host, sym_entry, row, col)

        note = tk.Frame(self._req_inner, bg=COLORS["bg_card"],
                        highlightthickness=1, highlightbackground=COLORS["border"])
        note.pack(fill="x", padx=14, pady=(14, 10))
        tk.Label(
            note,
            text=(
                "These symbols were identified from the V&V Plan PRD requirements as required "
                "on your device label.  Press  ▶ Start Verification  to check whether each "
                "symbol is physically present in the label image."
            ),
            bg=COLORS["bg_card"], fg=COLORS["text_muted"],
            font=FONT_SMALL, wraplength=700, justify="left",
            padx=12, pady=8,
        ).pack(anchor="w")

    def _make_required_symbol_card(self, parent, sym_entry: dict,
                                   row: int, col: int):
        symbol    = sym_entry.get("symbol", "?")
        prd_count = sym_entry.get("gtin_count", 0)
        prd_ids   = sym_entry.get("gtins", [])

        card = tk.Frame(parent, bg=COLORS["bg_card"],
                        highlightthickness=2,
                        highlightbackground=COLORS["accent3"])
        card.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")

        badge = tk.Frame(card, bg=COLORS["accent3"])
        badge.pack(fill="x")
        tk.Label(badge, text=symbol, bg=COLORS["accent3"], fg="#0F172A",
                 font=("Segoe UI", 9, "bold"), padx=8, pady=4).pack(anchor="w")

        count_frame = tk.Frame(card, bg=COLORS["bg_card"])
        count_frame.pack(fill="x", padx=10, pady=(8, 2))
        tk.Label(count_frame, text=str(prd_count),
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=("Segoe UI", 20, "bold")).pack(side="left")
        tk.Label(count_frame, text=" PRD(s) require this symbol",
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONT_BODY).pack(side="left", padx=(2, 0))

        tk.Label(card, text="REQUIRED ON LABEL", bg=COLORS["bg_card"],
                 fg=COLORS["accent3"], font=("Segoe UI", 7, "bold"),
                 padx=10).pack(anchor="w", pady=(0, 4))

        if prd_ids:
            prd_box = tk.Frame(card, bg=COLORS["bg"],
                               highlightthickness=1,
                               highlightbackground=COLORS["border"])
            prd_box.pack(fill="x", padx=10, pady=(2, 10))
            for prd_id in prd_ids[:5]:
                tk.Label(prd_box, text=prd_id, bg=COLORS["bg"],
                         fg=COLORS["text_muted"], font=FONT_MONO,
                         anchor="w", padx=6).pack(fill="x")
            if len(prd_ids) > 5:
                tk.Label(prd_box,
                         text=f"  … +{len(prd_ids) - 5} more",
                         bg=COLORS["bg"], fg=COLORS["text_muted"],
                         font=FONT_SMALL, anchor="w", padx=6,
                         pady=2).pack(fill="x")

    # ── Symbol Matches tab ────────────────────────────────────────────────

    def _make_grid_tab(self, parent):
        outer = tk.Frame(parent, bg=COLORS["bg"])
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(outer, bg=COLORS["bg"], highlightthickness=0)
        vsb = tk.Scrollbar(outer, orient="vertical",
                           command=self._canvas.yview,
                           bg=COLORS["bg_card"], troughcolor=COLORS["bg"])
        self._canvas.configure(yscrollcommand=vsb.set)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        self._grid_inner = tk.Frame(self._canvas, bg=COLORS["bg"])
        self._canvas_win = self._canvas.create_window(
            (0, 0), window=self._grid_inner, anchor="nw"
        )

        self._grid_inner.bind("<Configure>", self._on_grid_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind_all(
            "<MouseWheel>",
            lambda e: self._canvas.yview_scroll(-1*(e.delta//120), "units")
        )

        tk.Label(self._grid_inner,
                 text="Run a verification to see symbol results here.",
                 bg=COLORS["bg"], fg=COLORS["text_muted"],
                 font=FONT_BODY).pack(pady=60)

        return outer

    # ── Label Preview tab ─────────────────────────────────────────────────

    def _make_label_tab(self, parent):
        frame = tk.Frame(parent, bg=COLORS["bg"])
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self._label_canvas = tk.Canvas(frame, bg=COLORS["bg"],
                                       highlightthickness=0)
        vsb2 = tk.Scrollbar(frame, orient="vertical",
                             command=self._label_canvas.yview,
                             bg=COLORS["bg_card"], troughcolor=COLORS["bg"])
        hsb2 = tk.Scrollbar(frame, orient="horizontal",
                             command=self._label_canvas.xview,
                             bg=COLORS["bg_card"], troughcolor=COLORS["bg"])
        self._label_canvas.configure(yscrollcommand=vsb2.set,
                                     xscrollcommand=hsb2.set)
        self._label_canvas.grid(row=0, column=0, sticky="nsew")
        vsb2.grid(row=0, column=1, sticky="ns")
        hsb2.grid(row=1, column=0, sticky="ew")

        self._label_photo    = None
        self._label_info_lbl = tk.Label(
            frame, text="No label image loaded.",
            bg=COLORS["bg"], fg=COLORS["text_muted"], font=FONT_BODY,
        )
        self._label_info_lbl.grid(row=0, column=0)

        return frame

    # ── Status bar ────────────────────────────────────────────────────────

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=COLORS["bg_card"],
                       highlightthickness=1,
                       highlightbackground=COLORS["border"],
                       pady=4)
        bar.grid(row=2, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)

        tk.Label(bar, textvariable=self._status_var,
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONT_SMALL, anchor="w", padx=12).grid(
            row=0, column=0, sticky="w")
        tk.Label(bar, text="EVO MDD Verification Tool  ·  Philips Respironics",
                 bg=COLORS["bg_card"], fg="#334155",
                 font=FONT_SMALL, padx=12).grid(row=0, column=1, sticky="e")

    # ── Canvas helpers ────────────────────────────────────────────────────

    def _on_grid_configure(self, _event=None):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self._canvas.itemconfig(self._canvas_win, width=event.width)

    # ── Logging ───────────────────────────────────────────────────────────

    def _log(self, msg: str, level: str = "INFO"):
        try:
            self.log_text.configure(state="normal")
            prefix = {"INFO": "› ", "WARN": "⚠ ", "ERROR": "✗ ", "OK": "✔ "}.get(level, "  ")
            self.log_text.insert("end", prefix + msg + "\n", level)
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
            self.root.update_idletasks()
        except Exception:
            pass

    # ── Run ───────────────────────────────────────────────────────────────

    def _run(self):
        if self._running:
            return

        label_paths = self._label_paths
        icon_dir    = self._icon_dir.get().strip()
        config      = self._config_path.get().strip()

        if not label_paths:
            messagebox.showerror("Missing Input",
                                 "Please select at least one label image or PDF.",
                                 parent=self.root)
            return
        missing_files = [p for p in label_paths if not os.path.isfile(p)]
        if missing_files:
            messagebox.showerror("File Not Found",
                                 "These files were not found:\n" +
                                 "\n".join(missing_files),
                                 parent=self.root)
            return
        if not icon_dir or not os.path.isdir(icon_dir):
            messagebox.showerror("Missing Input",
                                 "Please select a valid Icon Library folder.",
                                 parent=self.root)
            return

        # Collect only the checked icons
        selected_icon_paths = [
            p for p, v in zip(self._icon_list_paths, self._icon_check_vars)
            if v.get()
        ]
        if not selected_icon_paths:
            messagebox.showerror(
                "No Icons Selected",
                "Please check at least one icon in the 'Icons to Verify' list.\n\n"
                "Use 'Select All' to choose every icon in the library.",
                parent=self.root,
            )
            return

        n = len(label_paths)
        self._running        = True
        self._batch_total    = 0
        self._batch_found    = 0
        self._batch_missing  = 0

        self._verify_btn.configure(state="disabled",
                                   text=f"Running…  (0 / {n})")
        self._pb_var.set(0.0)
        self._batch_lbl_var.set(
            f"Preparing…  0 / {n} label(s)"
        )
        self._status_var.set(f"Starting batch of {n} label(s)…")
        self._clear_results()

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        # Load preview of the first label immediately
        self._load_label_preview(label_paths[0])

        threading.Thread(
            target=self._run_batch,
            args=(label_paths, selected_icon_paths, config),
            daemon=True,
        ).start()

    def _load_label_preview(self, path: str):
        """Display the label (or PDF page 1) in the preview tab."""
        try:
            self._label_info_lbl.grid_remove()
            img = _open_image_from_path(path)
            img.thumbnail((1000, 1000), Image.LANCZOS)
            self._label_photo = ImageTk.PhotoImage(img)
            self._label_canvas.delete("all")
            self._label_canvas.create_image(4, 4, anchor="nw",
                                            image=self._label_photo)
            self._label_canvas.configure(
                scrollregion=(0, 0, img.width + 8, img.height + 8)
            )
            self._nb.select(self._label_verification_tab)
            self._label_sub_nb.select(self._label_tab)
        except Exception as e:
            self._log(f"Could not load label preview: {e}", "WARN")

    # ── Batch verification thread ─────────────────────────────────────────

    def _run_batch(self, label_paths: list, icon_paths: list, config_path: str):
        """Run verification for each label sequentially and post progress updates."""
        n = len(label_paths)
        self._log(f"Icons selected: {len(icon_paths)}")
        if config_path:
            self._log(f"Config: {os.path.basename(config_path)}")

        try:
            from label_verifier.controller import Controller, _load_config_from_ini
            config = {}
            if config_path and os.path.isfile(config_path):
                try:
                    config = _load_config_from_ini(config_path)
                except Exception:
                    logger.exception("Failed to load config — using defaults")
            ctrl = Controller(config=config)
            self._log("Controller ready.", "OK")
        except Exception as exc:
            logger.exception("Controller init failed")
            self.root.after(0, lambda e=exc: self._on_error(e))
            return

        for idx, path in enumerate(label_paths):
            fname = os.path.basename(path)

            # ── Update progress label before this label starts ────────
            self.root.after(0, lambda i=idx, total=n, f=fname:
                self._update_batch_progress(i, total, f, in_progress=True))

            self._log(f"[{idx+1}/{n}] {fname}")

            try:
                results, summary = ctrl.run(
                    input_paths=[path],
                    icon_paths=icon_paths,
                )
                self.root.after(
                    0,
                    lambda r=results, s=summary, p=path, i=idx, total=n:
                        self._on_single_result(r, s, p, i, total),
                )
            except Exception as exc:
                logger.exception(f"Verification failed for {fname}")
                self.root.after(
                    0,
                    lambda e=exc, f=fname, i=idx, total=n:
                        self._on_single_error(e, f, i, total),
                )

        # ── All done ──────────────────────────────────────────────────
        self.root.after(0, self._on_batch_complete)

    def _update_batch_progress(self, done_count: int, total: int,
                                current_fname: str, in_progress: bool = False):
        """Update the progress bar and batch label text on the main thread."""
        pct = (done_count / total) * 100.0 if total else 100.0
        self._pb_var.set(pct)

        if in_progress:
            self._batch_lbl_var.set(
                f"Label {done_count + 1} / {total}  —  {current_fname}"
            )
            self._verify_btn.configure(
                text=f"Running…  ({done_count} / {total} done)"
            )
        else:
            self._batch_lbl_var.set(
                f"Completed {done_count} / {total}  —  {current_fname}"
            )
            self._verify_btn.configure(
                text=f"Running…  ({done_count} / {total} done)"
            )

    # ── Per-label results handling ────────────────────────────────────────

    def _on_single_result(self, results, summary, label_path: str,
                           idx: int, total: int):
        """Called on the main thread after each individual label finishes."""
        fname = os.path.basename(label_path)

        # Normalise result format
        if isinstance(results, list) and results and hasattr(results[0], "decision"):
            symbols = [
                {
                    "found":      getattr(r, "decision", None) == "Pass",
                    "symbol":     getattr(r, "icon_name", ""),
                    "score":      getattr(r, "score", 0.0),
                    "match_snip": getattr(r, "match_snip", None),
                    "icon_snip":  getattr(r, "icon_snip", None),
                }
                for r in results
            ]
        elif isinstance(results, list):
            symbols = results
        else:
            symbols = results.get("matches", [])

        n_total   = len(symbols)
        n_found   = sum(1 for s in symbols if s.get("found", False))
        n_missing = n_total - n_found
        passed    = n_missing == 0

        # Accumulate totals
        self._batch_total   += n_total
        self._batch_found   += n_found
        self._batch_missing += n_missing

        status_text = "PASS ✔" if passed else f"FAIL ✗ ({n_missing} missing)"
        self._log(
            f"[{idx+1}/{total}] {fname}  →  {n_found}/{n_total}  {status_text}",
            "OK" if passed else "WARN",
        )

        # Append a section header + cards into the shared grid
        self._append_label_section(fname, symbols, passed, idx + 1, total)

        # Update progress bar to reflect completion of this label
        self._update_batch_progress(idx + 1, total, fname, in_progress=False)

        # Show Label Results sub-tab while batch is running
        self._nb.select(self._label_verification_tab)
        self._label_sub_nb.select(self._grid_tab)

    def _on_single_error(self, exc, fname: str, idx: int, total: int):
        """Called on the main thread when a single label fails."""
        self._log(f"[{idx+1}/{total}] {fname}  →  ERROR: {exc}", "ERROR")
        self._append_error_section(fname, exc, idx + 1, total)
        self._update_batch_progress(idx + 1, total, fname, in_progress=False)
        self._nb.select(self._label_verification_tab)
        self._label_sub_nb.select(self._grid_tab)

    def _on_batch_complete(self):
        """Called on the main thread after all labels have been processed."""
        self._running = False
        n = len(self._label_paths)

        self._verify_btn.configure(state="normal", text="▶  Start Verification")
        self._pb_var.set(100.0)
        self._batch_lbl_var.set(
            f"All {n} label(s) complete  ·  "
            f"{self._batch_found}/{self._batch_total} symbols found"
        )

        all_passed = self._batch_missing == 0
        self._labels_var.set(str(n))
        self._total_var.set(str(self._batch_total))
        self._found_var.set(str(self._batch_found))
        self._missing_var.set(str(self._batch_missing))

        if all_passed:
            self._verdict_var.set("✔  PASS")
            self._verdict_lbl.configure(fg=COLORS["success"])
        else:
            self._verdict_var.set("✗  FAIL")
            self._verdict_lbl.configure(fg=COLORS["danger"])

        self._status_var.set(
            f"Batch complete — {n} label(s)  ·  "
            f"{self._batch_found}/{self._batch_total} symbols found  ·  "
            f"{'PASS' if all_passed else 'FAIL'}"
        )
        self._log(
            f"Batch done: {n} label(s), "
            f"{self._batch_found}/{self._batch_total} symbols found  "
            f"({'PASS' if all_passed else 'FAIL'})",
            "OK" if all_passed else "WARN",
        )

    def _on_error(self, exc):
        """Fatal error — controller could not be initialised."""
        self._running = False
        self._verify_btn.configure(state="normal", text="▶  Start Verification")
        self._pb_var.set(0.0)
        self._batch_lbl_var.set("")
        self._status_var.set(f"Error: {exc}")
        self._verdict_var.set("ERROR")
        self._verdict_lbl.configure(fg=COLORS["danger"])
        self._log(str(exc), "ERROR")
        messagebox.showerror("Verification Error", str(exc), parent=self.root)

    # ── Grid population helpers ───────────────────────────────────────────

    def _clear_results(self):
        for w in self._grid_inner.winfo_children():
            w.destroy()
        self._photo_cache.clear()
        self._verdict_var.set("—")
        self._labels_var.set("—")
        self._total_var.set("—")
        self._found_var.set("—")
        self._missing_var.set("—")
        self._verdict_lbl.configure(fg=COLORS["text_muted"])

    def _append_label_section(self, fname: str, symbols: list,
                               passed: bool, num: int, total: int):
        """Insert a section header + symbol cards for one label into _grid_inner."""
        # ── Section header row ────────────────────────────────────────
        hdr = tk.Frame(self._grid_inner, bg=COLORS["bg_card"],
                       highlightthickness=1,
                       highlightbackground=COLORS["border"])
        hdr.pack(fill="x", padx=4, pady=(10 if num > 1 else 4, 4))

        badge_bg = COLORS["success"] if passed else COLORS["danger"]
        badge_tx = "PASS ✔" if passed else "FAIL ✗"

        tk.Label(hdr, text=f"  Label {num} / {total}",
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONT_SMALL, padx=4, pady=6).pack(side="left")
        tk.Label(hdr, text=fname,
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=("Segoe UI", 9, "bold"), padx=4).pack(side="left")
        tk.Label(hdr, text=badge_tx,
                 bg=badge_bg, fg="white",
                 font=("Segoe UI", 8, "bold"),
                 padx=8, pady=4).pack(side="right", padx=6, pady=4)

        # ── Symbol cards grid ─────────────────────────────────────────
        if not symbols:
            tk.Label(self._grid_inner,
                     text="No symbols returned for this label.",
                     bg=COLORS["bg"], fg=COLORS["text_muted"],
                     font=FONT_BODY).pack(pady=10)
            return

        COLS = 4
        grid_host = tk.Frame(self._grid_inner, bg=COLORS["bg"])
        grid_host.pack(fill="x", padx=4, pady=(0, 6))

        for i, sym in enumerate(symbols):
            col = i % COLS
            row = i // COLS
            grid_host.columnconfigure(col, weight=1)
            self._make_symbol_card(grid_host, sym, row, col)

        # Trigger scroll region update
        self._grid_inner.update_idletasks()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _append_error_section(self, fname: str, exc: Exception,
                               num: int, total: int):
        """Insert an error notice for a failed label."""
        hdr = tk.Frame(self._grid_inner, bg=COLORS["bg_card"],
                       highlightthickness=1,
                       highlightbackground=COLORS["danger"])
        hdr.pack(fill="x", padx=4, pady=(10 if num > 1 else 4, 4))

        tk.Label(hdr, text=f"  Label {num} / {total}",
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONT_SMALL, padx=4, pady=6).pack(side="left")
        tk.Label(hdr, text=fname,
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=("Segoe UI", 9, "bold"), padx=4).pack(side="left")
        tk.Label(hdr, text="ERROR ✗",
                 bg=COLORS["danger"], fg="white",
                 font=("Segoe UI", 8, "bold"),
                 padx=8, pady=4).pack(side="right", padx=6, pady=4)

        tk.Label(self._grid_inner, text=str(exc),
                 bg=COLORS["bg"], fg=COLORS["danger"],
                 font=FONT_SMALL, anchor="w", padx=12,
                 wraplength=700).pack(fill="x", pady=(0, 6))

    def _make_symbol_card(self, parent, sym: dict, row: int, col: int):
        found  = sym.get("found", False)
        name   = sym.get("symbol") or sym.get("title") or sym.get("name") or "?"
        score  = sym.get("score", None)
        fname  = sym.get("filename") or sym.get("image") or ""
        method = sym.get("method", "")

        border_color = COLORS["success"] if found else COLORS["danger"]

        card = tk.Frame(parent, bg=COLORS["bg_card"],
                        highlightthickness=2,
                        highlightbackground=border_color)
        card.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

        tk.Label(card,
                 text="FOUND" if found else "MISSING",
                 bg=COLORS["success"] if found else COLORS["danger"],
                 fg="white", font=FONT_SMALL, padx=6, pady=2).pack(fill="x")

        thumb_frame = tk.Frame(card, bg=COLORS["bg_card"], height=90)
        thumb_frame.pack(fill="x", pady=4)
        thumb_frame.pack_propagate(False)

        icon_dir = self._icon_dir.get()
        img_path = os.path.join(icon_dir, fname) if fname else ""

        loaded = False
        if os.path.isfile(img_path):
            try:
                img = Image.open(img_path)
                img.thumbnail((80, 80), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._photo_cache.append(photo)
                tk.Label(thumb_frame, image=photo,
                         bg=COLORS["bg_card"]).pack(expand=True)
                loaded = True
            except Exception:
                pass

        if not loaded:
            icon_snip = sym.get("icon_snip")
            if icon_snip is not None:
                try:
                    import cv2
                    snip_rgb = cv2.cvtColor(icon_snip, cv2.COLOR_BGR2RGB)
                    pil_img  = Image.fromarray(snip_rgb)
                    pil_img.thumbnail((80, 80), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(pil_img)
                    self._photo_cache.append(photo)
                    tk.Label(thumb_frame, image=photo,
                             bg=COLORS["bg_card"]).pack(expand=True)
                    loaded = True
                except Exception:
                    pass

        if not loaded:
            tk.Label(thumb_frame, text="?", bg=COLORS["bg_card"],
                     fg=COLORS["text_muted"],
                     font=("Segoe UI", 22)).pack(expand=True)

        tk.Label(card, text=name, bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONT_SMALL, wraplength=130, justify="center").pack(
            padx=6, pady=(0, 2))

        if score is not None:
            pct = f"{score * 100:.1f}%" if score <= 1 else f"{score:.1f}"
            score_fg = (COLORS["success"] if score >= 0.7
                        else COLORS["warn"] if score >= 0.4
                        else COLORS["danger"])
            tk.Label(card, text=f"Score: {pct}",
                     bg=COLORS["bg_card"], fg=score_fg,
                     font=FONT_SMALL).pack(pady=(0, 2))

        if method:
            tk.Label(card, text=method, bg=COLORS["bg_card"],
                     fg=COLORS["text_muted"], font=FONT_SMALL).pack(pady=(0, 4))


if __name__ == "__main__":
    root = tk.Tk()
    MainGUI(root)
    root.mainloop()