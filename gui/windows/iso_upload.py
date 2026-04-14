import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import logging

from utils.paths import get_icon_dir
from gui.theme import (
    COLORS, FONT_TITLE, FONT_SMALL, FONT_BODY, FONT_MONO, darken, contrast_text,
)
from core.state_manager import APP_STATE, fire_event

logger = logging.getLogger(__name__)

ISO_LIGHT_COLORS = dict(COLORS)
ISO_DARK_COLORS = {
    **ISO_LIGHT_COLORS,
    "bg": "#0F172A",
    "bg_card": "#111827",
    "bg_input": "#0B1220",
    "accent": "#60A5FA",
    "text": "#E5E7EB",
    "text_muted": "#94A3B8",
    "border": "#334155",
    "header_bg": "#1D4ED8",
    "header_subtle": "#BFDBFE",
    "success": "#4ADE80",
    "danger": "#F87171",
}

def library_is_built() -> bool:
    """Return True if the Icon Library already contains extracted symbols."""
    icon_dir = get_icon_dir()
    json_path = os.path.join(icon_dir, "iso_symbols.json")
    return os.path.isfile(json_path) and len(
        [f for f in os.listdir(icon_dir) if f.lower().endswith(".png")]
    ) > 0 if os.path.isdir(icon_dir) else False

class ISOLibraryUploadWindow(tk.Toplevel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title("ISO Symbol Library — EVO MDD Verification Tool")
        self.resizable(False, False)
        self._pdf_path = tk.StringVar()
        self._status   = tk.StringVar(value="")
        self._theme_name = "light"
        self._status_color = "text_muted"
        self._run_btn_mode = "build"
        self._is_running = False
        self._progress_mode = "determinate"
        self._progress_value = 0
        self._render_ui()
        self._center()
        if library_is_built():
            n = len([f for f in os.listdir(get_icon_dir()) if f.lower().endswith(".png")])
            self._set_done(f"Symbol library is ready — {n} symbols available.")

    def _colors(self) -> dict:
        return ISO_LIGHT_COLORS if self._theme_name == "light" else ISO_DARK_COLORS

    def _theme_button_label(self) -> str:
        return "☀ Light" if self._theme_name == "dark" else "☾ Dark"

    def _toggle_theme(self):
        self._theme_name = "dark" if self._theme_name == "light" else "light"
        self._render_ui()

    def _render_ui(self):
        self.configure(bg=self._colors()["bg"])
        for child in self.winfo_children():
            child.destroy()
        self._build_ui()
        self._apply_runtime_state()

    def _apply_runtime_state(self):
        c = self._colors()
        self._status_lbl.configure(fg=c.get(self._status_color, self._status_color))
        self._pb.configure(mode=self._progress_mode)
        self._pb_var.set(self._progress_value)
        if self._is_running and self._progress_mode == "indeterminate":
            self._pb.start(15)
        else:
            self._pb.stop()

        if self._run_btn_mode == "rebuild":
            bg = c["accent2"]
            fg = contrast_text(bg)
            self._run_btn.configure(
                state="normal",
                text="Rebuild Library",
                bg=bg,
                fg=fg,
                activebackground=darken(bg),
                activeforeground=fg,
            )
        else:
            bg = c["accent"]
            fg = contrast_text(bg)
            state = "disabled" if self._is_running else "normal"
            text = "Building…" if self._is_running else "Build Symbol Library"
            self._run_btn.configure(
                state=state,
                text=text,
                bg=bg,
                fg=fg,
                activebackground=darken(bg),
                activeforeground=fg,
            )

    def _center(self):
        self.update_idletasks()
        w, h = 560, 380
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _build_ui(self):
        c = self._colors()
        hdr = tk.Frame(self, bg=c["header_bg"], padx=20, pady=16)
        hdr.pack(fill="x")
        tk.Button(
            hdr,
            text=self._theme_button_label(),
            command=self._toggle_theme,
            bg=c["header_bg"],
            fg="white",
            activebackground=darken(c["header_bg"]),
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            font=FONT_SMALL,
            padx=10,
            pady=4,
            bd=0,
        ).pack(anchor="e")
        tk.Label(hdr, text="ISO Symbol Library", bg=c["header_bg"], fg="white", font=FONT_TITLE).pack(anchor="w")
        tk.Label(hdr, text="Upload the ISO 15223 standard PDF — symbols are extracted automatically", bg=c["header_bg"], fg=c["header_subtle"], font=FONT_SMALL).pack(anchor="w", pady=(4, 0))

        body = tk.Frame(self, bg=c["bg"], padx=28, pady=24)
        body.pack(fill="both", expand=True)

        tk.Label(body, text="The symbol reference library is built once from the ISO 15223 standard.\nUpload the PDF below and the extraction runs automatically in the background.", bg=c["bg"], fg=c["text"], font=FONT_BODY, justify="left", wraplength=500).pack(anchor="w", pady=(0, 20))

        pick_frame = tk.Frame(body, bg=c["bg"])
        pick_frame.pack(fill="x", pady=(0, 6))
        tk.Label(pick_frame, text="ISO 15223 Standard PDF", bg=c["bg"], fg=c["text_muted"], font=FONT_SMALL).pack(anchor="w", pady=(0, 4))

        row = tk.Frame(pick_frame, bg=c["bg"])
        row.pack(fill="x")
        entry = tk.Entry(row, textvariable=self._pdf_path, state="readonly", bg=c["bg_input"], fg=c["text"], readonlybackground=c["bg_input"], relief="flat", font=FONT_MONO, highlightthickness=1, highlightbackground=c["border"], insertbackground=c["text"])
        entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        tk.Button(row, text="Browse…", command=self._browse, bg=c["bg_card"], fg=c["text"], activebackground=c["border"], activeforeground=c["text"], relief="flat", cursor="hand2", font=FONT_SMALL, padx=10, pady=5, bd=0, highlightthickness=1, highlightbackground=c["border"]).pack(side="left")

        self._pb_var = tk.DoubleVar(value=0)
        self._pb = ttk.Progressbar(body, variable=self._pb_var, maximum=100, mode="indeterminate")
        self._pb.pack(fill="x", pady=(14, 4))

        self._status_lbl = tk.Label(body, textvariable=self._status, bg=c["bg"], fg=c["text_muted"], font=FONT_SMALL, anchor="w")
        self._status_lbl.pack(anchor="w")

        btn_row = tk.Frame(body, bg=c["bg"])
        btn_row.pack(fill="x", pady=(20, 0))
        self._run_btn = tk.Button(btn_row, text="Build Symbol Library", command=self._run, bg=c["accent"], fg=contrast_text(c["accent"]), activebackground=darken(c["accent"]), activeforeground=contrast_text(c["accent"]), relief="flat", cursor="hand2", font=("Segoe UI", 10, "bold"), padx=20, pady=10, bd=0)
        self._run_btn.pack(side="left")

        tk.Label(btn_row, text="This only needs to be done once.\nThe library is reused across all sessions.", bg=c["bg"], fg=c["text_muted"], font=FONT_SMALL, justify="left").pack(side="left", padx=16)

    def _refocus(self):
        self.lift(); self.focus_force()
        self.attributes("-topmost", True)
        self.after(150, lambda: self.attributes("-topmost", False))

    def _browse(self):
        from tkinter import filedialog
        p = filedialog.askopenfilename(parent=self, title="Select ISO 15223 Standard PDF", filetypes=[("PDF files", "*.pdf *.PDF")])
        if p:
            self._pdf_path.set(p)
            self._status.set("PDF selected. Click 'Build Symbol Library' to start.")
            self._status_color = "text_muted"
            self._status_lbl.configure(fg=self._colors()["text_muted"])
        self._refocus()

    def _run(self):
        if not self._pdf_path.get():
            messagebox.showwarning("No PDF Selected", "Please browse for the ISO 15223 standard PDF first.", parent=self)
            return
        self._is_running = True
        self._run_btn_mode = "build"
        self._progress_mode = "indeterminate"
        self._run_btn.configure(state="disabled", text="Building…")
        self._pb.configure(mode="indeterminate")
        self._pb.start(15)
        self._status.set("Extracting symbols from ISO 15223 PDF…")
        self._status_color = "accent"
        self._status_lbl.configure(fg=self._colors()["accent"])
        threading.Thread(target=self._extract_thread, daemon=True).start()

    def _extract_thread(self):
        try:
            from parsers.iso_symbol_extractor import extract_symbols_from_pdf, save_json_csv
            icon_dir = get_icon_dir()
            os.makedirs(icon_dir, exist_ok=True)

            def _silent_log(msg, level="INFO"): pass

            data = extract_symbols_from_pdf(self._pdf_path.get(), icon_dir, _silent_log, progress_fn=None)
            save_json_csv(data, icon_dir, _silent_log)

            APP_STATE["icon_library_path"] = icon_dir
            fire_event("icon_library_ready", icon_dir)

            self.after(0, self._set_done, f"Done — {len(data)} symbols extracted and saved to the library.")

        except Exception as e:
            logger.exception("ISO extraction failed")
            self.after(0, self._set_error, str(e))

    def _set_done(self, msg: str):
        self._is_running = False
        self._run_btn_mode = "rebuild"
        self._progress_mode = "determinate"
        self._progress_value = 100
        self._pb.stop()
        self._pb.configure(mode="determinate")
        self._pb_var.set(100)
        self._status.set(f"✔  {msg}")
        self._status_color = "success"
        self._status_lbl.configure(fg=self._colors()["success"])
        accent2 = self._colors()["accent2"]
        self._run_btn.configure(state="normal", text="Rebuild Library", bg=accent2, fg=contrast_text(accent2), activebackground=darken(accent2), activeforeground=contrast_text(accent2))

    def _set_error(self, msg: str):
        self._is_running = False
        self._run_btn_mode = "build"
        self._progress_mode = "determinate"
        self._progress_value = 0
        self._pb.stop()
        self._pb_var.set(0)
        self._status.set(f"✗  Error: {msg}")
        self._status_color = "danger"
        self._status_lbl.configure(fg=self._colors()["danger"])
        accent = self._colors()["accent"]
        self._run_btn.configure(state="normal", text="Build Symbol Library", bg=accent, fg=contrast_text(accent), activebackground=darken(accent), activeforeground=contrast_text(accent))