import tkinter as tk
from tkinter import ttk

COLORS = {
    "bg":           "#F5F7FA",
    "bg_card":      "#FFFFFF",
    "bg_input":     "#FFFFFF",
    "accent":       "#0B54A4",
    "accent_light": "#00A0DC",
    "accent2":      "#16A34A",
    "accent3":      "#D97706",
    "text":         "#1A1A2E",
    "text_muted":   "#6B7280",
    "border":       "#D1D5DB",
    "header_bg":    "#0B54A4",
    "danger":       "#DC2626",
    "success":      "#16A34A",
    "warn":         "#D97706",
    "log_bg":       "#F8FAFC",
    "log_text":     "#1E3A5F",
    "sidebar":      "#0B2B5C",
    "sidebar_hover":"#0B54A4",
    "header_subtle":"#BFDBFE",
    "header_faint": "#64748B",
    "sidebar_section":"#7B9CC8",
    "sidebar_status":"#93C5FD",
    "sidebar_footer":"#3D5A80",
    "auto_card_bg":"#0B3A7A",
    "auto_card_border":"#1E5FAD",
    "auto_success":"#4ADE80",
}

FONT_TITLE   = ("Segoe UI", 16, "bold")
FONT_HEADING = ("Segoe UI", 11, "bold")
FONT_BODY    = ("Segoe UI",  9)
FONT_MONO    = ("Consolas",  9)
FONT_SMALL   = ("Segoe UI",  8)

def get_theme_name() -> str:
    return "light"

def theme_toggle_label() -> str:
    return ""

def register_theme_callback(callback):
    return callback

def unregister_theme_callback(callback) -> None:
    return None

def contrast_text(hex_color: str) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return COLORS["text"]
    r, g, b = [int(color[i:i+2], 16) for i in (0, 2, 4)]
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#0F172A" if luminance > 0.6 else "white"

def apply_theme_to_widget_tree(widget, old_colors: dict, new_colors: dict) -> None:
    return None

def configure_global_ttk_styles(master=None) -> None:
    style = ttk.Style(master)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(
        "TNotebook",
        background=COLORS["bg"],
        borderwidth=0,
    )
    style.configure(
        "TNotebook.Tab",
        background=COLORS["bg_card"],
        foreground=COLORS["text_muted"],
        padding=[12, 6],
        font=FONT_BODY,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", COLORS["accent"]), ("active", lighten(COLORS["bg_card"]))],
        foreground=[("selected", "white"), ("active", COLORS["text"])],
    )
    style.configure(
        "TProgressbar",
        troughcolor=COLORS["border"],
        background=COLORS["accent"],
        bordercolor=COLORS["border"],
        lightcolor=COLORS["accent"],
        darkcolor=COLORS["accent"],
    )
    style.configure(
        "Treeview",
        background=COLORS["bg_card"],
        foreground=COLORS["text"],
        fieldbackground=COLORS["bg_card"],
        rowheight=26,
        font=FONT_SMALL,
    )
    style.configure(
        "Treeview.Heading",
        background=COLORS["bg_card"],
        foreground=COLORS["text_muted"],
        font=("Segoe UI", 8, "bold"),
        relief="flat",
    )
    style.map(
        "Treeview",
        background=[("selected", COLORS["accent"])],
        foreground=[("selected", "white")],
    )

def set_theme(name: str) -> None:
    return None

def toggle_theme() -> None:
    return None

def lighten(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return f"#{min(255,r+30):02x}{min(255,g+30):02x}{min(255,b+30):02x}"

def darken(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return f"#{max(0,r-20):02x}{max(0,g-20):02x}{max(0,b-20):02x}"

def styled_button(parent, text="", command=None, color=None, **kwargs) -> tk.Button:
    bg = color or COLORS["accent"]
    return tk.Button(
        parent, text=text, command=command, bg=bg, fg="white",
        activebackground=darken(bg), activeforeground="white",
        relief="flat", cursor="hand2", font=FONT_BODY, padx=12, pady=6, bd=0, **kwargs,
    )

class ProgressPopup(tk.Toplevel):
    _PENDING_FG = COLORS["text_muted"]
    _ACTIVE_FG  = COLORS["accent"]
    _DONE_FG    = COLORS["accent2"]
    _ERROR_FG   = COLORS["danger"]

    def __init__(self, parent, title: str = "Processing", steps: list = None):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=COLORS["bg"])
        self.resizable(False, False)
        self.grab_set()

        self._steps      = steps or []
        self._step_vars  = []
        self._detail_var = tk.StringVar(value="Starting…")
        self._status_var = tk.StringVar(value="")

        self._build_ui(title)
        self._center(parent)

    def _center(self, parent):
        self.update_idletasks()
        w, h = 480, 320
        try:
            px = parent.winfo_rootx() + parent.winfo_width()  // 2
            py = parent.winfo_rooty() + parent.winfo_height() // 2
        except Exception:
            px, py = 400, 300
        self.geometry(f"{w}x{h}+{px - w//2}+{py - h//2}")

    def _build_ui(self, title: str):
        hdr = tk.Frame(self, bg=COLORS["header_bg"], padx=16, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text=title, bg=COLORS["header_bg"], fg="white",
                 font=FONT_HEADING).pack(anchor="w")

        steps_frame = tk.Frame(self, bg=COLORS["bg_card"], padx=16, pady=12)
        steps_frame.pack(fill="x", padx=8, pady=(8, 0))

        for i, step_text in enumerate(self._steps):
            var = tk.StringVar(value=f"○  {step_text}")
            self._step_vars.append(var)
            lbl = tk.Label(steps_frame, textvariable=var,
                           bg=COLORS["bg_card"], fg=self._PENDING_FG,
                           font=FONT_BODY, anchor="w")
            lbl.pack(fill="x", pady=1)
            self._step_vars[i] = (var, lbl, step_text)

        detail_frame = tk.Frame(self, bg=COLORS["bg"], padx=16, pady=6)
        detail_frame.pack(fill="x", padx=8)
        tk.Label(detail_frame, textvariable=self._detail_var,
                 bg=COLORS["bg"], fg=COLORS["text_muted"],
                 font=FONT_SMALL, anchor="w", wraplength=440).pack(fill="x")

        pb_frame = tk.Frame(self, bg=COLORS["bg"], padx=16, pady=4)
        pb_frame.pack(fill="x", padx=8)
        style = ttk.Style(self)
        style.configure("Popup.Horizontal.TProgressbar",
                        troughcolor=COLORS["border"],
                        background=COLORS["accent"], thickness=6)
        self._pb = ttk.Progressbar(pb_frame, orient="horizontal",
                                   mode="indeterminate",
                                   style="Popup.Horizontal.TProgressbar")
        self._pb.pack(fill="x")
        self._pb.start(20)

        bottom = tk.Frame(self, bg=COLORS["bg"], padx=16, pady=8)
        bottom.pack(fill="x", padx=8, side="bottom")
        tk.Label(bottom, textvariable=self._status_var,
                 bg=COLORS["bg"], fg=COLORS["text_muted"],
                 font=FONT_SMALL, anchor="w").pack(side="left", fill="x", expand=True)
        self._close_btn = tk.Button(
            bottom, text="Close", command=self.destroy,
            bg=COLORS["border"], fg=COLORS["text"],
            activebackground=lighten(COLORS["border"]),
            activeforeground=COLORS["text"],
            relief="flat", cursor="hand2",
            font=FONT_SMALL, padx=10, pady=4, bd=0, state="disabled",
        )
        self._close_btn.pack(side="right")

    def set_step(self, index: int, detail: str = ""):
        for i, (var, lbl, text) in enumerate(self._step_vars):
            if i < index:
                var.set(f"✔  {text}"); lbl.configure(fg=self._DONE_FG)
            elif i == index:
                var.set(f"▶  {text}"); lbl.configure(fg=self._ACTIVE_FG)
            else:
                var.set(f"○  {text}"); lbl.configure(fg=self._PENDING_FG)
        if detail:
            self._detail_var.set(detail)
        self.update_idletasks()

    def set_detail(self, detail: str):
        self._detail_var.set(detail)
        self.update_idletasks()

    def finish(self, msg: str = "Complete."):
        for var, lbl, text in self._step_vars:
            var.set(f"✔  {text}"); lbl.configure(fg=self._DONE_FG)
        self._detail_var.set(msg)
        self._status_var.set("Finished")
        self._pb.stop()
        self._pb.configure(mode="determinate")
        ttk.Style(self).configure("Popup.Horizontal.TProgressbar",
                                  background=COLORS["accent2"])
        self._pb["value"] = 100
        self._close_btn.configure(state="normal", bg=COLORS["accent2"],
                                  fg="white")
        self.update_idletasks()

    def error(self, msg: str = "An error occurred."):
        self._detail_var.set(f"Error: {msg}")
        self._status_var.set("Failed")
        self._pb.stop()
        self._pb.configure(mode="determinate")
        ttk.Style(self).configure("Popup.Horizontal.TProgressbar",
                                  background=COLORS["danger"])
        self._pb["value"] = 100
        self._close_btn.configure(state="normal", bg=COLORS["danger"], fg="white")
        for var, lbl, text in self._step_vars:
            if var.get().startswith("▶"):
                lbl.configure(fg=self._ERROR_FG)
        self.update_idletasks()