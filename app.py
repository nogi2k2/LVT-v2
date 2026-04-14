"""
EVO MDD Verification Tool — Main Launcher
"""

import logging
import sys
import os
import threading
import tkinter as tk
from tkinter import messagebox

from gui.theme import (
    COLORS, darken, FONT_SMALL, contrast_text,
)
from core.state_manager import APP_STATE, register_callback
from gui.windows.iso_upload import library_is_built

LAUNCHER_LIGHT_COLORS = dict(COLORS)
LAUNCHER_DARK_COLORS = {
    **LAUNCHER_LIGHT_COLORS,
    "bg": "#0F172A",
    "bg_card": "#111827",
    "bg_input": "#111827",
    "accent": "#60A5FA",
    "text": "#E5E7EB",
    "text_muted": "#94A3B8",
    "border": "#334155",
    "header_bg": "#1D4ED8",
    "sidebar": "#020617",
    "sidebar_hover": "#0F172A",
    "header_subtle": "#BFDBFE",
    "sidebar_section": "#93C5FD",
    "sidebar_status": "#CBD5E1",
    "sidebar_footer": "#64748B",
    "auto_card_bg": "#0B2447",
    "auto_card_border": "#1D4ED8",
    "auto_success": "#4ADE80",
    "log_bg": "#0F172A",
    "log_text": "#E2E8F0",
}

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'error.log'), encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

class LauncherApp:
    _TOOLS = [
        {
            "id":       "iso_symbol_extractor",
            "title":    "ISO Symbol Library",
            "subtitle": "Upload ISO 15223 standard PDF\nSymbols extracted automatically · One-time setup",
            "step":     "Step 1",
            "color":    "#16A34A",
            "icon":     "⬢",
            "module":   "iso_symbol_extractor",
            "btn":      "Open Symbol Library",
        },
        {
            "id":       "vv_plan",
            "title":    "V&V Plan",
            "subtitle": "Upload V&V Plan document (.docx)\nLoads requirements · One-time setup",
            "step":     "Step 2",
            "color":    "#7C3AED",
            "icon":     "📋",
            "module":   "vv_plan",
            "btn":      "Load V&V Plan",
        },
        {
            "id":       "label_verifier",
            "title":    "Label Verification",
            "subtitle": "Verify label symbols against\nISO 15223 & V&V Plan requirements",
            "step":     "Step 3",
            "color":    "#0B54A4",
            "icon":     "✦",
            "module":   "label_verifier",
            "btn":      "Open Label Verification",
        },
        {
            "id":       "ifu_verifier",
            "title":    "IFU / Addendum Verification",
            "subtitle": "Verify IFU document meets\nV&V Plan PRD requirements",
            "step":     "Step 4",
            "color":    "#E87722",
            "icon":     "📄",
            "module":   "ifu_verifier",
            "btn":      "Open IFU Verification",
        },
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EVO MDD Verification Tool")
        self.root.resizable(True, True)
        self._open_windows: dict = {}
        self._status_vars:  dict = {}
        self._status_cache: dict = {}
        self._theme_name = "light"
        self._render_launcher()
        self._center()
        self._setup_automation_feedback()
        self._check_library_on_startup()

    def _colors(self) -> dict:
        return LAUNCHER_LIGHT_COLORS if self._theme_name == "light" else LAUNCHER_DARK_COLORS

    def _theme_button_label(self) -> str:
        return "☀ Light" if self._theme_name == "dark" else "☾ Dark"

    def _resolve_color(self, color_token: str) -> str:
        return self._colors().get(color_token, color_token)

    def _toggle_launcher_theme(self):
        self._theme_name = "dark" if self._theme_name == "light" else "light"
        self._render_launcher()

    def _render_launcher(self):
        self.root.configure(bg=self._colors()["bg"])
        for child in self.root.winfo_children():
            child.destroy()
        self._status_vars = {}
        self._build_ui()
        for tool_id, state in self._status_cache.items():
            self._set_tool_status(tool_id, state["message"], state["color"])

    def _center(self):
        self.root.update_idletasks()
        w, h = 980, 660
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self.root.minsize(720, 500)

    def _check_library_on_startup(self):
        from utils.paths import get_icon_dir
        if library_is_built():
            icon_dir = get_icon_dir()
            APP_STATE["icon_library_path"] = icon_dir
            n = len([f for f in os.listdir(icon_dir) if f.lower().endswith(".png")])
            self._set_tool_status("iso_symbol_extractor", f"✔  {n} symbols ready", "success")
            self._set_tool_status("label_verifier", "⚡ Symbol library auto-loaded", "accent")

    def _setup_automation_feedback(self):
        register_callback("icon_library_ready", self._on_icon_library_ready)
        register_callback("symbols_ready",      self._on_symbols_ready)
        register_callback("vv_plan_ready",      self._on_vv_plan_ready)

    def _on_icon_library_ready(self, path: str):
        n = len([f for f in os.listdir(path) if f.lower().endswith(".png")]) if os.path.isdir(path) else 0
        self.root.after(0, self._set_tool_status, "iso_symbol_extractor", f"✔  {n} symbols saved", "success")
        self.root.after(0, self._set_tool_status, "label_verifier", "⚡ Symbol library ready", "accent")
        self.root.after(0, self._set_tool_status, "ifu_verifier", "⚡ Symbol library ready", "accent")

    def _on_symbols_ready(self, summary: list):
        needed = [s["symbol"] for s in summary if s.get("gtin_count", 0) > 0]
        if needed:
            self.root.after(0, self._set_tool_status, "label_verifier", f"⚡ {', '.join(needed)} symbol(s) from V&V Plan", "accent")

    def _on_vv_plan_ready(self, data):
        n_label = len(getattr(data, "label_prds", []))
        n_ifu   = len(getattr(data, "ifu_prds",   []))
        self.root.after(0, self._set_tool_status, "vv_plan", f"✔  {n_label} label · {n_ifu} IFU PRDs", "success")
        self.root.after(0, self._set_tool_status, "label_verifier", f"⚡ V&V Plan loaded — {n_label} PRD(s)", "accent")
        self.root.after(0, self._set_tool_status, "ifu_verifier", f"⚡ V&V Plan loaded — {n_ifu} IFU PRD(s)", "accent")

    def _set_tool_status(self, tool_id: str, msg: str, color: str):
        self._status_cache[tool_id] = {"message": msg, "color": color}
        entry = self._status_vars.get(tool_id)
        if entry:
            entry["text"].set(msg)
            entry["label"].configure(fg=self._resolve_color(color))

    def _build_ui(self):
        c = self._colors()
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self._build_sidebar()
        self._build_main_area()

    def _build_sidebar(self):
        c = self._colors()
        sidebar = tk.Frame(self.root, bg=c["sidebar"], width=240)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        brand = tk.Frame(sidebar, bg=c["header_bg"], pady=20, padx=18)
        brand.pack(fill="x")
        tk.Label(brand, text="EVO MDD", bg=c["header_bg"], fg="white", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(brand, text="Verification Tool", bg=c["header_bg"], fg=c["header_subtle"], font=("Segoe UI", 9)).pack(anchor="w")
        tk.Label(brand, text="Philips Respironics · Systems V&V", bg=c["header_bg"], fg=c["sidebar_status"], font=FONT_SMALL).pack(anchor="w", pady=(4, 0))

        tk.Label(sidebar, text="WORKFLOW", bg=c["sidebar"], fg=c["sidebar_section"], font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=18, pady=(18, 4))
        for tool in self._TOOLS:
            self._make_nav_item(sidebar, tool)

        tk.Label(sidebar, text="AUTO-LINKS", bg=c["sidebar"], fg=c["sidebar_section"], font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=18, pady=(22, 4))
        auto_card = tk.Frame(sidebar, bg=c["auto_card_bg"], highlightthickness=1, highlightbackground=c["auto_card_border"])
        auto_card.pack(fill="x", padx=12, pady=2)
        inner = tk.Frame(auto_card, bg=c["auto_card_bg"], padx=12, pady=10)
        inner.pack(fill="x")
        tk.Label(inner, text="✔  Automation active", bg=c["auto_card_bg"], fg=c["auto_success"], font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tk.Label(inner, text="Step 1 → Steps 3 & 4  icon library\nStep 2 → Steps 3 & 4  requirements", bg=c["auto_card_bg"], fg=c["sidebar_status"], font=FONT_SMALL, justify="left").pack(anchor="w", pady=(4, 0))

        tk.Label(sidebar, text="v2.0  ·  EVO MDD", bg=c["sidebar"], fg=c["sidebar_footer"], font=FONT_SMALL).pack(side="bottom", pady=10)

    def _make_nav_item(self, parent, tool: dict):
        c = self._colors()
        item = tk.Frame(parent, bg=c["sidebar"], cursor="hand2")
        item.pack(fill="x", padx=6, pady=1)

        bar = tk.Frame(item, bg=c["sidebar"], width=4)
        bar.pack(side="left", fill="y")

        content = tk.Frame(item, bg=c["sidebar"], padx=10, pady=9)
        content.pack(side="left", fill="both", expand=True)

        tk.Label(content, text=tool["step"], bg=c["sidebar"], fg=c["sidebar_section"], font=("Segoe UI", 7, "bold")).pack(anchor="w")
        tk.Label(content, text=tool["title"].replace("\n", " "), bg=c["sidebar"], fg="white", font=("Segoe UI", 9, "bold"), anchor="w").pack(anchor="w")

        status_var = tk.StringVar(value="")
        status_lbl = tk.Label(content, textvariable=status_var, bg=c["sidebar"], fg=c["sidebar_status"], font=FONT_SMALL, anchor="w")
        status_lbl.pack(anchor="w")
        self._status_vars[tool["id"]] = {"text": status_var, "label": status_lbl}

        color = tool["color"]
        all_widgets = [item, bar, content, status_lbl]

        def on_enter(e, ws=all_widgets, b=bar, c=color):
            for w in ws: w.configure(bg=self._colors()["sidebar_hover"])
            b.configure(bg=c, width=4)

        def on_leave(e, ws=all_widgets, b=bar):
            base = self._colors()["sidebar"]
            for w in ws: w.configure(bg=base)
            b.configure(bg=base, width=4)

        for w in all_widgets:
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)
            w.bind("<Button-1>", lambda e, m=tool["module"]: self._open_tool(m))

    def _build_main_area(self):
        c = self._colors()
        main = tk.Frame(self.root, bg=c["bg"])
        main.grid(row=0, column=1, sticky="nsew")
        main.rowconfigure(1, weight=1)
        main.columnconfigure(0, weight=1)

        topbar = tk.Frame(main, bg=c["bg_card"], highlightthickness=1, highlightbackground=c["border"], padx=28, pady=16)
        topbar.grid(row=0, column=0, sticky="ew")
        topbar.columnconfigure(0, weight=1)
        title_wrap = tk.Frame(topbar, bg=c["bg_card"])
        title_wrap.grid(row=0, column=0, sticky="w")
        tk.Label(title_wrap, text="EVO MDD Verification Tool", bg=c["bg_card"], fg=c["accent"], font=("Segoe UI", 15, "bold")).pack(anchor="w")
        tk.Label(title_wrap, text="Automated medical symbol verification  ·  Trilogy EVO / Helix  ·  Philips Respironics", bg=c["bg_card"], fg=c["text_muted"], font=("Segoe UI", 9)).pack(anchor="w", pady=(3, 0))
        tk.Button(
            topbar,
            text=self._theme_button_label(),
            command=self._toggle_launcher_theme,
            bg=c["bg_card"],
            fg=c["text"],
            activebackground=c["border"],
            activeforeground=c["text"],
            relief="flat",
            cursor="hand2",
            font=FONT_SMALL,
            padx=10,
            pady=4,
            bd=0,
            highlightthickness=1,
            highlightbackground=c["border"],
        ).grid(row=0, column=1, sticky="e")

        cards_area = tk.Frame(main, bg=c["bg"])
        cards_area.grid(row=1, column=0, sticky="nsew", padx=20, pady=16)
        cards_area.columnconfigure(0, weight=1)
        cards_area.columnconfigure(1, weight=1)
        cards_area.rowconfigure(0, weight=1)
        cards_area.rowconfigure(1, weight=1)

        for i, tool in enumerate(self._TOOLS):
            self._make_tool_card(cards_area, tool, i // 2, i % 2)

        info = tk.Frame(main, bg=c["bg_card"], highlightthickness=1, highlightbackground=c["border"], padx=24, pady=8)
        info.grid(row=2, column=0, sticky="ew")
        tk.Label(info, text="Steps 1 & 2 are one-time setup  ·  Step 3: Label Verification uses the ISO library + V&V Plan  ·  Step 4: IFU / Addendum Verification uses the V&V Plan", bg=c["bg_card"], fg=c["text_muted"], font=FONT_SMALL).pack(anchor="w")

    def _make_tool_card(self, parent, tool: dict, row: int, col: int):
        c = self._colors()
        color = tool["color"]
        pad_l = 0 if col == 0 else 6
        pad_r = 0 if col == len(self._TOOLS) - 1 else 6

        card = tk.Frame(parent, bg=c["bg_card"], highlightthickness=1, highlightbackground=c["border"], cursor="hand2")
        card.grid(row=row, column=col, sticky="nsew", padx=(pad_l, pad_r), pady=4)
        card.columnconfigure(0, weight=1)

        tk.Frame(card, bg=color, height=4).pack(fill="x")

        body = tk.Frame(card, bg=c["bg_card"], padx=20, pady=18)
        body.pack(fill="both", expand=True)

        tk.Label(body, text=tool["step"], bg=color, fg=contrast_text(color), font=("Segoe UI", 7, "bold"), padx=8, pady=3).pack(anchor="w", pady=(0, 10))
        tk.Label(body, text=tool["icon"], bg=c["bg_card"], fg=color, font=("Segoe UI", 24)).pack(anchor="w")

        for line in tool["title"].split("\n"):
            tk.Label(body, text=line, bg=c["bg_card"], fg=c["text"], font=("Segoe UI", 11, "bold"), anchor="w").pack(anchor="w")

        tk.Frame(body, bg=c["border"], height=1).pack(fill="x", pady=(8, 8))
        for line in tool["subtitle"].split("\n"):
            tk.Label(body, text=line, bg=c["bg_card"], fg=c["text_muted"], font=FONT_SMALL, anchor="w").pack(anchor="w")

        btn = tk.Button(body, text=tool["btn"], command=lambda m=tool["module"]: self._open_tool(m), bg=color, fg=contrast_text(color), activebackground=darken(color), activeforeground=contrast_text(color), relief="flat", cursor="hand2", font=("Segoe UI", 9, "bold"), padx=16, pady=8, bd=0)
        btn.pack(anchor="w", pady=(14, 0))

        def _enter(e, c=card, col=color): c.configure(highlightbackground=col)
        def _leave(e, c=card):            c.configure(highlightbackground=self._colors()["border"])
        for w in (card, body):
            w.bind("<Enter>", _enter)
            w.bind("<Leave>", _leave)

    def _open_tool(self, module_name: str):
        existing = self._open_windows.get(module_name)
        if existing and existing.winfo_exists():
            self._focus_window(existing)
            return

        try:
            if module_name == "iso_symbol_extractor":
                from gui.windows.iso_upload import ISOLibraryUploadWindow
                win = ISOLibraryUploadWindow(self.root)
                self._open_windows[module_name] = win
                self._attach_child_window(module_name, win)
                self._focus_window(win)
                return

            if module_name == "vv_plan":
                from gui.windows.vv_plan import VVPlanWindow
                win = VVPlanWindow(self.root)
                self._open_windows[module_name] = win
                self._attach_child_window(module_name, win)
                self._focus_window(win)
                return

            if module_name == "label_verifier":
                from label_verifier.gui.main_gui import MainGUI
                win = tk.Toplevel(self.root)
                MainGUI(win, mode="label")
                self._open_windows[module_name] = win
                self._attach_child_window(module_name, win)
                self._focus_window(win)
                return

            if module_name == "ifu_verifier":
                from ifu_verifier.ifu_verifier_window import IFUVerifierWindow
                plan_data = APP_STATE.get("vv_plan_data")
                ifu_prds  = list(getattr(plan_data, "ifu_prds", [])) if plan_data else []
                if not ifu_prds and not plan_data:
                    messagebox.showinfo("Load V&V Plan First", "Please load a V&V Plan (Step 2) before opening IFU Verification.\nThe plan provides the IFU requirements to verify.", parent=self.root)
                    return
                win = IFUVerifierWindow(self.root, ifu_prds=ifu_prds, plan_data=plan_data)
                self._open_windows[module_name] = win
                self._attach_child_window(module_name, win)
                self._focus_window(win)
                return

        except ImportError as exc:
            messagebox.showerror("Module Not Found", f"Could not load '{module_name}'.\n\n{exc}", parent=self.root)
        except Exception as exc:
            logger.exception("Failed to open tool: %s", module_name)
            messagebox.showerror("Error", f"Failed to open tool:\n{exc}", parent=self.root)

    @staticmethod
    def _focus_window(win: tk.Toplevel):
        win.lift(); win.focus_force()
        win.attributes("-topmost", True)
        win.after(200, lambda: win.attributes("-topmost", False))

    def _attach_child_window(self, module_name: str, win: tk.Toplevel):
        closed = {"done": False}

        def _restore_launcher(*_args):
            if closed["done"]:
                return
            closed["done"] = True
            self._open_windows.pop(module_name, None)
            if self.root.winfo_exists():
                self.root.deiconify()
                self.root.lift()
                self.root.focus_force()

        win.protocol("WM_DELETE_WINDOW", lambda: (win.destroy(), _restore_launcher()))
        win.bind("<Destroy>", lambda e: e.widget is win and _restore_launcher(), add="+")
        self.root.withdraw()

def run_app():
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()

def _log_exception(exc_type, exc_value, exc_tb):
    logger.exception("Unhandled exception", exc_info=(exc_type, exc_value, exc_tb))

def _thread_exception_handler(args):
    exc_type  = getattr(args, "exc_type", None)
    exc_value = getattr(args, "exc_value", None)
    exc_tb    = getattr(args, "exc_traceback", None)
    if exc_type and exc_value and exc_tb:
        _log_exception(exc_type, exc_value, exc_tb)

if __name__ == "__main__":
    sys.excepthook = _log_exception
    try: threading.excepthook = _thread_exception_handler
    except AttributeError: pass
    try: run_app()
    except Exception:
        logger.exception("Exception in run_app")
        sys.exit(1)