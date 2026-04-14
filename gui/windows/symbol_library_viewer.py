"""
Symbol Library Viewer — Window 3
====================================
Browse, search, and inspect all extracted ISO 15223 symbols.
Loads from iso_symbols.json (created by the extractor).
"""

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext

try:
    from app import (
        COLORS, FONT_TITLE, FONT_HEADING, FONT_BODY, FONT_MONO, FONT_SMALL, _lighten
    )
except ImportError:
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


class LibraryWindow(tk.Toplevel):
    """Browse all extracted ISO 15223 symbols."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title("ISO 15223 Symbol Library — Reference Viewer")
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)

        self._symbols     = []
        self._filtered    = []
        self._lib_dir     = tk.StringVar()
        self._search_var  = tk.StringVar()
        self._img_cache   = {}          # ref → PhotoImage
        self._current_img = None        # keep reference alive

        self._build_ui()
        self._center()

        # Auto-load if default location exists
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "symbol_images")
        if os.path.exists(os.path.join(default_dir, "iso_symbols.json")):
            self._lib_dir.set(default_dir)
            self._load_library(default_dir)

    def _center(self):
        self.update_idletasks()
        w, h = 1200, 780
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._setup_styles()

        # Header
        hdr = tk.Frame(self, bg="#2D1B69", padx=16, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="ISO 15223 Symbol Library", bg="#2D1B69",
                 fg="white", font=FONT_TITLE).pack(side="left")
        tk.Label(hdr, text="  ·  Reference Viewer", bg="#2D1B69",
                 fg="#C4B5FD", font=FONT_BODY).pack(side="left")

        # Toolbar
        toolbar = tk.Frame(self, bg=COLORS["bg_card"], padx=12, pady=8)
        toolbar.pack(fill="x")

        # Folder picker
        tk.Label(toolbar, text="Library Folder:", bg=COLORS["bg_card"],
                 fg=COLORS["text_muted"], font=FONT_SMALL).pack(side="left", padx=(0, 4))
        dir_entry = tk.Entry(toolbar, textvariable=self._lib_dir, state="readonly",
                             bg=COLORS["bg_input"], fg=COLORS["text"],
                             readonlybackground=COLORS["bg_input"],
                             relief="flat", font=FONT_MONO, width=40,
                             highlightthickness=1, highlightbackground=COLORS["border"])
        dir_entry.pack(side="left", padx=(0, 4))
        tk.Button(toolbar, text="Browse…", command=self._browse_lib,
                  bg=COLORS["border"], fg=COLORS["text"],
                  activebackground=_lighten(COLORS["border"]),
                  activeforeground=COLORS["text"], relief="flat",
                  cursor="hand2", font=FONT_SMALL, padx=8, pady=4, bd=0).pack(side="left", padx=(0, 16))

        # Search
        tk.Label(toolbar, text="Search:", bg=COLORS["bg_card"],
                 fg=COLORS["text_muted"], font=FONT_SMALL).pack(side="left", padx=(0, 4))
        self._search_var.trace("w", self._on_search)
        search_entry = tk.Entry(toolbar, textvariable=self._search_var,
                                bg=COLORS["bg_input"], fg=COLORS["text"],
                                insertbackground=COLORS["text"], relief="flat",
                                font=FONT_MONO, width=26,
                                highlightthickness=1, highlightbackground=COLORS["border"])
        search_entry.pack(side="left", padx=(0, 8))

        self._count_lbl = tk.Label(toolbar, text="0 symbols", bg=COLORS["bg_card"],
                                    fg=COLORS["text_muted"], font=FONT_SMALL)
        self._count_lbl.pack(side="right")

        # Sort dropdown
        tk.Label(toolbar, text="Sort:", bg=COLORS["bg_card"],
                 fg=COLORS["text_muted"], font=FONT_SMALL).pack(side="right", padx=(0, 4))
        self._sort_var = tk.StringVar(value="Reference")
        sort_combo = ttk.Combobox(toolbar, textvariable=self._sort_var, width=12,
                                   values=["Reference", "Title"],
                                   state="readonly", font=FONT_SMALL)
        sort_combo.pack(side="right", padx=(0, 8))
        sort_combo.bind("<<ComboboxSelected>>", lambda _: self._refresh_list())

        # Main split
        paned = tk.PanedWindow(self, orient="horizontal", bg=COLORS["bg"],
                               sashwidth=6, sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=6, pady=6)

        left  = tk.Frame(paned, bg=COLORS["bg"])
        right = tk.Frame(paned, bg=COLORS["bg"])
        paned.add(left,  minsize=340)
        paned.add(right, minsize=400)

        self._build_list_panel(left)
        self._build_detail_panel(right)

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Lib.Treeview",
                         background=COLORS["bg_card"],
                         foreground=COLORS["text"],
                         fieldbackground=COLORS["bg_card"],
                         rowheight=28,
                         font=FONT_BODY)
        style.configure("Lib.Treeview.Heading",
                         background="#2D1B69",
                         foreground="white",
                         font=FONT_SMALL, relief="flat")
        style.map("Lib.Treeview", background=[("selected", COLORS["accent"])])

    def _build_list_panel(self, parent):
        """Left panel — symbol list."""
        list_frame = tk.Frame(parent, bg=COLORS["bg"])
        list_frame.pack(fill="both", expand=True)

        cols = ("ref", "title")
        self.sym_list = ttk.Treeview(list_frame, columns=cols, show="headings",
                                      selectmode="browse", style="Lib.Treeview")
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self.sym_list.yview)
        self.sym_list.configure(yscrollcommand=vsb.set)
        self.sym_list.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        self.sym_list.heading("ref",   text="Reference", anchor="w")
        self.sym_list.heading("title", text="Title", anchor="w")
        self.sym_list.column("ref",   width=80,  minwidth=60, anchor="w")
        self.sym_list.column("title", width=240, minwidth=80, anchor="w")

        self.sym_list.tag_configure("odd",  background="#1A2744")
        self.sym_list.tag_configure("even", background=COLORS["bg_card"])

        self.sym_list.bind("<<TreeviewSelect>>", self._on_select)

    def _build_detail_panel(self, parent):
        """Right panel — symbol image + full metadata."""
        # Image canvas
        img_frame = tk.Frame(parent, bg=COLORS["bg_card"], padx=2, pady=2)
        img_frame.pack(fill="x", pady=(0, 6))

        self._img_canvas = tk.Canvas(img_frame, width=256, height=256,
                                      bg="white", highlightthickness=0)
        self._img_canvas.pack()

        # Metadata scroll area
        meta_frame = scrolledtext.ScrolledText(
            parent, bg=COLORS["log_bg"], fg=COLORS["text"],
            font=FONT_MONO, relief="flat", wrap="word",
            state="disabled",
            highlightthickness=1, highlightbackground=COLORS["border"])
        meta_frame.pack(fill="both", expand=True)
        meta_frame.tag_config("head", foreground=COLORS["accent"],   font=("Segoe UI", 9, "bold"))
        meta_frame.tag_config("ref",  foreground=COLORS["accent3"],  font=("Segoe UI", 10, "bold"))
        meta_frame.tag_config("val",  foreground=COLORS["text"])
        meta_frame.tag_config("muted", foreground=COLORS["text_muted"])
        self._meta_text = meta_frame

        # No selection placeholder
        self._img_canvas.create_text(
            128, 128, text="Select a symbol\nto preview",
            fill=COLORS["text_muted"], font=FONT_BODY, justify="center")

    # ── Load library ──────────────────────────────────────────────────────

    def _refocus(self):
        """Re-raise this window after a file dialog closes on Windows."""
        self.lift()
        self.focus_force()
        self.attributes("-topmost", True)
        self.after(150, lambda: self.attributes("-topmost", False))

    def _browse_lib(self):
        d = filedialog.askdirectory(parent=self, title="Select Symbol Library Folder (containing iso_symbols.json)")
        if d:
            self._lib_dir.set(d)
            self._load_library(d)
        self._refocus()
    def _load_library(self, folder: str):
        json_path = os.path.join(folder, "iso_symbols.json")
        if not os.path.exists(json_path):
            messagebox.showwarning(
                "Not Found",
                f"iso_symbols.json not found in:\n{folder}\n\n"
                "Run the ISO Symbol Extractor first.",
                parent=self)
            return
        try:
            with open(json_path, encoding="utf-8") as f:
                self._symbols = json.load(f)
            self._img_cache.clear()
            self._filtered = list(self._symbols)
            self._refresh_list()
        except Exception as e:
            messagebox.showerror("Load Error", str(e), parent=self)

    def _refresh_list(self):
        for item in self.sym_list.get_children():
            self.sym_list.delete(item)

        data = list(self._filtered)
        sort_key = self._sort_var.get()
        if sort_key == "Title":
            data.sort(key=lambda d: d["title"].lower())
        else:
            data.sort(key=lambda d: d["reference"])

        for i, d in enumerate(data):
            tag = "odd" if i % 2 else "even"
            self.sym_list.insert("", "end", iid=d["reference"], tag=tag,
                                  values=(d["reference"], d["title"]))
        self._count_lbl.config(text=f"{len(data)} symbols")

    def _on_search(self, *_):
        q = self._search_var.get().lower().strip()
        if not q:
            self._filtered = list(self._symbols)
        else:
            self._filtered = [
                d for d in self._symbols
                if (q in d["reference"].lower() or
                    q in d["title"].lower() or
                    q in d["description"].lower() or
                    q in d["requirements"].lower())
            ]
        self._refresh_list()

    # ── Selection / Detail ────────────────────────────────────────────────
    def _on_select(self, _event):
        sel = self.sym_list.selection()
        if not sel:
            return
        ref = sel[0]
        sym = next((d for d in self._symbols if d["reference"] == ref), None)
        if not sym:
            return
        self._show_detail(sym)

    def _show_detail(self, sym: dict):
        # Load image
        self._img_canvas.delete("all")
        img_path = os.path.join(self._lib_dir.get(), sym["image"])
        if os.path.exists(img_path):
            try:
                from PIL import Image, ImageTk
                pil_img = Image.open(img_path).convert("RGB").resize((256, 256))
                photo   = ImageTk.PhotoImage(pil_img)
                self._current_img = photo
                self._img_canvas.create_image(128, 128, image=photo)
            except ImportError:
                # Pillow not installed — show text fallback
                self._img_canvas.create_text(
                    128, 128,
                    text=f"[{sym['reference']}]\n{sym['title']}\n\n(Install Pillow\nfor image preview)",
                    fill=COLORS["text_muted"], font=FONT_SMALL, justify="center")
            except Exception:
                self._img_canvas.create_text(
                    128, 128, text="Image not\navailable",
                    fill=COLORS["danger"], font=FONT_SMALL, justify="center")
        else:
            self._img_canvas.create_text(
                128, 128, text="Image file\nnot found",
                fill=COLORS["text_muted"], font=FONT_SMALL, justify="center")

        # Populate metadata
        meta = self._meta_text
        meta.configure(state="normal")
        meta.delete("1.0", "end")

        def row(label, value, tag="val"):
            meta.insert("end", f"{label}\n", "head")
            meta.insert("end", f"{value or '—'}\n\n", tag)

        meta.insert("end", f"{sym['reference']}  —  {sym['title']}\n\n", "ref")

        row("ISO Number",    sym.get("iso_number", ""))
        row("Description",   sym.get("description", ""))
        row("Requirements",  sym.get("requirements", ""))
        row("Notes",         sym.get("notes", ""))
        row("Restrictions",  sym.get("restrictions", ""))
        row("Image File",    sym.get("image", ""), "muted")

        meta.configure(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    win = LibraryWindow(root)
    win.mainloop()