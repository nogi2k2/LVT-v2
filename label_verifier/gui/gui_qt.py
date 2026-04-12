import sys
import os
import threading
from datetime import datetime
from PySide6 import QtWidgets, QtGui, QtCore
from .. import controller as _controller_mod
from .gui_shared import (
    list_icon_paths, format_result_row, open_report,
    find_latest_report, progress_percent, bgr_array_to_pil,
)

from label_verifier.config import get_default_icon_dir
DEFAULT_ICON_DIR = get_default_icon_dir()
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'configs', 'default_config.ini')
OUTPUT_PDF = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output', 'report.pdf')


class Worker(QtCore.QObject):
    progress = QtCore.Signal(dict)
    finished = QtCore.Signal(object, object)

    def __init__(self, ctrl, input_files, icon_paths, output_pdf):
        super().__init__()
        self.ctrl = ctrl
        self.input_files = input_files
        self.icon_paths = icon_paths
        self.output_pdf = output_pdf

    @QtCore.Slot()
    def run(self):
        try:
            results, summary = self.ctrl.run(self.input_files, self.icon_paths, self.output_pdf)
            self.finished.emit(results, summary)
        except Exception as e:
            self.finished.emit(None, e)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Label Verification (Qt)')
        self.resize(1200, 900)

        self.central = QtWidgets.QWidget()
        self.setCentralWidget(self.central)
        self.layout = QtWidgets.QVBoxLayout(self.central)

        # Section 1: Test Files and Icons (similar to Tk layout)
        self._build_input_section()

        # Section 2: Controls (Start / Cancel / Progress)
        self._build_control_section()

        # Section 3: Results
        self._build_results_section()

        # Footer
        self._build_footer()

        # State
        self.input_files = []
        self.icon_list_paths = []
        self.ctrl = None

    def _build_input_section(self):
        # Container
        section = QtWidgets.QGroupBox('')
        v = QtWidgets.QVBoxLayout(section)

        # Top row: select files and file count
        h = QtWidgets.QHBoxLayout()
        self.btn_select = QtWidgets.QPushButton('Select Test Labels')
        self.btn_select.clicked.connect(self.pick_files)
        h.addWidget(self.btn_select)
        self.file_count = QtWidgets.QLabel('No files selected')
        h.addWidget(self.file_count)
        v.addLayout(h)

        # Icon library path and browse
        path_h = QtWidgets.QHBoxLayout()
        path_h.addWidget(QtWidgets.QLabel('Icon Library:'))
        self.icon_dir_edit = QtWidgets.QLineEdit(DEFAULT_ICON_DIR)
        path_h.addWidget(self.icon_dir_edit)
        b = QtWidgets.QPushButton('Browse')
        b.clicked.connect(self.change_icon_dir)
        path_h.addWidget(b)
        v.addLayout(path_h)

        # Icon list + preview
        icons_h = QtWidgets.QHBoxLayout()

        # Icon list (scrollable)
        self.icon_list = QtWidgets.QListWidget()
        self.icon_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        # react to both changes in check state and selection changes so preview
        # updates whenever a user checks an icon or selects it
        self.icon_list.itemChanged.connect(self._on_icon_changed)
        self.icon_list.itemSelectionChanged.connect(self._on_icon_selection_changed)
        icons_h.addWidget(self.icon_list, 1)

        # Preview pane
        preview_v = QtWidgets.QVBoxLayout()
        preview_v.addWidget(QtWidgets.QLabel('Preview:'))
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setFixedSize(300, 300)
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        preview_v.addWidget(self.preview_label)
        icons_h.addLayout(preview_v, 1)

        v.addLayout(icons_h)

        self.layout.addWidget(section)

        # Populate icons initially
        self._populate_icons()

    def _build_control_section(self):
        section = QtWidgets.QGroupBox('')
        h = QtWidgets.QHBoxLayout(section)

        self.btn_start = QtWidgets.QPushButton('Start Verification')
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_verification)
        h.addWidget(self.btn_start)

        self.btn_cancel = QtWidgets.QPushButton('Cancel')
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        h.addWidget(self.btn_cancel)

        # Progress and status
        progress_v = QtWidgets.QVBoxLayout()
        progress_v.addWidget(QtWidgets.QLabel('Progress:'))
        self.progress_bar = QtWidgets.QProgressBar()
        progress_v.addWidget(self.progress_bar)
        self.status_label = QtWidgets.QLabel('Ready')
        progress_v.addWidget(self.status_label)
        h.addLayout(progress_v, 1)

        self.layout.addWidget(section)

    def _build_results_section(self):
        section = QtWidgets.QGroupBox('')
        v = QtWidgets.QVBoxLayout(section)

        self.table = QtWidgets.QTreeWidget()
        self.table.setColumnCount(4)
        self.table.setHeaderLabels(['Input', 'Icon', 'Decision', 'Score'])
        self.table.setIconSize(QtCore.QSize(64, 64))
        v.addWidget(self.table)

        self.layout.addWidget(section)

    def _build_footer(self):
        footer = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(footer)
        self.footer_status = QtWidgets.QLabel('Ready to process files...')
        h.addWidget(self.footer_status)
        h.addStretch(1)
        self.btn_open = QtWidgets.QPushButton('📄 Open PDF Report')
        self.btn_open.setEnabled(False)
        self.btn_open.clicked.connect(self.open_report)
        h.addWidget(self.btn_open)
        self.layout.addWidget(footer)

    # --- Helper methods ---
    def _populate_icons(self):
        self.icon_list.clear()
        icon_dir = self.icon_dir_edit.text()
        self.icon_list_paths = []
        if not os.path.exists(icon_dir):
            self.icon_list.addItem('Path not found: ' + icon_dir)
            return
        try:
            entries = os.listdir(icon_dir)
            icon_entries = []
            for e in sorted(entries):
                full = os.path.join(icon_dir, e)
                if os.path.isdir(full) and os.path.exists(os.path.join(full, 'icon.png')):
                    icon_entries.append(os.path.join(full, 'icon.png'))
            if not icon_entries:
                for e in sorted(entries):
                    full = os.path.join(icon_dir, e)
                    if os.path.isfile(full) and e.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        icon_entries.append(full)
            for p in icon_entries:
                    it = QtWidgets.QListWidgetItem(os.path.basename(p))
                    # keep the real path on the item for reliable lookup
                    it.setData(QtCore.Qt.UserRole, p)
                    it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable)
                    it.setCheckState(QtCore.Qt.Unchecked)
                    self.icon_list.addItem(it)
                    self.icon_list_paths.append(p)
        except Exception:
            self.icon_list.addItem('Error reading directory')

    def _on_icon_changed(self, item):
        """Handle item check-state changes to update preview and start button state."""
        try:
            path = item.data(QtCore.Qt.UserRole)
            # If item was checked, show its preview; if unchecked and no other
            # items are checked, clear preview.
            if item.checkState() == QtCore.Qt.Checked:
                self._show_preview(path)
            else:
                # find any other checked item and preview the first one
                for i in range(self.icon_list.count()):
                    it = self.icon_list.item(i)
                    if it.checkState() == QtCore.Qt.Checked:
                        p = it.data(QtCore.Qt.UserRole)
                        self._show_preview(p)
                        break
                else:
                    # nothing checked -> clear
                    self.preview_label.clear()
        except Exception:
            # safe fallback
            try:
                self.preview_label.setText('Preview not available')
            except Exception:
                pass
        finally:
            # keep start button enabled/disabled according to selection
            try:
                self._update_start_button_state()
            except Exception:
                pass

    def _on_icon_selection_changed(self):
        """When selection changes (user clicks item text), preview that item."""
        try:
            items = self.icon_list.selectedItems()
            if not items:
                return
            item = items[0]
            path = item.data(QtCore.Qt.UserRole)
            if path:
                self._show_preview(path)
        except Exception:
            pass

    def _show_preview(self, path):
        """Robustly load an image from path and display it in the preview_label."""
        try:
            # Try QPixmap first (fast, no PIL dependency)
            pix = QtGui.QPixmap(path)
            if pix and not pix.isNull():
                scaled = pix.scaled(300, 300, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                self.preview_label.setPixmap(scaled)
                return
        except Exception:
            pass

        # Fallback to PIL (handles more formats and conversions)
        try:
            from PIL import Image, ImageQt
            pil = Image.open(path)
            pil.thumbnail((300, 300), Image.LANCZOS)
            qimg = ImageQt.ImageQt(pil)
            pix = QtGui.QPixmap.fromImage(qimg)
            scaled = pix.scaled(300, 300, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.preview_label.setPixmap(scaled)
            return
        except Exception:
            pass

        # Last resort: textual fallback
        try:
            self.preview_label.setText(os.path.basename(path))
        except Exception:
            self.preview_label.setText('Preview not available')

    def change_icon_dir(self):
        dlg = QtWidgets.QFileDialog(self)
        new = dlg.getExistingDirectory(self, 'Select Icon Library Folder', self.icon_dir_edit.text())
        if new:
            self.icon_dir_edit.setText(new)
            self._populate_icons()

    def pick_files(self):
        dlg = QtWidgets.QFileDialog(self)
        files, _ = dlg.getOpenFileNames(self, 'Select Test Files', os.getcwd(), 'PDF and Images (*.pdf *.png *.jpg *.jpeg *.tif *.tiff)')
        if files:
            self.input_files = files
            self.file_count.setText(f"{len(files)} file(s) selected")
            self._update_start_button_state()

    def _update_start_button_state(self):
        has_files = len(getattr(self, 'input_files', [])) > 0
        has_icons = any(self.icon_list.item(i).checkState() == QtCore.Qt.Checked for i in range(self.icon_list.count()))
        self.btn_start.setEnabled(has_files and has_icons)

    def start_verification(self):
        # Collect selected icons
        selected = [self.icon_list_paths[i] for i in range(self.icon_list.count()) if self.icon_list.item(i).checkState() == QtCore.Qt.Checked]
        if not selected:
            QtWidgets.QMessageBox.warning(self, 'No Icons Selected', 'Please select at least one reference icon.')
            return
        # Prepare UI
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText('Initializing verification process...')
        self.table.clear()

        # Create controller
        try:
            self.ctrl = _controller_mod.Controller({}, self.on_progress)
        except Exception:
            self.ctrl = _controller_mod.Controller({}, self.on_progress)

        # Run controller.run in a worker thread
        self.thread = QtCore.QThread()
        self.worker = Worker(self.ctrl, self.input_files, selected, OUTPUT_PDF)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_finished)
        self.worker.progress.connect(self.on_progress)
        self.thread.start()

    def _on_finished(self, results, summary):
        if results is None:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Processing failed: {summary}')
            self.footer_status.setText('Processing failed')
        else:
            self._populate_results(results)
            self.btn_open.setEnabled(True)
            self.footer_status.setText(f'Complete: {sum(1 for r in results if r.decision=="Pass")}/{len(results)} passed')
        try:
            self.thread.quit()
            self.thread.wait(2000)
        except Exception:
            pass
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def _on_cancel(self):
        try:
            if hasattr(self, 'ctrl') and self.ctrl is not None:
                try:
                    self.ctrl.cancel()
                    self.status_label.setText('Cancellation requested...')
                except Exception:
                    pass
        except Exception:
            pass

    def on_progress(self, status):
        total = status.get('total', 1)
        current = status.get('current', 0)
        progress = int(100 * current / total) if total > 0 else 0
        # update UI
        try:
            self.progress_bar.setValue(progress)
            self.status_label.setText(status.get('status', ''))
        except Exception:
            pass

    def _populate_results(self, results):
        self.table.clear()
        for r in results:
            item = QtWidgets.QTreeWidgetItem([os.path.basename(r.input_path), r.icon_name, r.decision, f"{getattr(r, 'score', 0):.3f}"])
            if getattr(r, 'decision', '') == 'Pass':
                item.setBackground(2, QtGui.QBrush(QtGui.QColor('#d5f4e6')))
            else:
                item.setBackground(2, QtGui.QBrush(QtGui.QColor('#f8d7da')))
            # thumbnail
            try:
                if getattr(r, 'match_snip', None) is not None:
                    from PIL import Image, ImageQt
                    arr = r.match_snip
                    import cv2
                    if hasattr(arr, 'shape') and arr.ndim == 3 and arr.shape[2] == 3:
                        img = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(img)
                    else:
                        pil = Image.fromarray(arr)
                    pil.thumbnail((64,64), Image.LANCZOS)
                    qimg = ImageQt.ImageQt(pil)
                    pix = QtGui.QPixmap.fromImage(qimg)
                    icon = QtGui.QIcon(pix)
                    item.setIcon(0, icon)
            except Exception:
                pass
            self.table.addTopLevelItem(item)

    def open_report(self):
        outdir = os.path.dirname(OUTPUT_PDF)
        if os.path.isdir(outdir):
            pdfs = [os.path.join(outdir, f) for f in os.listdir(outdir) if f.lower().endswith('.pdf')]
            if pdfs:
                latest = max(pdfs, key=os.path.getmtime)
                try:
                    os.startfile(latest)
                    return
                except Exception:
                    pass
        if os.path.exists(OUTPUT_PDF):
            try:
                os.startfile(OUTPUT_PDF)
            except Exception:
                pass

    def pick_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Select Test Files', os.getcwd(), 'PDF and Images (*.pdf *.png *.jpg *.jpeg *.tif *.tiff)')
        if files:
            self.input_files = files
            self.file_count.setText(f"{len(files)} file(s) selected")
            self.btn_start.setEnabled(True)

    def start_verification(self):
        # For POC, take all icons from DEFAULT_ICON_DIR
        icon_dir = DEFAULT_ICON_DIR
        icons = []
        if os.path.isdir(icon_dir):
            for f in os.listdir(icon_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    icons.append(os.path.join(icon_dir, f))
        self.icon_paths = icons
        try:
            self.ctrl = _controller_mod.Controller({}, self.on_progress)
        except Exception:
            self.ctrl = _controller_mod.Controller({}, self.on_progress)

        # Run Controller.run in a background thread and emit finished
        self.thread = QtCore.QThread()
        self.worker = Worker(self.ctrl, self.input_files, self.icon_paths, OUTPUT_PDF)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_finished)
        self.worker.progress.connect(self.on_progress)
        self.thread.start()
        self.btn_start.setEnabled(False)

    def _on_finished(self, results, summary):
        if results is None:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Processing failed: {summary}')
        else:
            self._populate_results(results)
            self.btn_open.setEnabled(True)
        # cleanup
        try:
            self.thread.quit()
            self.thread.wait(2000)
        except Exception:
            pass
        self.btn_start.setEnabled(True)

    def on_progress(self, status):
        total = status.get('total', 1)
        current = status.get('current', 0)
        prog = int(100 * current / total) if total > 0 else 0
        self.progress.setValue(prog)

    def _populate_results(self, results):
        self.table.clear()
        for r in results:
            item = QtWidgets.QTreeWidgetItem([os.path.basename(r.input_path), r.icon_name, r.decision, f"{getattr(r, 'score', 0):.3f}"])
            if getattr(r, 'decision', '') == 'Pass':
                item.setBackground(2, QtGui.QBrush(QtGui.QColor('#d5f4e6')))
            else:
                item.setBackground(2, QtGui.QBrush(QtGui.QColor('#f8d7da')))
            # thumbnail
            try:
                if getattr(r, 'match_snip', None) is not None:
                    import numpy as np
                    from PIL import Image
                    arr = r.match_snip
                    if hasattr(arr, 'shape'):
                        import cv2
                        img = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(img)
                    else:
                        pil = Image.fromarray(arr)
                    pil.thumbnail((64,64), Image.LANCZOS)
                    qimg = QtGui.QImage(pil.tobytes(), pil.width, pil.height, QtGui.QImage.Format_RGB888)
                    icon = QtGui.QIcon(QtGui.QPixmap.fromImage(qimg))
                    item.setIcon(0, icon)
            except Exception:
                pass
            self.table.addTopLevelItem(item)

    def open_report(self):
        outdir = os.path.dirname(OUTPUT_PDF)
        if os.path.isdir(outdir):
            pdfs = [os.path.join(outdir, f) for f in os.listdir(outdir) if f.lower().endswith('.pdf')]
            if pdfs:
                latest = max(pdfs, key=os.path.getmtime)
                try:
                    os.startfile(latest)
                    return
                except Exception:
                    pass
        if os.path.exists(OUTPUT_PDF):
            try:
                os.startfile(OUTPUT_PDF)
            except Exception:
                pass


def run_app():
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
