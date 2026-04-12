import os
import tempfile
import threading

import streamlit as st

from label_verifier.config import get_default_icon_dir, get_output_dir
from label_verifier.gui.gui_shared import (
    find_latest_report,
    format_result_row,
    list_icon_paths as list_icons,
    progress_percent,
)
from label_verifier.utils import bgr_to_rgb

DEFAULT_ICON_DIR = get_default_icon_dir()
OUTPUT_DIR       = get_output_dir()
OUTPUT_PDF       = os.path.join(OUTPUT_DIR, 'report.pdf')


# ── Helpers ────────────────────────────────────────────────────────────────

def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_uploaded_files(uploaded_files) -> list:
    """Save Streamlit-uploaded files to a temp dir and return their paths."""
    tmp   = tempfile.mkdtemp(prefix='lv_streamlit_')
    saved = []
    for up in uploaded_files:
        dest = os.path.join(tmp, up.name)
        with open(dest, 'wb') as f:
            f.write(up.getbuffer())
        saved.append(dest)
    return saved


def run_controller_thread(ctrl, input_paths, icon_paths, output_pdf) -> None:
    """Run the verification pipeline in a background thread."""
    try:
        results, summary = ctrl.run(input_paths, icon_paths, output_pdf)
        st.session_state['results']     = results
        st.session_state['summary']     = summary
        st.session_state['run_success'] = True
    except Exception as e:
        st.session_state['run_success'] = False
        st.session_state['run_error']   = str(e)


# ── Main app ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title='Label Verification', layout='wide')
    st.title('Label Verification')

    # Initialise session state keys with defaults
    _defaults = {
        'progress_status': {'current': 0, 'total': 1, 'status': 'Ready'},
        'results':         [],
        'summary':         None,
        'ctrl':            None,
        'run_success':     None,
        'run_error':       None,
    }
    for key, default in _defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Input files ────────────────────────────────────────────────────────
    st.header('Test Files')
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded   = st.file_uploader('Upload files (PDF, PNG, JPG)',
                                       accept_multiple_files=True)
        paths_text = st.text_area('Or paste local file paths (one per line)',
                                   height=80)
        input_type = st.selectbox('Input type', ['Label', 'IFU'])
    with col2:
        icon_dir = st.text_input('Icon Library:', value=DEFAULT_ICON_DIR)

    # ── Icon selection ─────────────────────────────────────────────────────
    icons = list_icons(icon_dir)           # uses gui_shared.list_icon_paths
    if not icons:
        st.warning(f'No icons found in: {icon_dir}')

    st.subheader('Reference Icons')
    selected = {}
    left, right = st.columns([1, 3])
    with left:
        for i, p in enumerate(icons):
            selected[p] = st.checkbox(os.path.basename(p), key=f'icon_chk_{i}')
    with right:
        preview_path = next((p for p, v in selected.items() if v), None)
        if preview_path:
            st.image(preview_path, use_column_width=True)
        else:
            st.write('No icon selected')

    # ── Controls ───────────────────────────────────────────────────────────
    st.subheader('Processing Control')
    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        start  = st.button('Start Verification')
    with c2:
        cancel = st.button('Cancel')
    with c3:
        pbar  = st.progress(0)
        ptext = st.empty()

    # ── Handle Start ───────────────────────────────────────────────────────
    if start:
        st.session_state.update({
            'results':         [],
            'run_success':     None,
            'run_error':       None,
            'progress_status': {'current': 0, 'total': 1, 'status': 'Initializing...'},
        })

        input_paths: list = []
        if uploaded:
            input_paths.extend(save_uploaded_files(uploaded))
        if paths_text:
            input_paths.extend(
                ln.strip() for ln in paths_text.splitlines() if ln.strip()
            )

        if not input_paths:
            st.error('No input files supplied (upload or paste local paths).')
        else:
            selected_icons = [p for p, v in selected.items() if v]
            if not selected_icons:
                st.error('Please select at least one icon from the Icon Library.')
            else:
                ctrl = None
                try:
                    if input_type.lower() == 'ifu':
                        from label_verifier.ifu_verification import controller as _ifu
                        ctrl = _ifu.IFUController(
                            {}, st.session_state['progress_status']
                        )
                    else:
                        from label_verifier import controller as _ctrl
                        ctrl = _ctrl.Controller(
                            {}, st.session_state['progress_status']
                        )
                except Exception as e:
                    st.error(f'Failed to create controller: {e}')

                if ctrl:
                    st.session_state['ctrl'] = ctrl
                    ensure_output_dir()
                    threading.Thread(
                        target=run_controller_thread,
                        args=(ctrl, input_paths, selected_icons, OUTPUT_PDF),
                        daemon=True,
                    ).start()

    # ── Handle Cancel ──────────────────────────────────────────────────────
    if cancel:
        ctrl = st.session_state.get('ctrl')
        if ctrl is not None:
            try:
                ctrl.cancel()
                st.session_state['progress_status'] = {
                    'current': 0, 'total': 1, 'status': 'Cancellation requested...'
                }
            except Exception as e:
                st.error(f'Cancel failed: {e}')

    # ── Progress bar ───────────────────────────────────────────────────────
    status = st.session_state.get('progress_status', {})
    pbar.progress(progress_percent(status))          # uses gui_shared.progress_percent
    ptext.text(status.get('status', ''))

    # ── Results table ──────────────────────────────────────────────────────
    st.subheader('Results')
    for r in st.session_state.get('results', []):
        row       = format_result_row(r)             # uses gui_shared.format_result_row
        match_img = getattr(r, 'match_snip', None)
        if match_img is not None:
            try:
                match_img = bgr_to_rgb(match_img)
            except Exception:
                match_img = None

        img_col, txt_col = st.columns([1, 3])
        with img_col:
            if match_img is not None:
                st.image(match_img, width=64)
        with txt_col:
            st.write(f"**Input:** {row['filename']}")
            st.write(f"**Icon:** {row['icon']}")
            st.write(f"**Decision:** {row['decision']}")
            st.write(f"**Score:** {row['score']}")

    # ── Download report ────────────────────────────────────────────────────
    st.markdown('---')
    if st.button('📄 Download Latest Report'):
        latest = find_latest_report(OUTPUT_DIR)      # uses gui_shared.find_latest_report
        if latest:
            with open(latest, 'rb') as fh:
                st.download_button(
                    'Download PDF',
                    fh.read(),
                    file_name=os.path.basename(latest),
                )
        else:
            st.error('No PDF reports found. Run verification first.')


if __name__ == '__main__':
    main()