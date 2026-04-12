"""
gui_config.py — INI configuration loader for the Label Verification GUI.

Extracted from main_gui.py so config parsing is testable and separate
from UI concerns.
"""

import configparser
import logging
import os

from label_verifier.config import get_default_icon_dir

logger = logging.getLogger(__name__)

DEFAULT_ICON_DIR = get_default_icon_dir()

# Resolved path to the INI file (project_root/configs/default_config.ini)
_HERE        = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
CONFIG_FILE  = os.path.join(_PROJECT_ROOT, 'configs', 'default_config.ini')
OUTPUT_PDF   = os.path.join(_PROJECT_ROOT, 'output', 'report.pdf')


def load_config() -> dict:
    """Load and return the application configuration dict.

    Reads from CONFIG_FILE if it exists, otherwise returns safe defaults.
    All keys are normalised and legacy compatibility aliases are added.

    Returns:
        dict of configuration values ready for use by Controller and GUI.
    """
    project_root      = _PROJECT_ROOT
    default_debug_dir = os.path.join(project_root, 'output', 'converted_images')

    # ── Hardcoded defaults ─────────────────────────────────────────────────
    config: dict = {
        'dpi':                300,
        'detector_threshold': 0.70,
        'clahe':              True,
        'denoise':            False,
        'deskew':             False,
        'binarize':           False,
        'icon_dir':           DEFAULT_ICON_DIR,
        'store_debug_images': False,
        'debug_image_dir':    default_debug_dir,
        # Hough-style pose clustering
        'hough_bin_x':        16.0,
        'hough_bin_y':        16.0,
        'hough_bin_scale':    0.2,
        'hough_bin_angle':    15.0,
        'hough_min_votes':    3,
    }

    if not os.path.exists(CONFIG_FILE):
        logger.debug('Config file not found, using defaults: %s', CONFIG_FILE)
        return config

    try:
        parser = configparser.ConfigParser()
        parser.read(CONFIG_FILE)
        d = parser['DEFAULT']

        # ── DEFAULT section ────────────────────────────────────────────────
        config.update({
            'dpi':                int(d.get('dpi', '300')),
            'detector_threshold': float(d.get('detector_threshold', '0.70')),
            'store_debug_images': d.getboolean('store_debug_images', False),
            'debug_image_dir':    d.get('debug_image_dir', default_debug_dir),
        })

        # ── PREPROCESSING section ──────────────────────────────────────────
        if parser.has_section('PREPROCESSING'):
            p = parser['PREPROCESSING']
            config.update({
                'enable_clahe':            p.getboolean('enable_clahe', True),
                'clahe_clip_limit':        float(p.get('clahe_clip_limit', '2.0')),
                'clahe_tile_grid_size':    int(p.get('clahe_tile_grid_size', '8')),
                'enable_denoise':          p.getboolean('enable_denoise', True),
                'denoise_h':               int(p.get('denoise_h', '8')),
                'denoise_template_window': int(p.get('denoise_template_window', '7')),
                'denoise_search_window':   int(p.get('denoise_search_window', '21')),
                'enable_sharpening':       p.getboolean('enable_sharpening', True),
                'sharpening_method':       p.get('sharpening_method', 'unsharp_mask'),
                'unsharp_radius':          float(p.get('unsharp_radius', '1.0')),
                'unsharp_amount':          float(p.get('unsharp_amount', '0.5')),
                'unsharp_threshold':       int(p.get('unsharp_threshold', '0')),
                'laplacian_kernel_size':   int(p.get('laplacian_kernel_size', '3')),
                'enable_median_blur':      p.getboolean('enable_median_blur', True),
                'median_kernel_size':      int(p.get('median_kernel_size', '3')),
                'target_max_side':         int(p.get('target_max_side', '1200')),
                'store_preprocessor_single': p.getboolean('store_preprocessor_single', True),
            })
        else:
            # Backward compatibility — read from DEFAULT section
            config.update({
                'enable_clahe':            d.getboolean('clahe', True),
                'clahe_clip_limit':        float(d.get('clahe_clip', '2.0')),
                'enable_denoise':          d.getboolean('denoise', False),
                'denoise_h':               int(d.get('denoise_h', '8')),
                'enable_median_blur':      d.getboolean('enable_median_blur', True),
                'target_max_side':         int(d.get('target_max_side', '1200')),
                'store_preprocessor_single': d.getboolean('store_preprocessor_single', True),
            })

        # ── DBSCAN section ─────────────────────────────────────────────────
        if parser.has_section('DBSCAN'):
            dbs = parser['DBSCAN']
            config.update({
                'dbscan_eps':               float(dbs.get('dbscan_eps', '5.0')),
                'dbscan_min_samples':       int(dbs.get('dbscan_min_samples', '8')),
                'dbscan_min_area':          int(dbs.get('dbscan_min_area', '16')),
                'closing_ksize':            int(dbs.get('closing_ksize', '17')),
                'sift_candidate_clustering': dbs.getboolean('sift_candidate_clustering', False),
                'sift_ratio':               float(dbs.get('sift_ratio', '0.75')),
                'sift_min_matches':         int(dbs.get('sift_min_matches', '4')),
                'sift_point_radius':        int(dbs.get('sift_point_radius', '6')),
                'sift_debug_draw':          dbs.getboolean('sift_debug_draw', True),
            })

        # ── EMBEDDINGS section ─────────────────────────────────────────────
        if parser.has_section('EMBEDDINGS'):
            emb = parser['EMBEDDINGS']
            config.update({
                'embedding_sim_threshold': float(emb.get('embedding_sim_threshold', '0.0')),
                'use_mobilenet_embedder':  emb.getboolean('use_mobilenet_embedder', True),
                'siglip_model_name':       emb.get('siglip_model_name',
                                                    'google/siglip-base-patch16-224'),
                'siglip_device':           emb.get('siglip_device', 'cpu'),
                'sim_weight':              float(emb.get('sim_weight', '0.5')),
                'siglip_weight':           float(emb.get('siglip_weight', '0.5')),
            })

        # ── MATCHING section ───────────────────────────────────────────────
        if parser.has_section('MATCHING'):
            m = parser['MATCHING']
            config.update({
                'hough_bin_x':     float(m.get('hough_bin_x',    '16.0')),
                'hough_bin_y':     float(m.get('hough_bin_y',    '16.0')),
                'hough_bin_scale': float(m.get('hough_bin_scale', '0.2')),
                'hough_bin_angle': float(m.get('hough_bin_angle', '15.0')),
                'hough_min_votes': int(m.get('hough_min_votes',   '3')),
            })

        # ── VALIDATION section ─────────────────────────────────────────────
        if parser.has_section('VALIDATION'):
            v = parser['VALIDATION']
            config.update({
                'match_count_weight':  float(v.get('match_count_weight',  '0.4')),
                'inlier_ratio_weight': float(v.get('inlier_ratio_weight', '0.6')),
                'min_bbox_area':       int(v.get('min_bbox_area',         '16')),
            })

        # ── Legacy compatibility aliases ───────────────────────────────────
        config.update({
            'clahe':               config.get('enable_clahe',       True),
            'clahe_clip':          config.get('clahe_clip_limit',   2.0),
            'denoise':             config.get('enable_denoise',     True),
            'sift_median':         config.get('enable_median_blur', True),
            'sift_target_max_side': config.get('target_max_side',  1200),
            'deskew':              d.getboolean('deskew',   False),
            'binarize':            d.getboolean('binarize', False),
        })

    except Exception:
        logger.exception('Failed to parse config file: %s — using defaults', CONFIG_FILE)

    # ── Normalise debug_image_dir to absolute path ─────────────────────────
    try:
        dbg = config.get('debug_image_dir', '')
        if dbg and not os.path.isabs(dbg):
            config['debug_image_dir'] = os.path.abspath(
                os.path.join(project_root, dbg)
            )
    except Exception:
        logger.debug('Failed to normalise debug_image_dir', exc_info=True)

    return config