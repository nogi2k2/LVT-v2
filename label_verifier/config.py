"""
Centralised configuration helpers for label_verifier.

Provides runtime-resolved paths so no developer-specific absolute paths
are hardcoded in GUI or app files.
"""

import os


def get_default_icon_dir() -> str:
    """Return the default icon library directory.

    Resolution order:
      1. Environment variable  LV_ICON_DIR  (set this on any machine)
      2. 'Icon Library' folder relative to this package  (works when
         the project is checked out normally)
      3. Empty string — GUI will show an empty field, user can browse.
    """
    # 1. Environment variable — cleanest for CI or other machines
    env = os.environ.get('LV_ICON_DIR', '').strip()
    if env and os.path.isdir(env):
        return env

    # 2. Relative path: label_verifier/../Icon Library
    here = os.path.dirname(os.path.abspath(__file__))
    rel = os.path.normpath(os.path.join(here, '..', 'Icon Library'))
    if os.path.isdir(rel):
        return rel

    # 3. Safe fallback
    return ''


def get_output_dir() -> str:
    """Return the default output directory (project root / output)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, '..', 'output'))