import os
import sys

def get_base_dir() -> str:
    """Returns the base directory, compatible with PyInstaller."""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def get_icon_dir() -> str:
    """Returns the path to the Icon Library."""
    env = os.environ.get('LV_ICON_DIR', '').strip()
    if env and os.path.isdir(env):
        return env
    return os.path.join(get_base_dir(), "Icon Library")

def get_config_path() -> str:
    """Returns the default config path."""
    return os.path.join(get_base_dir(), "configs", "default_config.ini")