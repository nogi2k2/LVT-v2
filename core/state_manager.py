import logging

logger = logging.getLogger(__name__)

APP_STATE = {
    "icon_library_path": "",
    "required_symbols":  [],
    "verify_list":       [],
    "vv_plan_path":      "",
    "vv_plan_data":      None,
    "callbacks": {
        "icon_library_ready": [],
        "symbols_ready":      [],
        "vv_plan_ready":      [],
    },
}

def register_callback(event: str, fn) -> None:
    APP_STATE["callbacks"].setdefault(event, []).append(fn)

def unregister_callback(event: str, fn) -> None:
    callbacks = APP_STATE["callbacks"].get(event, [])
    APP_STATE["callbacks"][event] = [cb for cb in callbacks if cb is not fn]

def fire_event(event: str, *args) -> None:
    for fn in APP_STATE["callbacks"].get(event, []):
        try:
            fn(*args)
        except Exception:
            logger.exception("Callback error for event '%s'", event)