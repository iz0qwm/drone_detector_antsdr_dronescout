"""
Proximity configuration with code-defined defaults.

The updater deploys only backend/ and frontend/. Configuration defaults
must be defined here in code. The device's config/settings.json is never
overwritten by updates. Missing sections are handled gracefully.
"""

PROXIMITY_DEFAULTS = {
    "enabled": True,
    "evaluation_radius_m": 10000,
    "thresholds": {
        "monitor_entry_m": 3000,
        "monitor_exit_m": 3300,
        "caution_entry_m": 1500,
        "caution_exit_m": 1800,
        "warning_entry_m": 500,
        "warning_exit_m": 700,
    },
    "aircraft_source_stale_ms": 30000,
    "drone_stale_ms": 15000,
    "target_retention_ms": 60000,
    "pair_stale_grace_ms": 10000,
    "calculation_interval_ms": 5000,
    "adsb_net_refresh_interval_ms": 15000,
    "max_panel_entries": 5,
    "max_rendered_aircraft": 5,
    "movement_deadband_m": 50,
    "movement_history_window_s": 15,
    "source_precedence_tie_window_s": 3,
    "pulse_on_warning": True,
}

TRAFFIC_DEFAULTS = {
    "remoteid_enabled": True,
    "adsb_local_enabled": True,
    "adsb_net_enabled": False,
    "meshtastic_enabled": False,
}


def get_proximity_config():
    """
    Returns merged proximity config: settings.json values override code defaults.
    Missing keys are filled from PROXIMITY_DEFAULTS.
    """
    try:
        from config import SETTINGS
        saved = SETTINGS.get("proximity", {})
    except Exception:
        saved = {}

    merged = {**PROXIMITY_DEFAULTS, **saved}
    # Deep-merge thresholds
    merged["thresholds"] = {
        **PROXIMITY_DEFAULTS["thresholds"],
        **saved.get("thresholds", {}),
    }
    return merged


def get_traffic_config():
    """
    Returns merged traffic config with authoritative adsb_net_enabled.
    """
    try:
        from config import SETTINGS
        saved = SETTINGS.get("traffic", {})
    except Exception:
        saved = {}

    return {**TRAFFIC_DEFAULTS, **saved}


def is_adsb_net_enabled():
    """Check if ADSBNet is enabled in the authoritative backend setting."""
    return get_traffic_config().get("adsb_net_enabled", False)


def update_proximity_config(data):
    """
    Validate and persist proximity configuration changes.
    Returns (success: bool, error: str|None, merged: dict).
    """
    from config import SETTINGS, save_settings

    # Validate thresholds if provided
    thresholds = data.get("thresholds", {})
    for level in ("monitor", "caution", "warning"):
        entry_key = f"{level}_entry_m"
        exit_key = f"{level}_exit_m"
        entry = thresholds.get(entry_key)
        exit_val = thresholds.get(exit_key)

        if entry is not None and exit_val is not None:
            if entry >= exit_val:
                return False, f"{entry_key} must be less than {exit_key}", None

    # Validate positive numeric values
    for key in ("evaluation_radius_m", "calculation_interval_ms",
                "aircraft_source_stale_ms", "drone_stale_ms"):
        val = data.get(key)
        if val is not None and (not isinstance(val, (int, float)) or val <= 0):
            return False, f"{key} must be a positive number", None

    # Merge into existing
    current = SETTINGS.get("proximity", {})
    current_thresholds = current.get("thresholds", {})

    if thresholds:
        current_thresholds.update(thresholds)

    updated = {**current}
    for k, v in data.items():
        if k == "thresholds":
            continue
        updated[k] = v

    updated["thresholds"] = current_thresholds
    SETTINGS["proximity"] = updated
    save_settings()

    return True, None, get_proximity_config()


def update_traffic_config(data):
    """
    Update traffic configuration (including adsb_net_enabled).
    Returns (success: bool, merged: dict).
    """
    from config import SETTINGS, save_settings

    current = SETTINGS.get("traffic", {})
    current.update(data)
    SETTINGS["traffic"] = current
    save_settings()

    return True, get_traffic_config()
