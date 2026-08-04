"""
Proximity API routes.

GET /api/proximity/status  — returns latest snapshot (fast, non-blocking)
GET /api/proximity/config  — returns proximity configuration
POST /api/proximity/config — updates proximity configuration
"""
from flask import Blueprint, jsonify, request
from services.proximity.engine import engine
from services.proximity.config import (
    get_proximity_config,
    update_proximity_config,
    get_traffic_config,
    update_traffic_config,
)

proximity_bp = Blueprint("proximity", __name__)


@proximity_bp.route("/api/proximity/status")
def proximity_status():
    """
    Returns latest completed proximity snapshot.
    Does NOT fetch providers, perform calculations, or block on Internet.
    """
    snapshot = engine.get_snapshot()
    return jsonify(snapshot.to_dict())


@proximity_bp.route("/api/proximity/config")
def proximity_config_get():
    """Returns current proximity configuration (merged with defaults)."""
    return jsonify(get_proximity_config())


@proximity_bp.route("/api/proximity/config", methods=["POST"])
def proximity_config_post():
    """Update proximity configuration."""
    data = request.get_json(silent=True) or {}
    success, error, merged = update_proximity_config(data)
    if not success:
        return jsonify({"success": False, "error": error}), 400
    return jsonify({"success": True, "config": merged})


@proximity_bp.route("/api/settings/traffic")
def traffic_config_get():
    """Returns current traffic configuration (includes adsb_net_enabled)."""
    return jsonify(get_traffic_config())


@proximity_bp.route("/api/settings/traffic", methods=["POST"])
def traffic_config_post():
    """Update traffic configuration (e.g., adsb_net_enabled migration)."""
    data = request.get_json(silent=True) or {}
    success, merged = update_traffic_config(data)
    return jsonify({"success": success, "config": merged})
