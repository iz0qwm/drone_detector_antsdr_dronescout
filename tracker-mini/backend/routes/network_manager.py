from flask import Blueprint, jsonify
from services.network_manager import list_connections, scan_wifi
from flask import Blueprint, jsonify, request

network_manager_bp = Blueprint("network_manager", __name__)

from services.network_manager import (
    list_connections,
    scan_wifi,
    connect_wifi,
    disconnect_wifi,
    start_hotspot,
    stop_hotspot,
    hotspot_status
)

@network_manager_bp.route("/api/connections")
def connections():
    return jsonify(list_connections())


@network_manager_bp.route("/api/wifi-scan")
def wifi_scan():
    return jsonify(scan_wifi())

@network_manager_bp.route("/api/wifi/connect", methods=["POST"])
def wifi_connect():
    data = request.get_json()

    ssid = data.get("ssid")
    password = data.get("password", "")

    if not ssid:
        return jsonify({
            "success": False,
            "message": "SSID missing"
        }), 400

    return jsonify(connect_wifi(ssid, password))

@network_manager_bp.route("/api/wifi/disconnect", methods=["POST"])
def wifi_disconnect():
    return jsonify(disconnect_wifi())


@network_manager_bp.route("/api/ap/start", methods=["POST"])
def ap_start():
    return jsonify(start_hotspot())


@network_manager_bp.route("/api/ap/stop", methods=["POST"])
def ap_stop():
    return jsonify(stop_hotspot())


@network_manager_bp.route("/api/ap/status")
def ap_status():
    return jsonify(hotspot_status())