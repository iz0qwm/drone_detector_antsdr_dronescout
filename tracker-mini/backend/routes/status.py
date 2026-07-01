from flask import Blueprint, jsonify, request
from services.system import (
    get_system_status,
    restart_tracker,
    reboot_system,
    shutdown_system
)

status_bp = Blueprint("status", __name__)

@status_bp.route("/api/status")
def status():
    return jsonify(get_system_status())


@status_bp.route(
    "/api/system/restart",
    methods=["POST"]
)
def restart():

    restart_tracker()

    return jsonify({
        "success": True
    })


@status_bp.route(
    "/api/system/reboot",
    methods=["POST"]
)
def reboot():

    reboot_system()

    return jsonify({
        "success": True
    })


@status_bp.route(
    "/api/system/shutdown",
    methods=["POST"]
)
def shutdown():

    shutdown_system()

    return jsonify({
        "success": True
    })