from flask import Blueprint, jsonify

from services.logger import (
    get_logs,
    clear_logs
)

logs_bp = Blueprint(
    "logs",
    __name__
)


@logs_bp.route(
    "/api/logs",
    methods=["GET"]
)
def api_logs():

    return jsonify(
        get_logs()
    )


@logs_bp.route(
    "/api/logs/clear",
    methods=["POST"]
)
def api_clear_logs():

    clear_logs()

    return jsonify({
        "status": "ok"
    })