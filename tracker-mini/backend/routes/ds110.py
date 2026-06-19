from flask import Blueprint, jsonify, request

import services.ds110 as ds110
from services.logger import log

ds110_bp = Blueprint(
    "ds110",
    __name__
)


@ds110_bp.route("/api/ds110/status")
def ds110_status():

    return jsonify({
        "enabled": ds110.running
    })


@ds110_bp.route(
    "/api/ds110/enable",
    methods=["POST"]
)
def enable_ds110():
    data = request.json or {}
    enabled = data.get(
        "enabled",
        True
    )
    if enabled:
        log(
            "DS110",
            "Remote ID receiver enabled"
        )
        ds110.start()
    else:
        log(
            "DS110",
            "Remote ID receiver disabled"
        )
        ds110.stop()
        ds110.clear_aircraft()

    return jsonify({
        "success": True,
        "enabled": enabled
    })