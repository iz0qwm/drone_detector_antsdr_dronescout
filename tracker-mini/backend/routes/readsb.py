from flask import Blueprint
from flask import jsonify
from flask import request

from services.readsb import (
    is_enabled,
    set_enabled
)

from services.logger import log


readsb_bp = Blueprint(
    "readsb",
    __name__
)


@readsb_bp.route(
    "/api/readsb/status"
)
def readsb_status():

    enabled = is_enabled()

    return jsonify({
        "success": True,
        "enabled": enabled
    })


@readsb_bp.route(
    "/api/readsb/enable",
    methods=["POST"]
)
def readsb_enable():

    data = request.get_json(
        silent=True
    ) or {}

    enabled = bool(
        data.get(
            "enabled",
            False
        )
    )

    state = set_enabled(
        enabled
    )

    log(
        "READSB",
        f"Enable request: {enabled}"
    )

    return jsonify({
        "success": True,
        "enabled": state
    })