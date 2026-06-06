from flask import Blueprint, jsonify

from services.hardware import (
    get_hardware_status
)

hardware_bp = Blueprint(
    "hardware",
    __name__
)


@hardware_bp.route(
    "/api/hardware"
)
def hardware():

    return jsonify(
        get_hardware_status()
    )