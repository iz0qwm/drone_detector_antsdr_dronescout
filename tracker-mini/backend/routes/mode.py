from flask import Blueprint, jsonify
from services.mode_manager import (
    set_client_mode,
    set_field_mode,
    get_mode_status
)

mode_bp = Blueprint("mode", __name__)


@mode_bp.route("/api/mode/client", methods=["POST"])
def mode_client():
    return jsonify(set_client_mode())


@mode_bp.route("/api/mode/field", methods=["POST"])
def mode_field():
    return jsonify(set_field_mode())


@mode_bp.route("/api/mode/status")
def mode_status():
    return jsonify(get_mode_status())