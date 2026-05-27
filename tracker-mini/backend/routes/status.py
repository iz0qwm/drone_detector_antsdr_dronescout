from flask import Blueprint, jsonify
from services.system import get_system_status

status_bp = Blueprint("status", __name__)

@status_bp.route("/api/status")
def status():
    return jsonify(get_system_status())