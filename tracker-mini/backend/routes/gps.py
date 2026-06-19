from flask import Blueprint, jsonify
from services.gps import get_gps_status

gps_bp = Blueprint("gps", __name__)


@gps_bp.route("/status")
def gps_status():
    return jsonify(get_gps_status())