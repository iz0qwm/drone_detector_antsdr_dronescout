from flask import Blueprint, jsonify
from services.network import get_network_status

network_bp = Blueprint("network", __name__)

@network_bp.route("/api/network")
def network():
    return jsonify(get_network_status())