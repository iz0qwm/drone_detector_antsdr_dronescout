from flask import Blueprint, jsonify
from config import SETTINGS

settings_bp = Blueprint("settings", __name__)

@settings_bp.route("/api/settings")
def settings():

    return jsonify({
        "map": SETTINGS["map"],
        "ap_ssid": SETTINGS["ap_ssid"]
    })