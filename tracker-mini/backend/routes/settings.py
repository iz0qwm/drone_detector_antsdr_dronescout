from flask import Blueprint, jsonify
from flask import Blueprint, jsonify, request
from config import SETTINGS, save_settings
import glob
import os

settings_bp = Blueprint("settings", __name__)

@settings_bp.route("/api/settings")
def settings():

    return jsonify({
        "map": SETTINGS["map"],
        "ap_ssid": SETTINGS["ap_ssid"]
    })

@settings_bp.route(
    "/api/ds110/settings"
)
def get_ds110_settings():

    return jsonify(
        SETTINGS.get(
            "ds110",
            {
                "interface": "usb",
                "device": "/dev/ttyACM0",
                "baudrate": 115200
            }
        )
    )


@settings_bp.route(
    "/api/ds110/settings",
    methods=["POST"]
)
def save_ds110_settings():

    data = request.json

    SETTINGS["ds110"] = {

        "interface":
            data.get(
                "interface",
                "usb"
            ),

        "device":
            data.get(
                "device",
                "/dev/ttyACM0"
            ),

        "baudrate":
            int(
                data.get(
                    "baudrate",
                    115200
                )
            )
    }

    save_settings()

    return jsonify({
        "success": True
    })
    
@settings_bp.route(
    "/api/serial/ports"
)
def get_serial_ports():

    ports = set()

    patterns = [
        "/dev/serial*",
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
        "/dev/ttyAMA*",
        "/dev/ttyS*"
    ]

    for pattern in patterns:

        for device in glob.glob(pattern):

            if os.path.exists(device):
                ports.add(device)

    current_device = (
        SETTINGS
        .get("ds110", {})
        .get("device")
    )

    if current_device:
        ports.add(current_device)

    return jsonify({

        "current": current_device,

        "ports": sorted(
            list(ports)
        )

    })