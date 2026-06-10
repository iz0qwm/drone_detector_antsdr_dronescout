from flask import Blueprint, jsonify, request
from services.logger import log

from services.air_network import (
    get_bounds_args,
    get_network_aircraft
)

air_network_bp = Blueprint(
    "air_network",
    __name__
)


@air_network_bp.route(
    "/api/air/network"
)
def air_network():

    try:
        bounds = get_bounds_args(
            request
        )

        return jsonify(
            get_network_aircraft(
                bounds
            )
        )

    except Exception as e:

        import traceback

        log(
            "ADSB",
            "AIR NETWORK API ERROR:\n" + traceback.format_exc(),
            level="ERROR"
        )

        return jsonify({
            "success": False,
            "message": str(e),
            "aircraft": []
        }), 400