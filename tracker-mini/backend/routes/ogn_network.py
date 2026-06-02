from flask import Blueprint, jsonify, request

from services.ogn_network import (
    get_bounds_args,
    get_ogn_traffic
)

ogn_network_bp = Blueprint(
    "ogn_network",
    __name__
)


@ogn_network_bp.route(
    "/api/ogn/network"
)
def ogn_network():

    try:
        bounds = get_bounds_args(
            request
        )

        return jsonify(
            get_ogn_traffic(
                bounds
            )
        )

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e),
            "objects": [],
            "count": 0
        }), 400