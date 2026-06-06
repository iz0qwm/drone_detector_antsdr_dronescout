from flask import Blueprint, jsonify
from services.ds110 import get_aircraft

remoteid_bp = Blueprint(
    "remoteid",
    __name__
)

@remoteid_bp.route(
    "/api/remoteid/aircraft"
)
def remoteid_aircraft():

    return jsonify(
        get_aircraft()
    )