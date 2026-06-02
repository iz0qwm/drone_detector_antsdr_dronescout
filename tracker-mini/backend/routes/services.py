from flask import (
    Blueprint,
    jsonify
)

from services.services import (
    get_services_status
)

services_bp = Blueprint(
    "services",
    __name__
)

@services_bp.route(
    "/api/services"
)
def services():

    return jsonify(
        get_services_status()
    )