from flask import (
    Blueprint,
    jsonify,
    request
)

from services.dsc_settings import (
    get_dsc_settings,
    update_dsc_settings
)

dsc_bp = Blueprint(
    "dsc",
    __name__
)


@dsc_bp.route(
    "/api/dsc/settings"
)
def get_settings():

    return jsonify(
        get_dsc_settings()
    )


@dsc_bp.route(
    "/api/dsc/settings",
    methods=["POST"]
)
def save_settings_route():

    return jsonify(
        update_dsc_settings(
            request.json
        )
    )