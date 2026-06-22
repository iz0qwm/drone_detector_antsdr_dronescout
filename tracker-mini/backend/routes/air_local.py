from flask import Blueprint
from flask import jsonify
from flask import request

from services.air_local import (
    get_bounds_args,
    get_local_aircraft
)

air_local_bp = Blueprint(
    "air_local",
    __name__
)


@air_local_bp.route(
    "/api/air/local"
)
def air_local():

    try:

        bounds = get_bounds_args(
            request
        )

        show_all = (
            request.args.get("showAll")
            == "true"
        )

        return jsonify(
            get_local_aircraft(
                bounds,
                show_all
            )
        )

    except Exception as e:

        import traceback

        from services.logger import log

        log(
            "ADSB",
            "AIR LOCAL API ERROR:\n"
            + traceback.format_exc(),
            level="ERROR"
        )

        return jsonify({
            "success": False,
            "message": str(e),
            "aircraft": []
        }), 400