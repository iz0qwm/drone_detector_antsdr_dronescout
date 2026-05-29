from flask import Blueprint, Response
from services.maps import get_tile
from flask import jsonify
from services.maps import (
    get_tile,
    list_maps,
    get_storage_info,
    delete_map
)


maps_bp = Blueprint("maps", __name__)


@maps_bp.route("/tiles/<int:z>/<int:x>/<int:y>.png")
def tiles(z, x, y):

    tile = get_tile(z, x, y)

    if tile is None:
        return ("Tile not found", 404)

    return Response(tile, mimetype="image/png")

@maps_bp.route("/api/maps")
def maps_list():

    return jsonify(
        list_maps()
    )


@maps_bp.route("/api/maps/storage")
def maps_storage():

    return jsonify(
        get_storage_info()
    )


@maps_bp.route("/api/maps/<path:map_name>",
               methods=["DELETE"])
def remove_map(map_name):

    ok, msg = delete_map(map_name)

    return jsonify({
        "success": ok,
        "message": msg
    })