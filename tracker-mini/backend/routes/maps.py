from flask import Blueprint, Response
from services.maps import get_tile

maps_bp = Blueprint("maps", __name__)


@maps_bp.route("/tiles/<int:z>/<int:x>/<int:y>.png")
def tiles(z, x, y):

    tile = get_tile(z, x, y)

    if tile is None:
        return ("Tile not found", 404)

    return Response(tile, mimetype="image/png")