from flask import Blueprint, Response
from services.maps import get_tile
from flask import (
    jsonify,
    request
)
from services.maps import (
    get_tile,
    list_maps,
    get_storage_info,
    delete_map,
    update_map_description,
    load_map_provider,
    save_map_provider,
    set_map_active
)
import threading

from services.map_downloader import (
    create_download_job,
    create_offline_map,
    update_download_job,
    get_download_status,
    list_downloads
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


@maps_bp.route(
    "/api/maps/update-description",
    methods=["POST"]
)
def update_description():

    data = request.json

    map_name = data.get(
        "name",
        ""
    )

    description = data.get(
        "description",
        ""
    )

    update_map_description(
        map_name,
        description
    )

    return jsonify({
        "success": True
    })

@maps_bp.route(
    "/api/maps/downloads"
)
def downloads():

    return jsonify(
        list_downloads()
    )



@maps_bp.route(
    "/api/maps/download",
    methods=["POST"]
)
def download_map():

    data = request.json

    lat = float(data["lat"])
    lon = float(data["lon"])
    radius = float(data["radius"])

    description = data.get(
        "description",
        ""
    )

    job_id = create_download_job()

    def worker():

        try:

            update_download_job(
                job_id,
                status="running",
                message="Downloading map"
            )

            result = create_offline_map(
                lat=lat,
                lon=lon,
                radius_km=radius,
                description=description,
                job_id=job_id
            )

            update_download_job(
                job_id,
                status="completed",
                progress=100,
                message="Completed",
                result=result
            )

        except Exception as e:

            update_download_job(
                job_id,
                status="error",
                message=str(e)
            )

    threading.Thread(
        target=worker,
        daemon=True
    ).start()

    return jsonify({
        "success": True,
        "job_id": job_id
    })

@maps_bp.route(
    "/api/maps/download-status/<job_id>"
)
def download_status(job_id):

    return jsonify(
        get_download_status(job_id)
        or {}
    )



@maps_bp.route(
    "/api/maps/provider",
    methods=["GET"]
)
def get_provider():

    provider = load_map_provider()

    return jsonify({
        "provider": provider.get(
            "provider",
            "thunderforest"
        ),
        "configured": bool(
            provider.get(
                "api_key",
                ""
            )
        )
    })


@maps_bp.route(
    "/api/maps/provider",
    methods=["POST"]
)
def set_provider():

    data = request.json

    save_map_provider(
        data.get(
            "provider",
            "thunderforest"
        ),
        data.get(
            "api_key",
            ""
        )
    )

    return jsonify({
        "success": True
    })

@maps_bp.route(
    "/api/maps/set-active",
    methods=["POST"]
)
def set_active():

    data = request.json

    set_map_active(
        data["name"],
        data["active"]
    )

    return jsonify({
        "success": True
    })