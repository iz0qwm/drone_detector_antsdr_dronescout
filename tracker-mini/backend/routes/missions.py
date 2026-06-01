from flask import (
    Blueprint,
    jsonify,
    request
)

from services.missions import (
    list_missions,
    create_mission,
    get_mission,
    get_current_mission,
    set_current_mission,
    delete_mission,
    import_geojson,
    list_imported_layers
)

from pathlib import Path
import json


missions_bp = Blueprint(
    "missions",
    __name__
)


@missions_bp.route(
    "/api/missions",
    methods=["GET"]
)
def api_missions():

    return jsonify(
        list_missions()
    )


@missions_bp.route(
    "/api/missions/create",
    methods=["POST"]
)
def api_create_mission():

    data = request.get_json()

    name = (
        data.get("name")
        or ""
    ).strip()

    if not name:

        return jsonify({
            "success": False,
            "message":
                "Mission name required"
        }), 400

    mission = create_mission(
        name=name,
        description=data.get(
            "description",
            ""
        )
    )

    return jsonify({
        "success": True,
        "mission": mission
    })



@missions_bp.route(
    "/api/missions/<mission_id>",
    methods=["GET"]
)
def api_get_mission(mission_id):

    mission = get_mission(
        mission_id
    )

    if mission is None:

        return jsonify({
            "success": False,
            "message":
                "Mission not found"
        }), 404

    return jsonify(mission)

@missions_bp.route(
    "/api/missions/current",
    methods=["GET"]
)
def api_current_mission():

    mission = get_current_mission()

    return jsonify(mission)


@missions_bp.route(
    "/api/missions/select",
    methods=["POST"]
)
def api_select_mission():
    data = request.get_json()

    mission_id = data.get(
            "mission_id"
        )
    ok = set_current_mission(
            mission_id
        )
    return jsonify({
        "success": ok
    })

@missions_bp.route(
    "/api/missions/<mission_id>",
    methods=["DELETE"]
)
def api_delete_mission(
    mission_id
):

    ok = delete_mission(
        mission_id
    )

    return jsonify({
        "success": ok
    })


@missions_bp.route(
    "/api/missions/import-geojson",
    methods=["POST"]
)
def api_import_geojson():

    mission_id = request.form.get(
        "mission_id"
    )

    file = request.files.get(
        "file"
    )

    if not mission_id:

        return jsonify({
            "success": False,
            "message":
                "Missing mission id"
        }), 400

    if not file:

        return jsonify({
            "success": False,
            "message":
                "Missing file"
        }), 400

    result = import_geojson(
        mission_id,
        file
    )

    return jsonify({
        "success": True,
        "file": result
    })


@missions_bp.route(
    "/api/missions/<mission_id>/layers",
    methods=["GET"]
)
def api_layers(
    mission_id
):

    return jsonify(
        list_imported_layers(
            mission_id
        )
    )


@missions_bp.route(
    "/api/missions/<mission_id>/layers/<filename>",
    methods=["GET"]
)
def api_layer_content(
    mission_id,
    filename
):

    layer_file = (
        Path("/home/pi/tracker-mini/missions") /
        mission_id /
        "imports" /
        filename
    )

    if not layer_file.exists():

        return jsonify({
            "success": False
        }), 404

    with open(
        layer_file,
        "r"
    ) as f:

        data = json.load(f)

    return jsonify(data)