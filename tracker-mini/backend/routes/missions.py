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
    update_mission,
    import_geojson
)

from services.layer_storage import (
    list_layers,
    get_layer,
    save_layer,
    delete_layer
)


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

    if result is None:

        return jsonify({
            "success": False,
            "message": "Unable to import GeoJSON"
        }), 400

    return jsonify({
        "success": True,
        "layer": result
    })


@missions_bp.route(
    "/api/missions/<mission_id>/layers",
    methods=["GET"]
)
def api_layers(
    mission_id
):

    mission = get_mission(
        mission_id
    )

    if mission is None:

        return jsonify({
            "success": False,
            "message": "Mission not found"
        }), 404

    return jsonify(
        list_layers(
            mission_id
        )
    )


@missions_bp.route(
    "/api/missions/<mission_id>/layers/<layer_id>",
    methods=["GET"]
)
def api_layer_content(
    mission_id,
    layer_id
):

    layer = get_layer(
        mission_id,
        layer_id
    )

    if layer is None:

        return jsonify({
            "success": False,
            "message": "Layer not found"
        }), 404

    return jsonify(
        layer
    )


@missions_bp.route(
    "/api/missions/<mission_id>/layers",
    methods=["POST"]
)
def api_create_layer(
    mission_id
):

    mission = get_mission(
        mission_id
    )

    if mission is None:

        return jsonify({
            "success": False,
            "message": "Mission not found"
        }), 404

    data = request.get_json() or {}

    layer = {
        "name": data.get(
            "name",
            "New Layer"
        ),
        "type": data.get(
            "type",
            "generic"
        ),
        "geometry": data.get(
            "geometry",
            None
        ),
        "visible": data.get(
            "visible",
            True
        ),
        "locked": data.get(
            "locked",
            False
        ),
        "style": data.get(
            "style",
            {}
        ),
        "properties": data.get(
            "properties",
            {}
        ),
        "geojson": data.get(
            "geojson",
            None
        )
    }

    saved = save_layer(
        mission_id,
        layer
    )

    return jsonify({
        "success": True,
        "layer": saved
    })


@missions_bp.route(
    "/api/missions/<mission_id>/layers/<layer_id>",
    methods=["PUT"]
)
def api_update_layer(
    mission_id,
    layer_id
):

    mission = get_mission(
        mission_id
    )

    if mission is None:

        return jsonify({
            "success": False,
            "message": "Mission not found"
        }), 404

    existing = get_layer(
        mission_id,
        layer_id
    )

    if existing is None:

        return jsonify({
            "success": False,
            "message": "Layer not found"
        }), 404

    data = request.get_json() or {}

    existing.update(
        data
    )

    existing["id"] = layer_id

    saved = save_layer(
        mission_id,
        existing
    )

    return jsonify({
        "success": True,
        "layer": saved
    })



@missions_bp.route(
    "/api/missions/<mission_id>/layers/<layer_id>",
    methods=["DELETE"]
)
def api_delete_layer(
    mission_id,
    layer_id
):

    mission = get_mission(
        mission_id
    )

    if mission is None:

        return jsonify({
            "success": False,
            "message": "Mission not found"
        }), 404

    ok = delete_layer(
        mission_id,
        layer_id
    )

    return jsonify({
        "success": ok
    })


@missions_bp.route(
    "/api/missions/<mission_id>",
    methods=["PUT"]
)
def api_update_mission(
    mission_id
):

    mission = get_mission(
        mission_id
    )

    if mission is None:

        return jsonify({
            "success": False,
            "message": "Mission not found"
        }), 404

    data = request.get_json() or {}

    mission = update_mission(
        mission_id,
        data
    )

    return jsonify({
        "success": True,
        "mission": mission
    })