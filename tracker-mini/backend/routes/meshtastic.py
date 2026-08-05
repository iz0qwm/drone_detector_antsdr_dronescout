from flask import Blueprint, jsonify, request
from config import SETTINGS, save_settings
from services import meshtastic_service
from services.logger import log
from services import notification_service

meshtastic_bp = Blueprint(
    "meshtastic",
    __name__,
    url_prefix="/api/meshtastic"
)


@meshtastic_bp.route("/nodes")
def get_nodes():
    return jsonify({
        "ok": True,
        "nodes": meshtastic_service.get_nodes(),
        "alive": meshtastic_service.is_alive()
    })


@meshtastic_bp.route("/status")
def get_status():
    configured = SETTINGS.get(
        "traffic",
        {}
    ).get(
        "meshtastic_enabled",
        False
    )

    return jsonify({
        "ok": True,
        "configured": configured,
        "enabled": meshtastic_service.running,
        "alive": meshtastic_service.is_alive(),
        "nodes_count": len(
            meshtastic_service.get_nodes()
        )
    })


@meshtastic_bp.route("/gateway")
def get_gateway():

    return jsonify({
        "ok": True,
        "gateway": meshtastic_service.get_gateway_info()
    })

@meshtastic_bp.route(
    "/enable",
    methods=["POST"]
)
def enable_meshtastic():

    data = request.json or {}

    enabled = data.get(
        "enabled",
        True
    )

    SETTINGS.setdefault(
        "traffic",
        {}
    )[
        "meshtastic_enabled"
    ] = bool(enabled)

    save_settings()

    if enabled:

        log(
            "MESHTASTIC",
            "Meshtastic receiver enabled in settings"
        )

        meshtastic_service.start()

    else:

        log(
            "MESHTASTIC",
            "Meshtastic receiver disabled in settings"
        )

        meshtastic_service.stop()
        meshtastic_service.clear_nodes()

    return jsonify({
        "success": True,
        "configured": bool(enabled),
        "enabled": meshtastic_service.running
    })


@meshtastic_bp.route(
    "/nodes/reset",
    methods=["POST"]
)
def reset_nodes():

    try:
        meshtastic_service.reset_nodedb()

        return jsonify({
            "ok": True
        })

    except Exception as e:

        log(
            "MESHTASTIC",
            f"NodeDB reset error: {e}"
        )

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


@meshtastic_bp.route(
    "/nodes/<path:node_id>",
    methods=["DELETE"]
)
def remove_node(node_id):

    try:
        meshtastic_service.remove_node(
            node_id
        )

        return jsonify({
            "ok": True
        })

    except Exception as e:

        log(
            "MESHTASTIC",
            f"Remove node error: {e}"
        )

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500
    
