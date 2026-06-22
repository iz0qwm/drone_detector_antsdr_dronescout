from flask import Blueprint, jsonify, request
from services import meshtastic_service
from services.logger import log

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

    return jsonify({
        "ok": True,
        "enabled": meshtastic_service.running,
        "alive": meshtastic_service.is_alive(),
        "nodes_count": len(
            meshtastic_service.get_nodes()
        )
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

    if enabled:

        log(
            "MESHTASTIC",
            "Meshtastic receiver enabled"
        )

        meshtastic_service.start()

    else:

        log(
            "MESHTASTIC",
            "Meshtastic receiver disabled"
        )

        meshtastic_service.stop()
        meshtastic_service.clear_nodes()

    return jsonify({
        "success": True,
        "enabled": enabled
    })


