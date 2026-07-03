from flask import Blueprint, jsonify, request

from services import notification_service

notifications_bp = Blueprint(
    "notifications",
    __name__,
    url_prefix="/api/notifications"
)


@notifications_bp.route(
    "",
    methods=["GET"]
)
def get_notifications():

    return jsonify({
        "ok": True,
        "messages": notification_service.get_notifications()
    })


@notifications_bp.route(
    "",
    methods=["DELETE"]
)
def clear_notifications():

    notification_service.clear_notifications()

    return jsonify({
        "ok": True
    })


@notifications_bp.route(
    "/operator",
    methods=["POST"]
)
def send_to_operator():

    data = request.json or {}

    notification = notification_service.send_to_operator(
        node_id=data.get("node_id"),
        text=data.get("text"),
        category=data.get("category", "manual"),
        severity=data.get("severity", "info")
    )

    return jsonify({
        "ok": notification["status"] == "sent",
        "notification": notification
    })


@notifications_bp.route(
    "/all",
    methods=["POST"]
)
def send_to_all():

    data = request.json or {}

    notifications = notification_service.send_to_all_operators(
        text=data.get("text"),
        category=data.get("category", "manual"),
        severity=data.get("severity", "info")
    )

    return jsonify({
        "ok": True,
        "notifications": notifications
    })