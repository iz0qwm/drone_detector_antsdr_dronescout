import time
import uuid
from datetime import datetime, timezone

from services.logger import log
from services import meshtastic_service
from services.teams import (load_team)

notifications = []


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def add_notification(
    category,
    severity,
    text,
    target="all",
    target_node_id=None,
    source="Gateway",
    source_node_id=None,
    target_label=None,
    direction="outgoing",
    status="created",
    transport="meshtastic"
):
    notification = {
        "id": str(uuid.uuid4()),
        "timestamp": now_iso(),
        "category": category,
        "severity": severity,
        "source": source,
        "source_node_id": source_node_id,
        "target": target,
        "target_label": target_label,
        "target_node_id": target_node_id,
        "direction": direction,
        "transport": transport,
        "text": text,
        "status": status
    }

    notifications.append(notification)

    return notification


def get_notifications():
    return list(reversed(notifications))


def clear_notifications():
    notifications.clear()


def operator_label_for_node(node_id):
    if not node_id:
        return None

    team = load_team()

    for op in team.get("operators", []):
        if op.get("nodeId") == node_id:
            return (
                op.get("longName")
                or op.get("shortName")
                or node_id
            )

    return node_id


def send_to_operator(
    node_id,
    text,
    category="manual",
    severity="info",
    target_label=None
):
    notification = add_notification(
        category=category,
        severity=severity,
        text=text,
        target="operator",
        target_node_id=node_id,
        target_label=target_label or operator_label_for_node(node_id),
        source="Gateway",
        direction="outgoing",
        status="sending"
    )

    try:
        meshtastic_service.send_direct_message(
            node_id,
            text
        )

        notification["status"] = "sent"

        log(
            "NOTIFICATION",
            f"Sent to {node_id}: {text}"
        )

    except Exception as e:
        notification["status"] = "failed"
        notification["error"] = str(e)

        log(
            "NOTIFICATION",
            f"Send failed to {node_id}: {e}"
        )

    return notification


def send_to_all_operators(
    text,
    category="manual",
    severity="info"
):
    sent = []
    
    team = load_team()
    operators = team.get("operators", [])

    for op in operators:
        node_id = op.get("nodeId")

        if not op.get("online"):
            continue

        if not node_id:
            continue

        sent.append(
            send_to_operator(
                node_id=node_id,
                text=text,
                category=category,
                severity=severity,
                target_label=(
                    op.get("longName")
                    or op.get("shortName")
                    or node_id
                )
            )
        )

        time.sleep(0.2)

    return sent


def record_incoming_text(
    source_node_id,
    text,
    source_label=None,
    target_node_id=None,
    target_label="Gateway",
    category="manual",
    severity="info"
):
    return add_notification(
        category=category,
        severity=severity,
        text=text,
        target="gateway",
        target_node_id=target_node_id,
        target_label=target_label,
        source=source_label or source_node_id or "Operator",
        source_node_id=source_node_id,
        direction="incoming",
        status="received"
    )
