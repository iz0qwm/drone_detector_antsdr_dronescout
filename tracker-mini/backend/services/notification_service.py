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
    source="tracker",
    status="created"
):
    notification = {
        "id": str(uuid.uuid4()),
        "timestamp": now_iso(),
        "category": category,
        "severity": severity,
        "source": source,
        "target": target,
        "target_node_id": target_node_id,
        "text": text,
        "status": status
    }

    notifications.append(notification)

    return notification


def get_notifications():
    return list(reversed(notifications))


def clear_notifications():
    notifications.clear()


def send_to_operator(node_id, text, category="manual", severity="info"):
    notification = add_notification(
        category=category,
        severity=severity,
        text=text,
        target="operator",
        target_node_id=node_id,
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
        node_id = op.get("nodeId") or op.get("id")

        if not op.get("online"):
            continue

        if not node_id:
            continue

        sent.append(
            send_to_operator(
                node_id=node_id,
                text=text,
                category=category,
                severity=severity
            )
        )

        time.sleep(0.2)

    return sent