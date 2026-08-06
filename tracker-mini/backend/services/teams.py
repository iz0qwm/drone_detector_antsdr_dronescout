import time
from datetime import datetime, timezone
from services import meshtastic_service
import json
from pathlib import Path

from config import BASE_DIR, SETTINGS

MESHTASTIC_OPERATOR_STALE_MS = 600000
MESHTASTIC_OPERATOR_RETENTION_MS = 1800000

def node_name(node):

    return (
        node.get("longName")
        or node.get("shortName")
        or node.get("id")
        or "Unknown"
    )


REGIONS = {
    0: "UNSET",
    1: "US",
    2: "EU433",
    3: "EU868",
    4: "CN",
}



def get_current_mission_path():

    current = (
        BASE_DIR /
        "missions" /
        "current_mission.json"
    )

    if not current.exists():
        return None

    with open(current, "r") as f:
        data = json.load(f)

    mission_id = data.get("mission_id")

    if not mission_id:
        return None

    mission_path = (
        BASE_DIR /
        "missions" /
        mission_id
    )

    if not mission_path.exists():
        return None

    return mission_path


def get_team_file():

    mission = get_current_mission_path()

    if mission is None:
        return None

    return mission / "teams.json"



def load_team():

    team_file = get_team_file()

    if team_file is None:
        return {
            "operators": []
        }

    if not team_file.exists():

        data = {
            "operators": []
        }

        save_team(data)

        return data

    with open(team_file, "r") as f:
        return json.load(f)
    

def save_team(data):

    team_file = get_team_file()

    if team_file is None:
        return

    with open(team_file, "w") as f:

        json.dump(
            data,
            f,
            indent=2
        )

def get_team_operators():

    return load_team()["operators"]


def normalize_node(node):

    battery = node.get("batteryLevel")
    voltage = node.get("voltage")

    if battery == 101:
        battery = "External Power"

    if voltage is not None and voltage <= 0:
        voltage = None

    return {

        "nodeId": node.get("id"),
        "num": node.get("num"),

        "name": node_name(node),
        "shortName": node.get("shortName"),

        "hwModel": node.get("hwModel"),
        "role": node.get("role"),

        "lat": node.get("lat"),
        "lon": node.get("lon"),
        "altitude": node.get("altitude"),

        "battery": battery,
        "voltage": voltage,

        "channelUtilization":
            node.get("channelUtilization"),

        "snr": node.get("snr"),
        "hopStart": node.get("hopStart"),
        "hopLimit": node.get("hopLimit"),

        "viaMqtt": node.get("viaMqtt"),

        "last_seen": node.get("last_seen"),
        "lastHeard": node.get("lastHeard"),

    }


def _operator_stale_ms():

    return (
        SETTINGS
        .get("meshtastic", {})
        .get(
            "operator_stale_ms",
            MESHTASTIC_OPERATOR_STALE_MS
        )
    )


def _operator_retention_ms():

    return (
        SETTINGS
        .get("meshtastic", {})
        .get(
            "operator_retention_ms",
            MESHTASTIC_OPERATOR_RETENTION_MS
        )
    )


def _last_seen_timestamp(item):

    last_seen = item.get("last_seen")

    if not last_seen:
        return None

    try:
        dt = datetime.fromisoformat(
            last_seen.replace("Z", "+00:00")
        )

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.timestamp()
    except Exception:
        return None


def annotate_operator_freshness(operator, now=None):

    now = now if now is not None else time.time()
    copy = dict(operator)
    seen_ts = _last_seen_timestamp(copy)
    stale_ms = _operator_stale_ms()
    retention_ms = _operator_retention_ms()

    copy["stale_ms"] = stale_ms
    copy["retention_ms"] = retention_ms

    if seen_ts is None:
        copy["updatedAt"] = None
        copy["age_ms"] = None
        copy["stale"] = False
        copy["expired"] = False
        return copy

    age_ms = max(
        0,
        int((now - seen_ts) * 1000)
    )

    copy["updatedAt"] = int(seen_ts * 1000)
    copy["age_ms"] = age_ms
    copy["stale"] = age_ms > stale_ms
    copy["expired"] = age_ms > retention_ms

    return copy


def is_operator_expired(operator, now=None):

    return annotate_operator_freshness(
        operator,
        now
    ).get(
        "expired",
        False
    )


def is_gateway_node(node, gateway):

    gateway_num = gateway.get(
        "node_num"
    )

    node_num = node.get(
        "num"
    )

    return (
        gateway_num is not None
        and node_num == gateway_num
    )


def is_operator_node(node):

    name = node_name(
        node
    ).lower()

    short_name = (
        node.get("shortName")
        or ""
    ).lower()

    return (
        name.startswith("op")
        or name.startswith("operatore")
        or short_name.startswith("op")
    )


def get_team_status():

    refresh_operator_status()

    nodes = meshtastic_service.get_nodes()

    gateway = meshtastic_service.get_gateway_info()

    operators = []
    external_nodes = []

    gateway_node = None

    for node in nodes:

        normalized = normalize_node(
            node
        )

        if is_gateway_node(
            node,
            gateway
        ):

            gateway_node = normalized
            continue

        operator = find_configured_operator(node)

        if operator:

            bind_operator_node(
                operator,
                normalized
            )

            merged = operator.copy()

            merged.update(normalized)

            merged = annotate_operator_freshness(
                merged
            )

            if merged.get("expired"):
                continue

            operators.append(
                merged
            )

        else:

            external_nodes.append(
                normalized
            )

    return {
        "gateway": gateway,
        "gateway_node": gateway_node,
        "operators": operators,
        "external_nodes": external_nodes,
        "operator_freshness": {
            "stale_ms": _operator_stale_ms(),
            "retention_ms": _operator_retention_ms()
        },
        "messages": []
    }



def find_configured_operator(node):

    short_name = (
        node.get("shortName")
        or ""
    )

    for operator in get_team_operators():

        if (
            operator["shortName"]
            == short_name
        ):
            return operator

    return None


def bind_operator_node(operator, node):

    team = load_team()

    changed = False

    for op in team["operators"]:

        if op["id"] != operator["id"]:
            continue

        node_id = node.get("nodeId")

        seen_at = (
            node.get("last_seen")
            or datetime.now(timezone.utc).isoformat()
        )

        if op.get("nodeId") != node_id:

            op["nodeId"] = node_id
            changed = True

        if op.get("lastSeen") != seen_at:

            op["lastSeen"] = seen_at
            changed = True

        break

    if changed:

        save_team(team)



def refresh_operator_status():

    team = load_team()

    if not team["operators"]:
        return

    alive_nodes = set()

    for node in meshtastic_service.get_nodes():

        normalized = normalize_node(node)

        if is_operator_expired(normalized):
            continue

        alive_nodes.add(
            node["id"]
        )

    changed = False

    for op in team["operators"]:

        online = (
            op.get("nodeId") is not None
            and
            op.get("nodeId") in alive_nodes
        )

        if op.get("online") != online:

            op["online"] = online

            op["lastStatusUpdate"] = (
                datetime.utcnow().isoformat()
            )

            changed = True

    if changed:

        save_team(team)
