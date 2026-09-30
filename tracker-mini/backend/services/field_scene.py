"""Private, bounded projection of saved mission areas and received RID state."""
import json
import math
import re
import time
from datetime import datetime, timezone

SERIAL = "MTRK26-0001"
NODE_ID = "dsc-node02"
MAX_BYTES = 128 * 1024
MAX_RID_TARGETS = 32
RID_POSITION_MAX_AGE_MS = 10000


def text(value, maximum):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError("invalid_text")
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise ValueError("invalid_id")
    return value


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def point(value):
    if not isinstance(value, list) or len(value) != 2 or not all(number(v) for v in value):
        raise ValueError("invalid_coordinate")
    if abs(value[0]) > 180 or abs(value[1]) > 90:
        raise ValueError("invalid_coordinate")
    return list(value)


def geometry(feature):
    geom = feature["geometry"]
    props = feature.get("properties") or {}
    if geom["type"] == "Point" and props.get("leafletType") == "Circle":
        radius = props.get("radius")
        if not number(radius) or not 0 < radius <= 100000:
            raise ValueError("invalid_radius")
        return {"type": "Circle", "center": point(geom["coordinates"]), "radiusM": radius}
    if geom["type"] in ("Point", "MultiPoint", "LineString", "MultiLineString"):
        return None
    if geom["type"] != "Polygon":
        raise ValueError("unsupported_geometry")
    rings = geom["coordinates"]
    if not isinstance(rings, list) or not 1 <= len(rings) <= 20:
        raise ValueError("invalid_polygon")
    normalized, total = [], 0
    for ring in rings:
        if not isinstance(ring, list) or len(ring) < 4:
            raise ValueError("invalid_ring")
        total += len(ring)
        if total > 2000:
            raise ValueError("polygon_too_large")
        coords = [point(p) for p in ring]
        if coords[0] != coords[-1] or len(set(map(tuple, coords))) < 3:
            raise ValueError("invalid_ring")
        normalized.append(coords)
    return {"type": "Polygon", "coordinates": normalized}


def normalize(mission, layers):
    operation = {"id": identifier(mission["id"]), "name": text(mission["name"], 200)}
    areas = []
    for layer in layers:
        # Existing user draw layers have no source; imported layers explicitly do.
        if (layer.get("properties") or {}).get("source"):
            continue
        layer_id = identifier(layer["id"])
        geo = layer.get("geojson")
        if not isinstance(geo, dict):
            raise ValueError("invalid_geojson")
        features = geo.get("features") if geo.get("type") == "FeatureCollection" else [geo]
        if not isinstance(features, list) or len(features) > 100:
            raise ValueError("collection_too_large")
        for index, feature in enumerate(features):
            if feature.get("type") != "Feature":
                raise ValueError("invalid_feature")
            if (feature.get("properties") or {}).get("source"):
                continue
            shape = geometry(feature)
            if shape is None:
                continue
            area = {"id": f"{SERIAL}:{operation['id']}:{layer_id}:{index}",
                    "name": text(layer.get("name"), 200), "geometry": shape,
                    "origin": "LIVE", "source": "MINI_TRACKER"}
            color = (layer.get("style") or {}).get("color")
            if isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                area["style"] = {"color": color}
            areas.append(area)
            if len(areas) > 100:
                raise ValueError("too_many_areas")
    if len({a["id"] for a in areas}) != len(areas):
        raise ValueError("duplicate_area")
    return {"operation": operation, "areas": areas}


def normalize_rid_targets(aircraft, sampled_at):
    """Project recent DS110 Location observations without area containment."""
    targets = []
    if not isinstance(aircraft, list):
        raise ValueError("invalid_rid_state")
    for item in aircraft:
        if not isinstance(item, dict) or item.get("source") not in ("RemoteID", "DJI DroneID"):
            continue
        serial = item.get("serial")
        if not isinstance(serial, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", serial):
            continue
        lat, lon = item.get("lat"), item.get("lon")
        if not number(lat) or not number(lon) or abs(lat) > 90 or abs(lon) > 180 or (lat == 0 and lon == 0):
            continue
        try:
            observed = datetime.fromisoformat(item["position_observed_at"].replace("Z", "+00:00"))
            if observed.tzinfo is None:
                continue
            observed_at = int(observed.astimezone(timezone.utc).timestamp() * 1000)
        except (KeyError, AttributeError, TypeError, ValueError, OverflowError):
            continue
        if observed_at <= 0 or observed_at > sampled_at + 2000 or sampled_at - observed_at > RID_POSITION_MAX_AGE_MS:
            continue
        source_id = "remoteid" if item["source"] == "RemoteID" else "dji"
        target = {"id": f"rid:{source_id}:{serial}", "name": f"Remote ID {serial}",
                  "type": "RID", "origin": "LIVE", "source": "LOCAL_RX",
                  "position": {"lat": lat, "lon": lon}, "observedAt": observed_at,
                  "observedAtMs": observed_at, "receivedAtUtcMs": observed_at,
                  "timeBasis": "UTC", "timestampQuality": "SOURCE",
                  "sources": ["LOCAL_RX"], "positionSource": "MINI_TRACKER_RID",
                  "sourceEpoch": f"{NODE_ID}_{source_id}"}
        altitude = item.get("altitude")
        if number(altitude) and -1000 < altitude <= 30000:
            target["altitude"] = {"value": altitude, "unit": "m", "reference": "WGS84_ELLIPSOID"}
        heading = item.get("heading")
        if number(heading) and 0 <= heading <= 360:
            target["heading"] = heading % 360
        targets.append(target)
    targets.sort(key=lambda target: target["id"])
    unique = {target["id"]: target for target in targets}
    return list(unique.values())[:MAX_RID_TARGETS]


class SceneCollector:
    def __init__(self, current, layers, selected, now=time.time, traffic=lambda: []):
        self.current, self.layers, self.selected, self.now, self.traffic = current, layers, selected, now, traffic

    def collect(self):
        base = {"schemaVersion": 1, "nodeId": NODE_ID, "serial": SERIAL,
                "sampledAt": int(self.now() * 1000)}
        try:
            selected = self.selected()
            if selected is not None:
                identifier(selected)
            mission = self.current()
            if selected is None and mission is None:
                projection = {"operation": None, "areas": []}
            else:
                if not mission or mission.get("id") != selected:
                    raise ValueError("selection_unavailable")
                identifier(selected)
                projection = normalize(mission, self.layers(selected))
            if self.selected() != selected:
                raise ValueError("selection_changed")
            projection["targets"] = normalize_rid_targets(self.traffic(), base["sampledAt"])
            result = {**base, "status": "OK", **projection}
            if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > MAX_BYTES:
                raise ValueError("snapshot_too_large")
            return result
        except Exception:
            # Never publish partial/empty success on failed disk reads/normalization.
            # No file contents, paths or private geometry in logs or error payloads.
            return {**base, "status": "ERROR", "errorCode": "SCENE_READ_FAILED"}


def default_collector():
    from services.missions import get_current_mission
    from services.layer_storage import list_layers, mission_layers_dir
    from services.mission_storage import get_current_mission_id, MISSIONS_DIR

    def read_selection():
        if not MISSIONS_DIR.is_dir():
            raise ValueError("mission_storage_unavailable")
        return get_current_mission_id()

    def read_layers(mission_id):
        if not mission_layers_dir(mission_id).is_dir():
            raise ValueError("layers_unavailable")
        return list_layers(mission_id)

    def read_traffic():
        try:
            from services.ds110 import get_aircraft
        except ModuleNotFoundError as error:
            if error.name != "pymavlink":
                raise
            return []  # Development host without the DS110 runtime dependency.
        return get_aircraft()

    return SceneCollector(get_current_mission, read_layers, read_selection, traffic=read_traffic)
