"""Private, bounded projection of saved mission areas. No network or hardware."""
import json
import math
import re
import time

SERIAL = "MTRK26-0001"
NODE_ID = "dsc-node02"
MAX_BYTES = 128 * 1024


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


class SceneCollector:
    def __init__(self, current, layers, selected, now=time.time):
        self.current, self.layers, self.selected, self.now = current, layers, selected, now

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

    return SceneCollector(get_current_mission, read_layers, read_selection)
