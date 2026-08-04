"""
Distance calculation and coordinate validation for the proximity engine.
"""
import math


def haversine_meters(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance in meters between two WGS84 points.
    """
    R = 6_371_000  # Earth mean radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def is_valid_position(lat, lon, source_type="aircraft"):
    """
    Validate coordinates. Source-specific rules:
    - Drones: reject (0.0, 0.0) as ODID no-position sentinel.
    - Aircraft: accept (0.0, 0.0) because readsb omits fields
      for missing positions rather than using a sentinel.
    """
    if lat is None or lon is None:
        return False

    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return False

    if not math.isfinite(lat) or not math.isfinite(lon):
        return False

    if lat < -90 or lat > 90:
        return False

    if lon < -180 or lon > 180:
        return False

    # ODID sentinel: drones report (0,0) when no GPS fix
    if source_type == "drone" and lat == 0.0 and lon == 0.0:
        return False

    return True


def bounding_box_filter(center_lat, center_lon, radius_m, targets):
    """
    Fast pre-filter: reject targets outside a lat/lon bounding box
    approximating the given radius. Returns list of targets within box.
    """
    # Approximate degrees per meter at given latitude
    lat_deg_per_m = 1.0 / 111_320
    lon_deg_per_m = 1.0 / (111_320 * math.cos(math.radians(center_lat)))

    dlat = radius_m * lat_deg_per_m
    dlon = radius_m * lon_deg_per_m

    min_lat = center_lat - dlat
    max_lat = center_lat + dlat
    min_lon = center_lon - dlon
    max_lon = center_lon + dlon

    result = []
    for t in targets:
        tlat = t.get("latitude") if isinstance(t, dict) else getattr(t, "latitude", None)
        tlon = t.get("longitude") if isinstance(t, dict) else getattr(t, "longitude", None)

        if tlat is None or tlon is None:
            continue

        if min_lat <= tlat <= max_lat and min_lon <= tlon <= max_lon:
            result.append(t)

    return result


def evaluate_pairs(drones, targets, radius_m):
    """
    For each drone, calculate distances to all targets within radius.
    Returns list of (drone_id, target_id, distance_m) sorted by distance.
    """
    pairs = []

    for drone in drones:
        dlat = drone.get("lat") or drone.get("latitude")
        dlon = drone.get("lon") or drone.get("longitude")
        drone_id = drone.get("serial") or drone.get("id") or "unknown"

        if not is_valid_position(dlat, dlon, "drone"):
            continue

        # Pre-filter targets within bounding box
        nearby = bounding_box_filter(dlat, dlon, radius_m, targets)

        for target in nearby:
            tlat = target.get("latitude") if isinstance(target, dict) else getattr(target, "latitude", None)
            tlon = target.get("longitude") if isinstance(target, dict) else getattr(target, "longitude", None)
            target_id = (
                target.get("track_id")
                if isinstance(target, dict)
                else getattr(target, "track_id", None)
            ) or "unknown"

            if tlat is None or tlon is None:
                continue

            distance = haversine_meters(dlat, dlon, tlat, tlon)

            if distance <= radius_m:
                pairs.append({
                    "drone_id": drone_id,
                    "target_id": target_id,
                    "distance_m": distance,
                })

    # Sort by distance ascending
    pairs.sort(key=lambda p: p["distance_m"])
    return pairs
