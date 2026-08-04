"""
Aircraft normalization, source provenance, and deduplication.
"""
import time
from dataclasses import dataclass, field
from services.proximity.calc import is_valid_position


@dataclass
class NormalizedTarget:
    """Deduplicated aircraft representation with source provenance."""
    track_id: str
    icao: str = None
    callsign: str = None
    latitude: float = None
    longitude: float = None
    altitude: float = None
    altitude_reference: str = "unknown"
    ground_speed: float = None
    track_heading: float = None
    updated_at: float = 0.0  # seconds epoch, most recent from any source
    primary_source: str = "ADSBRx"
    sources: list = field(default_factory=list)
    source_timestamps: dict = field(default_factory=dict)
    is_helicopter: bool = False
    category: str = None

    def is_fresh(self, stale_ms):
        """Target is fresh if at least one source has a fresh timestamp."""
        now = time.time()
        threshold_s = stale_ms / 1000.0
        for ts in self.source_timestamps.values():
            if ts and (now - ts) < threshold_s:
                return True
        return False

    @property
    def source_label(self):
        """UI-facing source provenance label."""
        if "ADSBRx" in self.sources and "ADSBNet" in self.sources:
            return "RX+NET"
        elif "ADSBRx" in self.sources:
            return "RX"
        elif "ADSBNet" in self.sources:
            return "NET"
        return "?"


@dataclass
class ProviderResult:
    """Result of a single provider fetch attempt."""
    provider: str
    successful: bool
    aircraft: list = field(default_factory=list)
    fetch_timestamp: float = 0.0
    response_duration_ms: float = 0.0
    error: str = None


def normalize_adsbrx(aircraft_list):
    """
    Convert ADSBRx aircraft list to NormalizedTarget list.
    Input format: air_local.get_local_aircraft() response items.
    """
    targets = []
    now = time.time()

    for ac in aircraft_list:
        lat = ac.get("lat")
        lon = ac.get("lon")

        if not is_valid_position(lat, lon, "aircraft"):
            continue

        icao = ac.get("icao")
        if not icao:
            continue

        track_id = icao.lower()

        # Convert updatedAt (ms epoch) to seconds
        updated_ms = ac.get("updatedAt") or 0
        updated_s = updated_ms / 1000.0 if updated_ms > 1_000_000_000_000 else updated_ms

        targets.append(NormalizedTarget(
            track_id=track_id,
            icao=icao,
            callsign=ac.get("callsign"),
            latitude=lat,
            longitude=lon,
            altitude=ac.get("altitude"),
            altitude_reference="baro_msl",
            ground_speed=ac.get("speed"),
            track_heading=ac.get("heading"),
            updated_at=updated_s if updated_s > 0 else now,
            primary_source="ADSBRx",
            sources=["ADSBRx"],
            source_timestamps={"ADSBRx": updated_s if updated_s > 0 else now},
            is_helicopter=bool(ac.get("isHelicopter")),
            category=ac.get("category"),
        ))

    return targets


def normalize_adsbnet(aircraft_list):
    """
    Convert ADSBNet aircraft list to NormalizedTarget list.
    Input format: air_network.get_network_aircraft() response items.
    """
    targets = []
    now = time.time()

    for ac in aircraft_list:
        lat = ac.get("lat")
        lon = ac.get("lon")

        if not is_valid_position(lat, lon, "aircraft"):
            continue

        icao = ac.get("icao")
        if not icao:
            # Synthetic ID for unmerged network targets
            track_id = f"net_{id(ac)}"
        else:
            track_id = icao.lower()

        updated_ms = ac.get("updatedAt") or 0
        updated_s = updated_ms / 1000.0 if updated_ms > 1_000_000_000_000 else updated_ms

        # Determine altitude reference from source
        source = ac.get("source", "")
        if "OPENSKY" in source:
            alt_ref = "geo_msl"
        else:
            alt_ref = "baro_msl"

        targets.append(NormalizedTarget(
            track_id=track_id,
            icao=icao,
            callsign=ac.get("callsign"),
            latitude=lat,
            longitude=lon,
            altitude=ac.get("altitude"),
            altitude_reference=alt_ref,
            ground_speed=ac.get("speed"),
            track_heading=ac.get("heading"),
            updated_at=updated_s if updated_s > 0 else now,
            primary_source="ADSBNet",
            sources=["ADSBNet"],
            source_timestamps={"ADSBNet": updated_s if updated_s > 0 else now},
            is_helicopter=bool(ac.get("isHelicopter")),
            category=ac.get("category"),
        ))

    return targets


def merge_targets(adsbrx_targets, adsbnet_targets, tie_window_s=3.0):
    """
    Deduplicate and merge ADSBRx and ADSBNet targets.

    Position precedence (timestamp-aware):
    1. Discard stale positions (handled externally by freshness check)
    2. Prefer newest valid position
    3. Within tie_window_s, prefer ADSBRx
    4. Supplement missing fields from the other source
    5. Never replace newer with older
    """
    # Index ADSBRx by track_id
    rx_index = {}
    for t in adsbrx_targets:
        rx_index[t.track_id] = t

    # Index ADSBNet by track_id
    net_index = {}
    for t in adsbnet_targets:
        if t.track_id in net_index:
            # Keep newer among net duplicates
            if t.updated_at > net_index[t.track_id].updated_at:
                net_index[t.track_id] = t
        else:
            net_index[t.track_id] = t

    merged = {}

    # Process all ADSBRx targets
    for tid, rx in rx_index.items():
        if tid in net_index:
            net = net_index[tid]
            merged[tid] = _merge_pair(rx, net, tie_window_s)
        else:
            merged[tid] = rx

    # Add ADSBNet-only targets
    for tid, net in net_index.items():
        if tid not in merged:
            merged[tid] = net

    return list(merged.values())


def _merge_pair(rx, net, tie_window_s):
    """Merge an ADSBRx and ADSBNet target for the same ICAO."""
    rx_ts = rx.source_timestamps.get("ADSBRx", 0)
    net_ts = net.source_timestamps.get("ADSBNet", 0)

    # Determine position winner
    time_diff = abs(rx_ts - net_ts)
    if time_diff <= tie_window_s:
        # Within tie window: prefer ADSBRx
        winner = rx
        primary = "ADSBRx"
    elif rx_ts >= net_ts:
        winner = rx
        primary = "ADSBRx"
    else:
        winner = net
        primary = "ADSBNet"

    # Build merged target using winner's position
    result = NormalizedTarget(
        track_id=rx.track_id,
        icao=rx.icao or net.icao,
        callsign=rx.callsign or net.callsign,
        latitude=winner.latitude,
        longitude=winner.longitude,
        altitude=winner.altitude,
        altitude_reference=winner.altitude_reference,
        ground_speed=winner.ground_speed if winner.ground_speed is not None else (
            rx.ground_speed if rx.ground_speed is not None else net.ground_speed
        ),
        track_heading=winner.track_heading if winner.track_heading is not None else (
            rx.track_heading if rx.track_heading is not None else net.track_heading
        ),
        updated_at=max(rx_ts, net_ts),
        primary_source=primary,
        sources=["ADSBRx", "ADSBNet"],
        source_timestamps={"ADSBRx": rx_ts, "ADSBNet": net_ts},
        is_helicopter=rx.is_helicopter or net.is_helicopter,
        category=rx.category or net.category,
    )

    return result
