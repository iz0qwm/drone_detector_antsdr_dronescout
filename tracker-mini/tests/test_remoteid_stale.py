"""
Remote ID freshness tests.

These tests exercise the DS110 in-memory cache lifecycle only. They do not
validate physical DS110 hardware reception.
"""
from datetime import datetime, timezone
import importlib
import sys
import types
import struct


def _load_ds110(monkeypatch):
    sys.modules.pop("services.ds110", None)

    pymavlink = types.ModuleType("pymavlink")
    pymavlink.mavutil = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "pymavlink", pymavlink)

    config = types.ModuleType("config")
    config.SETTINGS = {
        "remoteid": {
            "marker_stale_ms": 45000,
            "marker_retention_ms": 180000,
        },
        "proximity": {
            "drone_stale_ms": 15000,
            "target_retention_ms": 60000,
        }
    }
    monkeypatch.setitem(sys.modules, "config", config)

    dsc_bridge = types.ModuleType("services.dsc_bridge")
    dsc_bridge.send_detected_drone_to_dsc = lambda drone: False
    monkeypatch.setitem(sys.modules, "services.dsc_bridge", dsc_bridge)

    return importlib.import_module("services.ds110")


def _iso_timestamp(epoch_seconds):
    return datetime.fromtimestamp(
        epoch_seconds,
        tz=timezone.utc
    ).isoformat()


def test_get_aircraft_marks_stale_remoteid_tracks(monkeypatch):
    ds110 = _load_ds110(monkeypatch)
    now = 1000000.0
    monkeypatch.setattr(ds110.time, "time", lambda: now)

    ds110.remoteid_aircraft["fresh"] = {
        "serial": "fresh",
        "lat": 41.0,
        "lon": 12.0,
        "last_seen": _iso_timestamp(now - 5),
    }
    ds110.remoteid_aircraft["stale"] = {
        "serial": "stale",
        "lat": 41.1,
        "lon": 12.1,
        "last_seen": _iso_timestamp(now - 50),
    }

    aircraft = {
        item["serial"]: item
        for item in ds110.get_aircraft()
    }

    assert aircraft["fresh"]["stale"] is False
    assert aircraft["fresh"]["age_ms"] == 5000
    assert aircraft["fresh"]["updatedAt"] == int((now - 5) * 1000)
    assert aircraft["stale"]["stale"] is True
    assert aircraft["stale"]["age_ms"] == 50000
    assert aircraft["stale"]["stale_ms"] == 45000
    assert aircraft["stale"]["retention_ms"] == 180000


def test_get_aircraft_removes_expired_remoteid_tracks(monkeypatch):
    ds110 = _load_ds110(monkeypatch)
    now = 1000000.0
    monkeypatch.setattr(ds110.time, "time", lambda: now)

    ds110.remoteid_aircraft["expired"] = {
        "serial": "expired",
        "lat": 41.0,
        "lon": 12.0,
        "last_seen": _iso_timestamp(now - 181),
    }

    assert ds110.get_aircraft() == []
    assert "expired" not in ds110.remoteid_aircraft


def test_only_valid_location_renews_private_position_observation(monkeypatch):
    ds110 = _load_ds110(monkeypatch)
    first = _iso_timestamp(1000000)
    later = _iso_timestamp(1000005)
    track = {}
    ds110.merge_odid_aircraft(track, {"source": "RemoteID", "serial": "TEST123",
                                     "lat": 42.0, "lon": 12.0, "last_seen": first})
    assert track["position_observed_at"] == first
    ds110.merge_odid_aircraft(track, {"source": "RemoteID", "serial": "TEST123",
                                     "operator_id": "operator", "lat": None, "lon": None,
                                     "last_seen": later})
    assert track["last_seen"] == later
    assert track["position_observed_at"] == first
    ds110.merge_odid_aircraft(track, {"lat": 0.0, "lon": 0.0, "last_seen": later})
    assert track["position_observed_at"] == first
    ds110.merge_odid_aircraft(track, {"lat": 42.001, "lon": 12.001,
                                     "last_seen": later})
    assert track["position_observed_at"] == later


def test_location_decodes_geometric_altitude_and_separate_zero_height(monkeypatch):
    ds110 = _load_ds110(monkeypatch)
    block = bytearray(25)
    block[0] = 0x10  # OpenDroneID Location
    block[5:9] = struct.pack("<i", 420000000)
    block[9:13] = struct.pack("<i", 120000000)
    block[15:17] = struct.pack("<H", 2236)  # (118 + 1000) * 2
    block[17:19] = struct.pack("<H", 2000)  # (0 + 1000) * 2
    decoded = ds110.decode_odid_pack(block, 1)
    assert decoded["altitude"] == 118
    assert decoded["height"] == 0
    assert decoded["lat"] == 42
    assert decoded["lon"] == 12
