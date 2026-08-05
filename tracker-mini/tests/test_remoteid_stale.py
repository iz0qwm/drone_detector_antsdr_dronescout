"""
Remote ID freshness tests.

These tests exercise the DS110 in-memory cache lifecycle only. They do not
validate physical DS110 hardware reception.
"""
from datetime import datetime, timezone
import importlib
import sys
import types


def _load_ds110(monkeypatch):
    sys.modules.pop("services.ds110", None)

    pymavlink = types.ModuleType("pymavlink")
    pymavlink.mavutil = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "pymavlink", pymavlink)

    config = types.ModuleType("config")
    config.SETTINGS = {
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
        "last_seen": _iso_timestamp(now - 20),
    }

    aircraft = {
        item["serial"]: item
        for item in ds110.get_aircraft()
    }

    assert aircraft["fresh"]["stale"] is False
    assert aircraft["fresh"]["age_ms"] == 5000
    assert aircraft["fresh"]["updatedAt"] == int((now - 5) * 1000)
    assert aircraft["stale"]["stale"] is True
    assert aircraft["stale"]["age_ms"] == 20000


def test_get_aircraft_removes_expired_remoteid_tracks(monkeypatch):
    ds110 = _load_ds110(monkeypatch)
    now = 1000000.0
    monkeypatch.setattr(ds110.time, "time", lambda: now)

    ds110.remoteid_aircraft["expired"] = {
        "serial": "expired",
        "lat": 41.0,
        "lon": 12.0,
        "last_seen": _iso_timestamp(now - 80),
    }

    assert ds110.get_aircraft() == []
    assert "expired" not in ds110.remoteid_aircraft
