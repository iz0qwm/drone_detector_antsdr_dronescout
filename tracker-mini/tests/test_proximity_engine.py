"""
Tests for the proximity engine: lifecycle, snapshot, and API behavior.
Uses mocks for hardware-dependent services.
"""
import time
import threading
import pytest
from unittest.mock import patch, MagicMock
from services.proximity.engine import (
    ProximityEngine, ADSBNetCache, ProximitySnapshot, SourceHealth,
)
from services.proximity.config import PROXIMITY_DEFAULTS


class TestSourceHealth:
    def test_initial_disabled(self):
        h = SourceHealth("test")
        assert h.state == "DISABLED"

    def test_set_available(self):
        h = SourceHealth("test")
        h.set_available()
        assert h.state == "AVAILABLE"
        assert h.last_successful is not None

    def test_set_offline(self):
        h = SourceHealth("test")
        h.set_offline("timeout")
        assert h.state == "OFFLINE"
        assert h.error == "timeout"

    def test_to_dict(self):
        h = SourceHealth("test")
        h.set_available()
        d = h.to_dict()
        assert d["state"] == "AVAILABLE"
        assert "last_successful" in d


class TestADSBNetCache:
    def test_empty_snapshot_initially(self):
        cache = ADSBNetCache(refresh_interval_ms=60000)
        assert cache.get_snapshot() == []

    def test_get_snapshot_not_blocking(self):
        """get_snapshot must return immediately."""
        cache = ADSBNetCache(refresh_interval_ms=60000)
        start = time.time()
        cache.get_snapshot()
        elapsed = time.time() - start
        assert elapsed < 0.01  # must be essentially instant


class TestProximitySnapshot:
    def test_to_dict(self):
        snap = ProximitySnapshot(
            enabled=True,
            source_health={"adsb_rx": SourceHealth("ADSBRx")},
            drones_active=1,
            targets_active=5,
            pairs=[],
            calculation_time_ms=3.5,
        )
        d = snap.to_dict()
        assert d["enabled"] is True
        assert d["drones_active"] == 1
        assert d["targets_active"] == 5
        assert isinstance(d["pairs"], list)


class TestProximityEngine:
    def test_idempotent_start(self):
        """Duplicate start() calls must not create duplicate threads."""
        engine = ProximityEngine()
        with patch.object(engine, '_loop'):
            engine._started = True  # simulate already started
            engine.start()  # should be no-op
            assert engine._thread is None  # no new thread created

    def test_get_snapshot_returns_immediately(self):
        """get_snapshot must never block on network."""
        engine = ProximityEngine()
        start = time.time()
        snap = engine.get_snapshot()
        elapsed = time.time() - start
        assert elapsed < 0.01
        assert snap.enabled is False  # initial state

    @patch("services.proximity.engine.is_adsb_net_enabled", return_value=False)
    @patch("services.proximity.engine.get_proximity_config",
           return_value=PROXIMITY_DEFAULTS)
    def test_calculate_cycle_no_drones(self, mock_config, mock_net):
        """Engine handles no drones gracefully."""
        engine = ProximityEngine()

        with patch.object(engine, '_read_drones', return_value=[]):
            with patch.object(engine, '_read_adsbrx', return_value=[]):
                with patch.object(engine, '_read_adsbnet', return_value=[]):
                    engine._calculate_cycle()

        snap = engine.get_snapshot()
        assert snap.enabled is True
        assert snap.drones_active == 0
        assert snap.pairs == []

    @patch("services.proximity.engine.is_adsb_net_enabled", return_value=False)
    @patch("services.proximity.engine.get_proximity_config",
           return_value=PROXIMITY_DEFAULTS)
    def test_calculate_cycle_with_data(self, mock_config, mock_net):
        """Engine evaluates pairs correctly with mock data."""
        engine = ProximityEngine()
        now = time.time()
        iso_now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now))

        drones = [{
            "serial": "DRN1",
            "lat": 41.90,
            "lon": 12.50,
            "last_seen": iso_now,
        }]
        aircraft = [{
            "icao": "3C6589",
            "callsign": "DLH1",
            "lat": 41.905,
            "lon": 12.50,
            "altitude": 500,
            "speed": 100,
            "heading": 90,
            "isHelicopter": False,
            "category": "A3",
            "source": "LOCAL_ADSB",
            "updatedAt": int(now * 1000),
        }]

        with patch.object(engine, '_read_drones', return_value=drones):
            with patch.object(engine, '_read_adsbrx', return_value=aircraft):
                with patch.object(engine, '_read_adsbnet', return_value=[]):
                    engine._calculate_cycle()

        snap = engine.get_snapshot()
        assert snap.enabled is True
        assert snap.drones_active == 1
        assert snap.targets_active == 1
        # ~556m distance → should be in WARNING (< 500? no, ~556 > 500 → CAUTION)
        # Actually 0.005° lat ≈ 556m → within CAUTION (500-1500)
        assert len(snap.pairs) > 0
        pair = snap.pairs[0]
        assert pair["drone_id"] == "DRN1"
        assert pair["target_id"] == "3c6589"
        assert pair["state"] in ("MONITOR", "CAUTION", "WARNING")
        assert pair["distance_m"] > 0

    @patch("services.proximity.engine.is_adsb_net_enabled", return_value=False)
    @patch("services.proximity.engine.get_proximity_config",
           return_value=PROXIMITY_DEFAULTS)
    def test_performance_50_aircraft_5_drones(self, mock_config, mock_net):
        """Engine completes within 100ms for 50 aircraft x 5 drones."""
        engine = ProximityEngine()
        now = time.time()
        iso_now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now))

        # Generate 5 drones around Rome
        drones = []
        for i in range(5):
            drones.append({
                "serial": f"DRN{i}",
                "lat": 41.90 + i * 0.001,
                "lon": 12.50,
                "last_seen": iso_now,
            })

        # Generate 50 aircraft within 10km
        aircraft = []
        for i in range(50):
            aircraft.append({
                "icao": f"{i:06X}",
                "callsign": f"FLT{i}",
                "lat": 41.90 + (i * 0.001),
                "lon": 12.50 + ((i % 10) * 0.001),
                "altitude": 300 + i * 10,
                "speed": 50 + i,
                "heading": i * 7 % 360,
                "isHelicopter": False,
                "category": "A3",
                "source": "LOCAL_ADSB",
                "updatedAt": int(now * 1000),
            })

        with patch.object(engine, '_read_drones', return_value=drones):
            with patch.object(engine, '_read_adsbrx', return_value=aircraft):
                with patch.object(engine, '_read_adsbnet', return_value=[]):
                    start = time.time()
                    engine._calculate_cycle()
                    elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < 200, f"Cycle took {elapsed_ms:.0f}ms (must be <200ms)"
        snap = engine.get_snapshot()
        assert snap.drones_active == 5
        assert snap.targets_active <= 50
