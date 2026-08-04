"""
Tests for aircraft normalization and source merge.
"""
import time
import pytest
from services.proximity.normalize import (
    normalize_adsbrx,
    normalize_adsbnet,
    merge_targets,
    NormalizedTarget,
    ProviderResult,
)


class TestNormalizeADSBRx:
    def test_basic_normalization(self):
        aircraft = [{
            "icao": "3C6589",
            "callsign": "DLH123",
            "lat": 41.9,
            "lon": 12.5,
            "altitude": 500,
            "speed": 120.0,
            "heading": 90.0,
            "isHelicopter": False,
            "category": "A3",
            "source": "LOCAL_ADSB",
            "updatedAt": int(time.time() * 1000),
        }]
        targets = normalize_adsbrx(aircraft)
        assert len(targets) == 1
        t = targets[0]
        assert t.track_id == "3c6589"
        assert t.icao == "3C6589"
        assert t.primary_source == "ADSBRx"
        assert "ADSBRx" in t.sources
        assert t.latitude == 41.9
        assert t.source_label == "RX"

    def test_missing_icao_skipped(self):
        aircraft = [{"lat": 41.9, "lon": 12.5, "updatedAt": 1000}]
        targets = normalize_adsbrx(aircraft)
        assert len(targets) == 0

    def test_invalid_position_skipped(self):
        aircraft = [{"icao": "ABC", "lat": None, "lon": 12.5, "updatedAt": 1000}]
        targets = normalize_adsbrx(aircraft)
        assert len(targets) == 0


class TestNormalizeADSBNet:
    def test_basic_normalization(self):
        aircraft = [{
            "icao": "4B1A2E",
            "callsign": "SWR42",
            "lat": 46.0,
            "lon": 8.5,
            "altitude": 800,
            "speed": 200.0,
            "heading": 270.0,
            "isHelicopter": False,
            "category": "A5",
            "source": "OPENSKY",
            "updatedAt": int(time.time() * 1000),
        }]
        targets = normalize_adsbnet(aircraft)
        assert len(targets) == 1
        t = targets[0]
        assert t.track_id == "4b1a2e"
        assert t.primary_source == "ADSBNet"
        assert t.source_label == "NET"
        assert t.altitude_reference == "geo_msl"

    def test_solarmonitor_alt_reference(self):
        aircraft = [{
            "icao": "ABC123",
            "lat": 41.0,
            "lon": 12.0,
            "altitude": 300,
            "source": "SOLARMONITOR_ADSB",
            "updatedAt": int(time.time() * 1000),
        }]
        targets = normalize_adsbnet(aircraft)
        assert targets[0].altitude_reference == "baro_msl"


class TestMergeTargets:
    def test_no_overlap(self):
        rx = [NormalizedTarget(track_id="aaa", latitude=41.0, longitude=12.0,
                               sources=["ADSBRx"], source_timestamps={"ADSBRx": time.time()})]
        net = [NormalizedTarget(track_id="bbb", latitude=42.0, longitude=13.0,
                                sources=["ADSBNet"], source_timestamps={"ADSBNet": time.time()})]
        merged = merge_targets(rx, net)
        assert len(merged) == 2

    def test_duplicate_icao_merged(self):
        now = time.time()
        rx = [NormalizedTarget(track_id="3c6589", icao="3C6589", latitude=41.9, longitude=12.5,
                               primary_source="ADSBRx", sources=["ADSBRx"],
                               source_timestamps={"ADSBRx": now})]
        net = [NormalizedTarget(track_id="3c6589", icao="3C6589", latitude=41.91, longitude=12.51,
                                primary_source="ADSBNet", sources=["ADSBNet"],
                                source_timestamps={"ADSBNet": now - 5})]
        merged = merge_targets(rx, net)
        assert len(merged) == 1
        t = merged[0]
        assert t.sources == ["ADSBRx", "ADSBNet"]
        assert t.source_label == "RX+NET"
        # ADSBRx position preferred (newer)
        assert t.latitude == 41.9
        assert t.primary_source == "ADSBRx"

    def test_newer_adsbnet_wins(self):
        """When ADSBNet is significantly newer, it should win."""
        now = time.time()
        rx = [NormalizedTarget(track_id="abc", latitude=41.0, longitude=12.0,
                               primary_source="ADSBRx", sources=["ADSBRx"],
                               source_timestamps={"ADSBRx": now - 10})]
        net = [NormalizedTarget(track_id="abc", latitude=41.1, longitude=12.1,
                                primary_source="ADSBNet", sources=["ADSBNet"],
                                source_timestamps={"ADSBNet": now})]
        merged = merge_targets(rx, net, tie_window_s=3.0)
        assert len(merged) == 1
        t = merged[0]
        assert t.primary_source == "ADSBNet"
        assert t.latitude == 41.1

    def test_tie_window_prefers_adsbrx(self):
        """Within tie window, ADSBRx is preferred."""
        now = time.time()
        rx = [NormalizedTarget(track_id="abc", latitude=41.0, longitude=12.0,
                               primary_source="ADSBRx", sources=["ADSBRx"],
                               source_timestamps={"ADSBRx": now - 1})]
        net = [NormalizedTarget(track_id="abc", latitude=41.1, longitude=12.1,
                                primary_source="ADSBNet", sources=["ADSBNet"],
                                source_timestamps={"ADSBNet": now})]
        merged = merge_targets(rx, net, tie_window_s=3.0)
        assert merged[0].primary_source == "ADSBRx"
        assert merged[0].latitude == 41.0

    def test_callsign_supplemented(self):
        """ADSBNet callsign fills in when ADSBRx lacks it."""
        now = time.time()
        rx = [NormalizedTarget(track_id="abc", icao="ABC", callsign=None,
                               latitude=41.0, longitude=12.0,
                               sources=["ADSBRx"], source_timestamps={"ADSBRx": now})]
        net = [NormalizedTarget(track_id="abc", icao="ABC", callsign="FLT42",
                                latitude=41.0, longitude=12.0,
                                sources=["ADSBNet"], source_timestamps={"ADSBNet": now - 5})]
        merged = merge_targets(rx, net)
        assert merged[0].callsign == "FLT42"

    def test_adsbnet_only_target_preserved(self):
        """Targets only in ADSBNet (outside local reception) are kept."""
        now = time.time()
        net = [NormalizedTarget(track_id="xyz", latitude=50.0, longitude=10.0,
                                sources=["ADSBNet"], source_timestamps={"ADSBNet": now})]
        merged = merge_targets([], net)
        assert len(merged) == 1
        assert merged[0].track_id == "xyz"


class TestProviderResult:
    def test_empty_successful(self):
        """Empty aircraft list with successful=True is valid (empty sky)."""
        pr = ProviderResult(
            provider="OpenSky",
            successful=True,
            aircraft=[],
            fetch_timestamp=time.time(),
            response_duration_ms=150,
        )
        assert pr.successful is True
        assert len(pr.aircraft) == 0

    def test_failure(self):
        pr = ProviderResult(
            provider="SolarMonitor",
            successful=False,
            aircraft=[],
            fetch_timestamp=time.time(),
            response_duration_ms=4000,
            error="Timeout",
        )
        assert pr.successful is False
        assert pr.error == "Timeout"
