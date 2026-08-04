"""
Flask application integration tests for the proximity feature.
Verifies blueprint registration, API routes, and engine lifecycle.
Uses a simplified approach: test the routes directly without full app startup.
"""
import json
import time
import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# The Flask app imports many hardware services at module level.
# To test routes without hardware, we create a minimal Flask app
# with only the proximity blueprint.

from flask import Flask
from services.proximity.config import (
    get_proximity_config, update_proximity_config,
    get_traffic_config, update_traffic_config,
    PROXIMITY_DEFAULTS, TRAFFIC_DEFAULTS,
)
from services.proximity.engine import ProximityEngine, ProximitySnapshot, SourceHealth


@pytest.fixture
def test_app():
    """Minimal Flask app with proximity blueprint only."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    from routes.proximity import proximity_bp
    app.register_blueprint(proximity_bp)

    return app


@pytest.fixture
def client(test_app):
    return test_app.test_client()


# Mock SETTINGS to avoid needing the actual config file
@pytest.fixture(autouse=True)
def mock_settings():
    mock_settings_dict = {"traffic": {}, "map": {"default_lat": 41.9, "default_lon": 12.5, "default_zoom": 7}}
    with patch("services.proximity.config.SETTINGS", mock_settings_dict, create=True):
        # Also patch the config module import inside config.py
        with patch.dict("sys.modules", {"config": MagicMock(SETTINGS=mock_settings_dict, save_settings=MagicMock())}):
            yield mock_settings_dict


class TestBlueprintRegistration:
    """Verify proximity routes are registered and reachable."""

    def test_status_route_exists(self, client):
        res = client.get("/api/proximity/status")
        assert res.status_code == 200

    def test_config_get_route_exists(self, client):
        res = client.get("/api/proximity/config")
        assert res.status_code == 200

    def test_config_post_route_exists(self, client):
        res = client.post(
            "/api/proximity/config",
            data=json.dumps({"enabled": True}),
            content_type="application/json",
        )
        assert res.status_code == 200

    def test_traffic_settings_route_exists(self, client):
        res = client.get("/api/settings/traffic")
        assert res.status_code == 200


class TestProximityStatus:
    """Verify /api/proximity/status behavior."""

    def test_returns_json(self, client):
        res = client.get("/api/proximity/status")
        data = res.get_json()
        assert data is not None

    def test_response_schema(self, client):
        res = client.get("/api/proximity/status")
        data = res.get_json()
        assert "enabled" in data
        assert "pairs" in data
        assert "source_health" in data
        assert "drones_active" in data
        assert "targets_active" in data
        assert "calculation_time_ms" in data
        assert "last_calculated" in data

    def test_pairs_is_list(self, client):
        res = client.get("/api/proximity/status")
        data = res.get_json()
        assert isinstance(data["pairs"], list)

    def test_does_not_trigger_network(self, client):
        """API must return instantly without network requests."""
        start = time.time()
        client.get("/api/proximity/status")
        elapsed = time.time() - start
        assert elapsed < 1.0


class TestProximityConfig:
    """Verify /api/proximity/config behavior."""

    def test_returns_defaults_when_no_section(self, client):
        res = client.get("/api/proximity/config")
        data = res.get_json()
        assert data["enabled"] is True
        assert "thresholds" in data
        assert data["thresholds"]["monitor_entry_m"] == 3000
        assert data["evaluation_radius_m"] == 10000

    def test_invalid_thresholds_rejected(self, client):
        res = client.post(
            "/api/proximity/config",
            data=json.dumps({
                "thresholds": {
                    "monitor_entry_m": 4000,
                    "monitor_exit_m": 3000,
                }
            }),
            content_type="application/json",
        )
        assert res.status_code == 400
        data = res.get_json()
        assert data["success"] is False

    def test_valid_update_succeeds(self, client):
        res = client.post(
            "/api/proximity/config",
            data=json.dumps({"evaluation_radius_m": 8000}),
            content_type="application/json",
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True


class TestTrafficSettings:
    """Verify /api/settings/traffic behavior."""

    def test_returns_defaults(self, client):
        res = client.get("/api/settings/traffic")
        data = res.get_json()
        assert "adsb_net_enabled" in data
        assert "remoteid_enabled" in data

    def test_adsb_net_disabled_by_default(self, client):
        res = client.get("/api/settings/traffic")
        data = res.get_json()
        assert data["adsb_net_enabled"] is False

    def test_update_adsb_net_enabled(self, client):
        res = client.post(
            "/api/settings/traffic",
            data=json.dumps({"adsb_net_enabled": True}),
            content_type="application/json",
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True


class TestEngineIntegration:
    """Verify engine behavior via API."""

    def test_initial_snapshot_disabled(self):
        """Engine not started returns disabled snapshot."""
        engine = ProximityEngine()
        snap = engine.get_snapshot()
        assert snap.enabled is False
        assert snap.pairs == []

    def test_idempotent_start(self):
        """Multiple start calls don't create duplicate threads."""
        engine = ProximityEngine()
        engine._started = True
        engine.start()  # should no-op
        assert engine._thread is None

    def test_snapshot_non_blocking(self):
        """get_snapshot never blocks."""
        engine = ProximityEngine()
        start = time.time()
        engine.get_snapshot()
        assert (time.time() - start) < 0.01
