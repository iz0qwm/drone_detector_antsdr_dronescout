"""
Meshtastic route tests.

These tests keep the serial interface mocked. They verify that the HTTP
enable control updates persistent traffic configuration before starting the
worker, matching the real service guard in meshtastic_service.start().
"""
import importlib
import sys
import types

import pytest
from flask import Flask


@pytest.fixture
def meshtastic_client(monkeypatch):
    settings = {
        "traffic": {
            "meshtastic_enabled": False,
        }
    }
    saved = []

    def save_settings():
        saved.append(
            dict(settings["traffic"])
        )

    fake_config = types.SimpleNamespace(
        SETTINGS=settings,
        save_settings=save_settings,
    )

    fake_service = types.SimpleNamespace(
        running=False,
        clear_called=False,
    )

    def start():
        if settings["traffic"].get(
            "meshtastic_enabled",
            False,
        ):
            fake_service.running = True

    def stop():
        fake_service.running = False

    def clear_nodes():
        fake_service.clear_called = True

    fake_service.start = start
    fake_service.stop = stop
    fake_service.clear_nodes = clear_nodes
    fake_service.get_nodes = lambda: []
    fake_service.is_alive = lambda: False
    fake_service.get_gateway_info = lambda: {
        "connected": fake_service.running,
    }

    monkeypatch.setitem(
        sys.modules,
        "config",
        fake_config,
    )
    monkeypatch.setitem(
        sys.modules,
        "services.meshtastic_service",
        fake_service,
    )
    monkeypatch.setitem(
        sys.modules,
        "services.notification_service",
        types.SimpleNamespace(),
    )
    sys.modules.pop(
        "routes.meshtastic",
        None,
    )

    module = importlib.import_module(
        "routes.meshtastic"
    )

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(
        module.meshtastic_bp
    )

    return (
        app.test_client(),
        settings,
        saved,
        fake_service,
    )


def test_enable_persists_config_before_starting_service(
    meshtastic_client,
):
    client, settings, saved, service = meshtastic_client

    res = client.post(
        "/api/meshtastic/enable",
        json={
            "enabled": True,
        },
    )

    data = res.get_json()

    assert res.status_code == 200
    assert data["success"] is True
    assert data["configured"] is True
    assert data["enabled"] is True
    assert service.running is True
    assert settings["traffic"]["meshtastic_enabled"] is True
    assert saved[-1]["meshtastic_enabled"] is True


def test_disable_persists_config_and_clears_nodes(
    meshtastic_client,
):
    client, settings, saved, service = meshtastic_client
    settings["traffic"]["meshtastic_enabled"] = True
    service.running = True

    res = client.post(
        "/api/meshtastic/enable",
        json={
            "enabled": False,
        },
    )

    data = res.get_json()

    assert res.status_code == 200
    assert data["success"] is True
    assert data["configured"] is False
    assert data["enabled"] is False
    assert service.running is False
    assert service.clear_called is True
    assert settings["traffic"]["meshtastic_enabled"] is False
    assert saved[-1]["meshtastic_enabled"] is False


def test_status_reports_configured_and_running_state(
    meshtastic_client,
):
    client, settings, _saved, service = meshtastic_client
    settings["traffic"]["meshtastic_enabled"] = True
    service.running = False

    res = client.get(
        "/api/meshtastic/status"
    )

    data = res.get_json()

    assert res.status_code == 200
    assert data["ok"] is True
    assert data["configured"] is True
    assert data["enabled"] is False
