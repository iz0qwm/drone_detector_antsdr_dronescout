"""
Meshtastic operator freshness tests.

These tests use mocked Meshtastic node data. They verify marker lifecycle
metadata only and do not validate physical Meshtastic radio operation.
"""
from datetime import datetime, timezone
from pathlib import Path
import importlib
import sys
import types


def _iso_timestamp(epoch_seconds):
    return datetime.fromtimestamp(
        epoch_seconds,
        tz=timezone.utc
    ).isoformat()


def _load_teams(monkeypatch, nodes, team):
    sys.modules.pop("services.teams", None)

    config = types.ModuleType("config")
    config.BASE_DIR = Path(".")
    config.SETTINGS = {
        "meshtastic": {
            "operator_stale_ms": 600000,
            "operator_retention_ms": 1800000,
        }
    }
    monkeypatch.setitem(sys.modules, "config", config)

    fake_meshtastic = types.ModuleType(
        "services.meshtastic_service"
    )
    fake_meshtastic.get_nodes = lambda: nodes
    fake_meshtastic.get_gateway_info = lambda: {
        "node_num": 1,
    }
    monkeypatch.setitem(
        sys.modules,
        "services.meshtastic_service",
        fake_meshtastic,
    )

    if "services" in sys.modules:
        monkeypatch.setattr(
            sys.modules["services"],
            "meshtastic_service",
            fake_meshtastic,
            raising=False,
        )

    module = importlib.import_module(
        "services.teams"
    )

    monkeypatch.setattr(
        module,
        "load_team",
        lambda: team,
    )
    monkeypatch.setattr(
        module,
        "save_team",
        lambda data: None,
    )

    return module


def test_operator_freshness_marks_stale_before_retention(
    monkeypatch,
):
    now = 1000000.0

    team = {
        "operators": [
            {
                "id": 1,
                "longName": "Operator One",
                "shortName": "OP1",
                "nodeId": "!op1",
            }
        ]
    }
    nodes = [
        {
            "id": "!op1",
            "num": 10,
            "shortName": "OP1",
            "lat": 41.0,
            "lon": 12.0,
            "last_seen": _iso_timestamp(now - 700),
        }
    ]

    teams = _load_teams(
        monkeypatch,
        nodes,
        team,
    )
    monkeypatch.setattr(
        teams.time,
        "time",
        lambda: now,
    )

    status = teams.get_team_status()

    assert status["operator_freshness"] == {
        "stale_ms": 600000,
        "retention_ms": 1800000,
    }
    assert len(status["operators"]) == 1
    assert status["operators"][0]["stale"] is True
    assert status["operators"][0]["expired"] is False
    assert status["operators"][0]["age_ms"] == 700000


def test_operator_freshness_removes_after_retention(
    monkeypatch,
):
    now = 1000000.0

    team = {
        "operators": [
            {
                "id": 1,
                "longName": "Operator One",
                "shortName": "OP1",
                "nodeId": "!op1",
            }
        ]
    }
    nodes = [
        {
            "id": "!op1",
            "num": 10,
            "shortName": "OP1",
            "lat": 41.0,
            "lon": 12.0,
            "last_seen": _iso_timestamp(now - 1900),
        }
    ]

    teams = _load_teams(
        monkeypatch,
        nodes,
        team,
    )
    monkeypatch.setattr(
        teams.time,
        "time",
        lambda: now,
    )

    status = teams.get_team_status()

    assert status["operators"] == []
