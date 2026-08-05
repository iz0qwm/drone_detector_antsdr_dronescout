"""
Tests for network ADS-B provider normalization and merge behavior.
Network calls are mocked; these tests do not contact external providers.
"""
import pytest

from services import air_network


BOUNDS = {
    "minLat": 41.8,
    "maxLat": 42.0,
    "minLon": 12.4,
    "maxLon": 12.6,
}


def test_readsb_point_item_normalized():
    item = {
        "hex": "ABC123",
        "flight": " TEST1 ",
        "lat": 41.9,
        "lon": 12.5,
        "alt_baro": 1000,
        "gs": 100,
        "track": 370,
        "category": "A3",
        "seen": 2,
    }

    ac = air_network.aircraft_from_readsb_item(
        item,
        "AIRPLANES_LIVE",
        BOUNDS,
        show_all=False,
        response_now_seconds=1000,
    )

    assert ac["icao"] == "abc123"
    assert ac["callsign"] == "TEST1"
    assert ac["source"] == "AIRPLANES_LIVE"
    assert ac["heading"] == 10
    assert ac["altitude"] == pytest.approx(304.8)
    assert ac["speed"] == pytest.approx(51.4444)
    assert ac["updatedAt"] == 998000


def test_readsb_point_item_filters_stale_tracks():
    item = {
        "hex": "ABC123",
        "lat": 41.9,
        "lon": 12.5,
        "alt_baro": 1000,
        "seen": air_network.READSB_STALE_SECONDS + 1,
    }

    ac = air_network.aircraft_from_readsb_item(
        item,
        "ADSB_LOL",
        BOUNDS,
        show_all=False,
        response_now_seconds=1000,
    )

    assert ac is None


def test_merge_aircraft_combines_sources_and_prefers_newer_position():
    older = {
        "icao": "abc123",
        "callsign": "abc123",
        "lat": 41.9,
        "lon": 12.5,
        "altitude": None,
        "speed": None,
        "heading": 0,
        "isHelicopter": False,
        "source": "AIRPLANES_LIVE",
        "updatedAt": 1000,
    }
    newer = {
        "icao": "abc123",
        "callsign": "REAL1",
        "lat": 41.91,
        "lon": 12.51,
        "altitude": 400,
        "speed": 40,
        "heading": 90,
        "isHelicopter": False,
        "source": "ADSB_LOL",
        "updatedAt": 2000,
    }

    merged = air_network.merge_aircraft([older], [newer])

    assert len(merged) == 1
    assert merged[0]["callsign"] == "REAL1"
    assert merged[0]["lat"] == 41.91
    assert merged[0]["source"] == "ADSB_LOL+AIRPLANES_LIVE"


def test_get_network_aircraft_keeps_solarmonitor_adsb_paused(monkeypatch):
    called = {"solarmonitor": False}

    def paused_fetcher(*_args, **_kwargs):
        called["solarmonitor"] = True
        return []

    monkeypatch.setattr(air_network, "fetch_solarmonitor", paused_fetcher)
    monkeypatch.setattr(air_network, "fetch_airplanes_live", lambda *_args: [])
    monkeypatch.setattr(air_network, "fetch_adsb_lol", lambda *_args: [])
    monkeypatch.setattr(air_network, "fetch_ogn", lambda *_args: [])
    monkeypatch.setattr(air_network, "fetch_opensky", lambda *_args: [])

    result = air_network.get_network_aircraft(BOUNDS, show_all=False)

    assert called["solarmonitor"] is False
    assert "solarmonitor" not in result["sources"]
    assert set(result["sources"]) == {
        "airplanes_live",
        "adsb_lol",
        "ogn",
        "opensky",
    }
