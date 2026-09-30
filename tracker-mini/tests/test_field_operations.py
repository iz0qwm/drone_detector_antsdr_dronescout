"""No Pi/network required. Standard unittest, also collected by pytest."""
import copy
import ast
import json
import importlib.util
import sys
import tempfile
import threading
import time
import unittest
import types
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from services.field_scene import SceneCollector, normalize, normalize_rid_targets, normalize_adsb_targets, normalize_team_operators, default_collector, SERIAL
from services.field_sender import FieldSender, start_field_sender
from services import field_adsb, layer_storage, mission_storage


def layer(shape="Polygon"):
    feature = {"type": "Feature", "properties": {"leafletType": shape},
               "geometry": {"type": "Polygon", "coordinates": [[[12, 42], [13, 42], [13, 43], [12, 42]]]}}
    if shape == "Circle":
        feature["geometry"] = {"type": "Point", "coordinates": [12, 42]}
        feature["properties"]["radius"] = 125
    return {"id": "layer_1", "name": "Saved area", "style": {"color": "#12ab34"},
            "properties": {}, "geojson": feature}


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.mission = {"id": "mission_001", "name": "Field mission"}
        self.layers = [layer()]
        self.collector = SceneCollector(lambda: self.mission, lambda _: self.layers,
                                        lambda: self.mission["id"] if self.mission else None)

    def test_selected_mission_and_polygon(self):
        result = self.collector.collect()
        self.assertEqual(result["operation"], self.mission)
        area = result["areas"][0]
        self.assertEqual(area["id"], SERIAL + ":mission_001:layer_1:0")
        self.assertEqual(area["geometry"]["coordinates"][0][0], [12, 42])
        self.assertEqual(area["style"]["color"], "#12ab34")
        self.assertNotIn("altitude", area)

    def test_no_selected_mission(self):
        self.mission = None
        result = self.collector.collect()
        self.assertEqual(result["status"], "OK")
        self.assertIsNone(result["operation"])
        self.assertEqual(result["areas"], [])

    def test_configured_meshtastic_operator_is_separate_from_traffic_and_area(self):
        sampled = 2000000000000
        self.collector.now = lambda: sampled / 1000
        operator = {"id": 7, "nodeId": "!A1B2C3D4", "longName": "Soccorritore Vescovio",
                    "shortName": "SV01", "position_lat": 41.9, "position_lon": 12.5,
                    "position_altitude": 80, "position_observed_at": "2033-05-18T03:33:19+00:00"}
        self.collector.team_status = lambda: {"operators": [operator],
                                              "external_nodes": [{"nodeId": "!11111111", "position_lat": 42}]}
        result = self.collector.collect()
        self.assertEqual(result["team"], [{"id": "mesh:!a1b2c3d4", "nodeId": "!a1b2c3d4",
                                           "name": "Soccorritore Vescovio", "shortName": "SV01",
                                           "role": "TEAM_OPERATOR", "origin": "LIVE", "source": "MESHTASTIC",
                                           "position": {"lat": 41.9, "lon": 12.5}, "observedAt": sampled - 1000,
                                           "altitude": {"value": 80, "unit": "m", "reference": "UNKNOWN"}}])
        self.assertEqual(result["targets"], [])
        self.collector.team_status = lambda: {"operators": [], "external_nodes": [operator]}
        self.assertEqual(self.collector.collect()["team"], [])

    def test_meshtastic_position_time_and_ambiguous_association(self):
        sampled = 2000000000000
        base = {"id": 7, "nodeId": "!a1b2c3d4", "longName": "Operator One", "shortName": "OP01",
                "position_lat": 41.9, "position_lon": 12.5,
                "position_observed_at": "2033-05-18T03:33:19+00:00"}
        self.assertEqual(len(normalize_team_operators({"operators": [base]}, sampled)), 1)
        self.assertEqual(normalize_team_operators({"operators": [{**base, "position_lat": None}]}, sampled), [])
        self.assertEqual(normalize_team_operators({"operators": [{**base, "position_observed_at": "2033-05-18T03:03:19+00:00", "last_seen": sampled}]}, sampled), [])
        self.assertEqual(normalize_team_operators({"operators": [base, {**base, "nodeId": "!11111111"}]}, sampled), [])
        self.assertEqual(normalize_team_operators({"operators": []}, sampled), [])

    def test_meshtastic_non_position_packet_and_poll_do_not_refresh_position(self):
        source = Path(__file__).resolve().parents[1] / "backend" / "services" / "meshtastic_service.py"
        fake_gps = types.ModuleType("services.gps")
        fake_gps.get_gps_status = lambda: {}
        fake_meshtastic = types.ModuleType("meshtastic")
        fake_serial = types.ModuleType("meshtastic.serial_interface")
        fake_serial.SerialInterface = object
        fake_pubsub = types.ModuleType("pubsub")
        fake_pubsub.pub = types.SimpleNamespace()
        modules = {"services.gps": fake_gps, "meshtastic": fake_meshtastic,
                   "meshtastic.serial_interface": fake_serial, "pubsub": fake_pubsub}
        spec = importlib.util.spec_from_file_location("meshtastic_position_under_test", source)
        service = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(service)
        service.log = lambda *args: None
        service.record_text_packet = lambda *args: None
        observed = int(time.time()) - 10
        node = {"user": {"longName": "Operator", "shortName": "OP01"},
                "position": {"latitude": 41.9, "longitude": 12.5}, "lastHeard": observed}
        interface = types.SimpleNamespace(nodes={"!a1b2c3d4": node})
        service.on_receive({"fromId": "!a1b2c3d4", "rxTime": observed + 1,
                            "decoded": {"portnum": "POSITION_APP", "position": {
                                "latitude": 41.9, "longitude": 12.5, "timestamp": observed}}}, interface)
        original = service.meshtastic_nodes["!a1b2c3d4"]["position_observed_at"]
        self.assertEqual(original, datetime.fromtimestamp(observed, timezone.utc).isoformat())
        node["lastHeard"] = int(time.time())
        service.on_receive({"fromId": "!a1b2c3d4", "decoded": {"portnum": "TEXT_MESSAGE_APP", "text": "hello"}}, interface)
        service.update_node_from_meshtastic("!a1b2c3d4", node)
        self.assertEqual(service.meshtastic_nodes["!a1b2c3d4"]["position_observed_at"], original)

    def test_live_rid_outside_area_is_stable_and_geometric_altitude_is_preserved(self):
        seen = "2033-05-18T03:33:20+00:00"
        sampled = 2000000000000
        aircraft = {"source": "RemoteID", "serial": "TEST1596A34", "lat": 41.0, "lon": 11.0,
                    "position_observed_at": seen, "last_seen": seen,
                    "altitude": 118, "height": 0, "heading": 65, "speed": 0}
        self.collector.now = lambda: sampled / 1000
        self.collector.traffic = lambda: [aircraft]
        first = self.collector.collect()
        self.assertEqual(first["status"], "OK")
        self.assertEqual(len(first["targets"]), 1)
        target = first["targets"][0]
        self.assertEqual(target["id"], "rid:remoteid:TEST1596A34")
        self.assertEqual(target["position"], {"lat": 41.0, "lon": 11.0})
        self.assertEqual(target["altitude"], {"value": 118, "unit": "m", "reference": "WGS84_ELLIPSOID"})
        self.assertNotIn("height", target)
        self.assertEqual(self.collector.collect()["targets"][0]["id"], target["id"])
        self.mission = None
        self.assertEqual(self.collector.collect()["targets"][0]["id"], target["id"])

    def test_private_rid_uses_position_time_not_generic_packet_age(self):
        sampled = 2000000000000
        self.collector.now = lambda: sampled / 1000
        aircraft = {"source": "RemoteID", "serial": "TEST1", "lat": 42.0, "lon": 12.0,
                    "position_observed_at": "2033-05-18T03:33:19+00:00",
                    "last_seen": "2033-05-18T03:33:20+00:00", "altitude": -1000, "height": 0}
        self.collector.traffic = lambda: [aircraft]
        target = self.collector.collect()["targets"][0]
        self.assertNotIn("altitude", target)
        self.assertEqual(target["observedAt"], sampled - 1000)
        aircraft["last_seen"] = "2033-05-18T03:33:30+00:00"
        self.collector.now = lambda: (sampled + 11000) / 1000
        self.assertEqual(self.collector.collect()["targets"], [])
        aircraft["position_observed_at"] = "2033-05-18T03:33:31+00:00"
        self.assertEqual(self.collector.collect()["targets"][0]["id"], target["id"])

    def test_private_rid_list_is_bounded_and_ignores_missing_location_time(self):
        self.collector.now = lambda: 2000000000
        base = {"source": "RemoteID", "lat": 42.0, "lon": 12.0,
                "position_observed_at": "2033-05-18T03:33:20+00:00"}
        self.collector.traffic = lambda: [{**base, "serial": f"TEST{index:03d}"} for index in range(40)] + [
            {"source": "RemoteID", "serial": "MISSING", "lat": 42.0, "lon": 12.0}]
        result = self.collector.collect()
        self.assertEqual(result["status"], "OK")
        self.assertEqual(len(result["targets"]), 32)
        self.assertEqual(result["targets"][0]["id"], "rid:remoteid:TEST000")
        self.assertEqual(result["targets"][-1]["id"], "rid:remoteid:TEST031")

    def test_dji_rid_source_has_stable_distinct_identity(self):
        self.collector.now = lambda: 2000000000
        self.collector.traffic = lambda: [{"source": "DJI DroneID", "serial": "TEST1", "lat": 42.0,
                                           "lon": 12.0, "position_observed_at": "2033-05-18T03:33:20+00:00"}]
        target = self.collector.collect()["targets"][0]
        self.assertEqual(target["id"], "rid:dji:TEST1")
        self.assertEqual(target["sourceEpoch"], "dsc-node02_dji")

    def test_local_adsb_coexists_with_rid_outside_area_and_preserves_motion(self):
        sampled = 2000000000000
        self.collector.now = lambda: sampled / 1000
        self.collector.traffic = lambda: [{"source": "RemoteID", "serial": "TEST1", "lat": 42.0, "lon": 12.0,
                                           "position_observed_at": "2033-05-18T03:33:20+00:00"}]
        self.collector.air_traffic = lambda: {"now": sampled / 1000, "aircraft": [
            {"hex": "Ab12Cd", "type": "adsb_icao", "flight": " EJU624Q  ", "lat": 41.0, "lon": 11.0,
             "seen_pos": 1.0, "alt_geom": 1000, "alt_baro": 900, "gs": 100, "track": 124}]}
        targets = self.collector.collect()["targets"]
        self.assertEqual([target["id"] for target in targets], ["rid:remoteid:TEST1", "adsb:ab12cd"])
        aircraft = targets[1]
        self.assertEqual(aircraft["name"], "EJU624Q")
        self.assertEqual(aircraft["observedAt"], sampled - 1000)
        self.assertEqual(aircraft["altitude"], {"value": 304.8, "unit": "m", "reference": "WGS84_ELLIPSOID"})
        self.assertAlmostEqual(aircraft["speedMps"], 51.4444)
        self.assertEqual(aircraft["heading"], 124)
        self.assertEqual(aircraft["source"], "ADSBRx")

    def test_adsb_file_reread_never_renews_position_and_same_icao_recovers(self):
        sampled = 2000000000000
        self.collector.now = lambda: sampled / 1000
        readsb = {"now": sampled / 1000, "aircraft": [
            {"hex": "ABC123", "type": "adsb_icao", "lat": 42.0, "lon": 12.0, "seen_pos": 1, "alt_baro": 500}]}
        self.collector.air_traffic = lambda: readsb
        first = self.collector.collect()["targets"][0]
        self.assertEqual(first["id"], "adsb:abc123")
        self.assertEqual(first["name"], "ABC123")
        self.assertEqual(first["altitude"], {"value": 152.4, "unit": "m", "reference": "BARO"})
        self.collector.now = lambda: (sampled + 5000) / 1000
        self.assertEqual(self.collector.collect()["targets"][0]["observedAt"], first["observedAt"])
        readsb["now"] = (sampled + 5000) / 1000
        readsb["aircraft"][0]["seen_pos"] = 6
        self.assertEqual(self.collector.collect()["targets"][0]["observedAt"], first["observedAt"])
        self.collector.now = lambda: (sampled + 11000) / 1000
        self.assertEqual(self.collector.collect()["targets"], [])
        readsb["now"] = (sampled + 11000) / 1000
        readsb["aircraft"][0]["seen_pos"] = 0
        self.assertEqual(self.collector.collect()["targets"][0]["id"], first["id"])

    def test_adsb_altitude_unknown_zero_and_explicit_rotorcraft(self):
        now = 2000000000000
        base = {"type": "adsb_icao", "lat": 42.0, "lon": 12.0, "seen_pos": 0}
        data = {"now": now / 1000, "aircraft": [
            {**base, "hex": "000001", "alt_geom": 0, "alt_baro": 900},
            {**base, "hex": "000002", "alt_geom": "bad", "alt_baro": 0},
            {**base, "hex": "000003", "alt_baro": "ground"},
            {**base, "hex": "000004", "category": "A7", "alt_baro": 4000},
            {**base, "hex": "000005", "alt_baro": 4000},
            {**base, "hex": "~000006", "alt_geom": 100}]}
        targets = {target["id"]: target for target in normalize_adsb_targets(data, now)}
        self.assertEqual(targets["adsb:000001"]["altitude"], {"value": 0, "unit": "m", "reference": "WGS84_ELLIPSOID"})
        self.assertEqual(targets["adsb:000002"]["altitude"], {"value": 0, "unit": "m", "reference": "BARO"})
        self.assertNotIn("altitude", targets["adsb:000003"])
        self.assertEqual(targets["adsb:000004"]["targetClass"], "ROTORCRAFT")
        self.assertNotIn("adsb:000005", targets)
        self.assertNotIn("adsb:000006", targets)

    def test_adsb_source_checks_and_combined_target_bound(self):
        now = 2000000000000
        data = {"now": now / 1000, "aircraft": [
            {"hex": f"{index:06x}", "type": "adsb_icao", "lat": 42.0, "lon": 12.0,
             "seen_pos": 1} for index in range(40)] + [
            {"hex": "ffffff", "type": "mlat", "lat": 42.0, "lon": 12.0, "seen_pos": 0}]}
        self.assertEqual(len(normalize_adsb_targets(data, now)), 32)
        self.assertEqual(normalize_adsb_targets({"now": now / 1000, "aircraft": [
            {"hex": "abc123", "type": "adsb_icao", "lat": 42.0, "lon": 12.0, "seen_pos": 11}]}, now), [])
        self.assertEqual(normalize_adsb_targets({"now": None, "aircraft": data["aircraft"]}, now), [])
        self.assertEqual(normalize_adsb_targets({"now": now / 1000 + 30, "aircraft": data["aircraft"]}, now), [])
        for invalid in ({"lat": 91}, {"lon": 181}, {"lat": 0, "lon": 0}, {"seen_pos": -1},
                        {"type": "mlat"}, {"type": "adsb_icao_nt"}):
            item = {"hex": "abc123", "type": "adsb_icao", "lat": 42.0, "lon": 12.0, "seen_pos": 0, **invalid}
            self.assertEqual(normalize_adsb_targets({"now": now / 1000, "aircraft": [item]}, now), [])
        self.collector.now = lambda: now / 1000
        self.collector.traffic = lambda: [{"source": "RemoteID", "serial": "TEST1", "lat": 42.0, "lon": 12.0,
                                           "position_observed_at": "2033-05-18T03:33:20+00:00"}]
        self.collector.air_traffic = lambda: data
        combined = self.collector.collect()["targets"]
        self.assertEqual(len(combined), 32)
        self.assertEqual(combined[0]["id"], "rid:remoteid:TEST1")

    def test_local_readsb_adapter_handles_file_and_missing_device(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aircraft.json"
            with patch.object(field_adsb, "READSB_JSON", str(path)):
                self.assertEqual(field_adsb.read_local_readsb()["aircraft"], [])
                path.write_text(json.dumps({"now": 2000000000, "aircraft": [{"hex": "abc123"}]}), encoding="utf-8")
                self.assertEqual(field_adsb.read_local_readsb()["aircraft"][0]["hex"], "abc123")
                path.write_text("{broken", encoding="utf-8")
                self.assertEqual(field_adsb.read_local_readsb()["aircraft"], [])
                path.write_bytes(b" " * (field_adsb.MAX_READSB_BYTES + 1))
                self.assertEqual(field_adsb.read_local_readsb()["aircraft"], [])

    def test_rectangle(self):
        self.layers = [layer("Rectangle")]
        self.assertEqual(self.collector.collect()["areas"][0]["geometry"]["type"], "Polygon")

    def test_circle_radius_retained(self):
        self.layers = [layer("Circle")]
        self.assertEqual(self.collector.collect()["areas"][0]["geometry"],
                         {"type": "Circle", "center": [12, 42], "radiusM": 125})

    def test_feature_collection_stable_children(self):
        self.layers[0]["geojson"] = {"type": "FeatureCollection", "features": [layer()["geojson"], layer("Circle")["geojson"]]}
        first = self.collector.collect()["areas"]
        self.assertEqual(len(first), 2)
        self.assertTrue(first[1]["id"].endswith(":1"))
        self.assertEqual(first, self.collector.collect()["areas"])

    def test_dsc_and_other_imports_excluded(self):
        for source in ("dsc", "import", "other"):
            self.layers[0]["properties"]["source"] = source
            self.assertEqual(self.collector.collect()["areas"], [])

    def test_poi_excluded(self):
        self.layers[0]["geojson"] = {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [12, 42]}}
        self.assertEqual(self.collector.collect()["areas"], [])

    def test_invalid_geometry_reported_without_partial_empty_success(self):
        self.layers.append(layer())
        self.layers[1]["geojson"]["geometry"]["coordinates"][0][0] = [181, 42]
        result = self.collector.collect()
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["errorCode"], "SCENE_READ_FAILED")
        self.assertNotIn("areas", result)

    def test_rename_and_edit(self):
        before = self.collector.collect()["areas"][0]
        self.layers[0]["name"] = "Renamed"
        self.layers[0]["geojson"] = layer("Circle")["geojson"]
        after = self.collector.collect()["areas"][0]
        self.assertEqual(after["id"], before["id"])
        self.assertEqual(after["name"], "Renamed")
        self.assertEqual(after["geometry"]["type"], "Circle")

    def test_delete_and_successful_empty(self):
        self.assertEqual(len(self.collector.collect()["areas"]), 1)
        self.layers.clear()
        self.assertEqual(self.collector.collect()["areas"], [])
        self.assertEqual(self.collector.collect()["status"], "OK")

    def test_mission_switch_replaces_ids(self):
        before = self.collector.collect()["areas"][0]["id"]
        self.mission = {"id": "mission_002", "name": "Second"}
        after = self.collector.collect()
        self.assertNotEqual(before, after["areas"][0]["id"])
        self.assertEqual(after["operation"]["name"], "Second")

    def test_read_failure_not_empty(self):
        self.collector.layers = Mock(side_effect=OSError("private path"))
        self.assertEqual(self.collector.collect()["status"], "ERROR")
        self.assertNotIn("private path", json.dumps(self.collector.collect()))

    def test_concurrent_selection_and_missing_mission(self):
        self.collector.selected = Mock(side_effect=["mission_001", "mission_002"])
        self.assertEqual(self.collector.collect()["status"], "ERROR")
        self.collector.selected = lambda: "mission_001"
        self.mission = None
        self.assertEqual(self.collector.collect()["status"], "ERROR")

    def test_bounds_nonfinite_and_radius(self):
        for bad in ([float("nan"), 42], [12, 91], [True, 42], [12, 42, 10]):
            item = layer("Circle"); item["geojson"]["geometry"]["coordinates"] = bad
            with self.assertRaises(ValueError): normalize(self.mission, [item])
        for radius in (0, -1, 100001, float("inf")):
            item = layer("Circle"); item["geojson"]["properties"]["radius"] = radius
            with self.assertRaises(ValueError): normalize(self.mission, [item])
        with self.assertRaises(ValueError): normalize(self.mission, [layer()] * 101)


class StorageTests(unittest.TestCase):
    def test_actual_selected_saved_storage_create_edit_delete_switch_and_failure(self):
        # Only the unrelated zone-download dependency is stubbed; mission/layer readers are real.
        with patch.dict(sys.modules, {"services.dsc_client": Mock()}):
            from services import missions
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"services.missions": missions}):
            root = Path(directory)
            with patch.object(missions, "MISSIONS_DIR", root), patch.object(layer_storage, "MISSIONS_DIR", root), \
                 patch.object(mission_storage, "MISSIONS_DIR", root), \
                 patch.object(mission_storage, "CURRENT_FILE", root / "current_mission.json"):
                for mid in ("mission_001", "mission_002"):
                    (root / mid / "layers").mkdir(parents=True)
                    (root / mid / "mission.json").write_text(json.dumps({"id": mid, "name": mid}))
                mission_storage.set_current_mission_id("mission_001")
                collector = default_collector()
                saved = layer_storage.save_layer("mission_001", layer())
                self.assertEqual(len(collector.collect()["areas"]), 1)
                saved["name"] = "Saved rename"
                layer_storage.save_layer("mission_001", saved)
                self.assertEqual(collector.collect()["areas"][0]["name"], "Saved rename")
                layer_storage.delete_layer("mission_001", saved["id"])
                self.assertEqual(collector.collect()["areas"], [])
                mission_storage.set_current_mission_id("mission_002")
                self.assertEqual(collector.collect()["operation"]["id"], "mission_002")
                (root / "mission_002" / "layers" / "broken.json").write_text("{")
                self.assertEqual(collector.collect()["status"], "ERROR")
                mission_storage.set_current_mission_id("mission_003")
                self.assertEqual(collector.collect()["status"], "ERROR")
                with patch.object(mission_storage, "MISSIONS_DIR", root / "unavailable"):
                    self.assertEqual(default_collector().collect()["status"], "ERROR")


class SenderTests(unittest.TestCase):
    def make_sender(self, post):
        self.revision = 0
        collector = Mock()
        collector.collect.side_effect = lambda: {"status": "OK", "sampledAt": int(time.time()*1000), "revision": self.revision}
        return FieldSender(collector, post, "test-only-token-" + "x" * 32)

    def test_timeout_does_not_block_collection_and_retry_is_latest_only(self):
        entered, release = threading.Event(), threading.Event()
        sent = []
        def post(url, **kwargs):
            sent.append(copy.deepcopy(kwargs["json"]))
            self.assertEqual(kwargs["timeout"], (2, 3))
            self.assertFalse(kwargs["allow_redirects"])
            entered.set(); release.wait(2)
            raise TimeoutError()
        sender = self.make_sender(post)
        sender.collect_once()
        worker = threading.Thread(target=sender.send_once)
        worker.start()
        self.assertTrue(entered.wait(1))
        for self.revision in (1, 2, 3): sender.collect_once()
        self.assertEqual(sender.latest["revision"], 3)
        release.set(); worker.join(2)
        sender.send_once()
        self.assertEqual([x["revision"] for x in sent], [0, 3])

    def test_worker_one_in_flight_and_collector_keeps_running(self):
        entered, release = threading.Event(), threading.Event()
        def post(*args, **kwargs):
            entered.set(); release.wait(1); return Mock(status_code=200)
        sender = self.make_sender(Mock(side_effect=post)); sender.interval = .02
        sender.start()
        try:
            self.assertTrue(entered.wait(1))
            self.revision = 7
            deadline = time.monotonic() + 1
            while sender.latest["revision"] != 7 and time.monotonic() < deadline: time.sleep(.01)
            self.assertEqual(sender.latest["revision"], 7)
            self.assertEqual(sender.post.call_count, 1)
        finally:
            sender.stop(); release.set()
            for thread in sender.threads: thread.join(2)

    def test_expired_snapshot_not_sent(self):
        post = Mock(); sender = self.make_sender(post)
        sender.latest = {"sampledAt": 1}
        self.assertFalse(sender.send_once()); post.assert_not_called()

    def test_disabled_without_private_config(self):
        with patch("services.field_sender.Path.exists", return_value=False):
            self.assertIsNone(start_field_sender())

    def test_import_verification_does_not_start_private_sender(self):
        source = Path(__file__).resolve().parents[1] / "backend" / "app.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        # Execute only the new startup call/guard, without importing hardware services.
        startup = []
        for statement in tree.body:
            if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "start_field_sender" for n in ast.walk(statement)):
                startup.append(statement)
        isolated = compile(ast.Module(body=startup, type_ignores=[]), "startup", "exec")
        sender = Mock(); app = Mock()
        exec(isolated, {"__name__": "app", "start_field_sender": sender, "app": app})
        sender.assert_not_called()
        exec(isolated, {"__name__": "__main__", "start_field_sender": sender, "app": app})
        sender.assert_called_once()

    def test_invalid_secret_and_wrong_node_disable_startup(self):
        import services.field_sender as module
        for token, node in [("short", "dsc-node02"), ("x" * 40, "other-node")]:
            settings = Mock(); settings.get_dsc_settings.return_value = {"node_id": node}
            with patch.dict(sys.modules, {"services.dsc_settings": settings}), \
                 patch.object(module.Path, "exists", return_value=True), \
                 patch.object(module.Path, "read_text", return_value=json.dumps({"enabled": True, "token": token})), \
                 patch.object(module, "default_collector") as factory, patch("services.logger.log"):
                self.assertIsNone(start_field_sender())
                factory.assert_not_called()

    def test_response_closed_no_secret_logging(self):
        response = Mock(status_code=401); sender = self.make_sender(Mock(return_value=response))
        sender.log = Mock(); sender.collect_once(); self.assertFalse(sender.send_once())
        response.close.assert_called_once()
        self.assertNotIn(sender.token, str(sender.log.call_args))


if __name__ == "__main__":
    unittest.main()
