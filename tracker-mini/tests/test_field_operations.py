"""No Pi/network required. Standard unittest, also collected by pytest."""
import copy
import ast
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from services.field_scene import SceneCollector, normalize, default_collector, SERIAL
from services.field_sender import FieldSender, start_field_sender
from services import layer_storage, mission_storage


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
