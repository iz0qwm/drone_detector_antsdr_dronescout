"""Opt-in private upload; independent collector and one latest-only HTTP worker."""
import json
import os
import socket
import threading
import time
from pathlib import Path

from services.field_scene import NODE_ID, default_collector

ENDPOINT = "https://europe-west8-droneskycheck-d0136.cloudfunctions.net/ingestFieldOperation"
CONFIG_PATH = "/etc/tracker-mini/field-operations.json"


class FieldSender:
    def __init__(self, collector, post, token, log=lambda *args, **kwargs: None, interval=5):
        self.collector, self.post, self.token, self.log = collector, post, token, log
        self.interval = interval
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.threads = []
        self.last_error = None

    def collect_once(self):
        snapshot = self.collector.collect()
        with self.lock:
            self.latest = snapshot
        self.ready.set()
        if snapshot["status"] == "ERROR":
            self._report("SCENE_READ_FAILED")
        return snapshot

    def _report(self, code):
        if code != self.last_error:
            self.log("FIELD", code, level="WARNING")
        self.last_error = code

    def send_once(self):
        with self.lock:
            snapshot = self.latest
        if snapshot is None or time.time() * 1000 - snapshot["sampledAt"] > 15000:
            return False
        try:
            # Stream/no body read prevents an unbounded trickling response body.
            response = self.post(ENDPOINT, json=snapshot,
                                 headers={"Authorization": "Bearer " + self.token},
                                 timeout=(2, 3), allow_redirects=False, stream=True)
            try:
                if response.status_code != 200:
                    self._report("UPLOAD_HTTP_" + str(response.status_code))
                    return False
            finally:
                response.close()
            if snapshot["status"] == "OK":
                self.last_error = None
            return True
        except Exception:
            self._report("UPLOAD_UNAVAILABLE")
            return False

    def _collect_loop(self):
        while not self.stop_event.is_set():
            self.collect_once()
            self.stop_event.wait(self.interval)

    def _send_loop(self):
        delay = self.interval
        while not self.stop_event.is_set():
            if not self.ready.wait(1):
                continue
            self.ready.clear()
            success = self.send_once()
            if success:
                delay = self.interval
            else:
                self.stop_event.wait(delay)
                delay = min(30, delay * 2)

    def start(self):
        # All filesystem/network work stays off the application startup thread.
        self.threads = [threading.Thread(target=fn, daemon=True, name=name)
                        for fn, name in ((self._collect_loop, "field-collector"),
                                         (self._send_loop, "field-uploader"))]
        for thread in self.threads:
            thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        self.ready.set()


_sender = None


def start_field_sender():
    global _sender
    if _sender is not None:
        return _sender
    from services.logger import log
    try:
        path = Path(os.environ.get("FIELD_OPERATIONS_CONFIG", CONFIG_PATH))
        if not path.exists():
            return None
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("enabled") is not True:
            return None
        token = config.get("token", "")
        if not isinstance(token, str) or not 32 <= len(token) <= 256 or not token.isascii() or any(c.isspace() for c in token):
            raise ValueError("invalid_token")
        from services.dsc_settings import get_dsc_settings
        if (get_dsc_settings().get("node_id") or socket.gethostname()) != NODE_ID:
            raise ValueError("identity_mismatch")
        import requests
        _sender = FieldSender(default_collector(), requests.post, token, log).start()
        log("FIELD", "Private operational-area sender started")
        return _sender
    except Exception:
        log("FIELD", "Private sender disabled: check configuration and node identity", level="ERROR")
        return None
