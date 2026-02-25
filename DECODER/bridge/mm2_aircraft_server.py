#!/usr/bin/env python3
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

MSG_PORT = 30003
HTTP_BIND = "127.0.0.1"
HTTP_PORT = 9090

# TTL: se un aereo non aggiorna nulla per X secondi, sparisce dal JSON
AIRCRAFT_TTL_SEC = 180

lock = threading.Lock()
aircraft = {}   # hex -> dict
messages_total = 0


def _now_ms():
    return int(time.time() * 1000)


def _safe_int(s):
    try:
        return int(s)
    except Exception:
        return None


def _safe_float(s):
    try:
        return float(s)
    except Exception:
        return None


def update_from_msg(line: str):
    """
    Parse BaseStation "MSG,..." line.
    Format (common):
    MSG,transmissionType,sessionID,aircraftID,hex,flightID,dateGen,timeGen,dateLog,timeLog,
        callsign,altitude,groundSpeed,track,lat,lon,verticalRate,squawk,alert,emergency,spi,isOnGround
    """
    global messages_total

    parts = line.strip().split(",")
    if len(parts) < 5:
        return
    if parts[0] != "MSG":
        return

    # Indici "classici"
    # 0 MSG
    # 1 transmissionType
    # 4 hex
    hex_id = parts[4].strip().lower()
    if not hex_id:
        return

    # Campi opzionali (possono essere stringhe vuote)
    callsign = parts[10].strip() if len(parts) > 10 else ""
    altitude = _safe_int(parts[11]) if len(parts) > 11 and parts[11] else None
    gs = _safe_float(parts[12]) if len(parts) > 12 and parts[12] else None
    track = _safe_float(parts[13]) if len(parts) > 13 and parts[13] else None
    vr = _safe_int(parts[16]) if len(parts) > 16 and parts[16] else None
    lat = _safe_float(parts[14]) if len(parts) > 14 and parts[14] else None
    lon = _safe_float(parts[15]) if len(parts) > 15 and parts[15] else None
    squawk = parts[17].strip() if len(parts) > 17 else ""

    # isOnGround (0/1) se presente
    on_ground = None
    if len(parts) > 21 and parts[21] != "":
        on_ground = (parts[21].strip() == "1")

    now = time.time()

    with lock:
        messages_total += 1
        d = aircraft.get(hex_id)
        if not d:
            d = {
                "hex": hex_id,
                "seen": 0,          # secondi (dump1090 style)
                "seen_pos": 0,      # secondi
                "messages": 0,
                "_last_any": now,
                "_last_pos": None,
            }
            aircraft[hex_id] = d

        d["messages"] = d.get("messages", 0) + 1
        d["_last_any"] = now

        if callsign:
            # dump1090 usa "flight" spesso con padding, qui lo mettiamo pulito
            d["flight"] = callsign.strip()

        if altitude is not None:
            # dump1090 usa "alt_baro" o "alt_geom" a seconda; qui mettiamo alt_baro
            d["alt_baro"] = altitude

        if gs is not None:
            d["gs"] = gs   # float ok
        if track is not None:
            d["track"] = track  # float ok

        if vr is not None:
            d["baro_rate"] = vr

        if squawk:
            d["squawk"] = squawk

        if on_ground is not None:
            d["gnd"] = on_ground

        if lat is not None and lon is not None:
            d["lat"] = lat
            d["lon"] = lon
            d["_last_pos"] = now
            d["last_pos_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))

def msg_reader_loop(host="127.0.0.1", port=MSG_PORT):
    """
    Connects to ModeSMixer MSG outServer and consumes lines.
    Reconnects on failure.
    """
    backoff = 1
    while True:
        try:
            s = socket.create_connection((host, port), timeout=10)
            s.settimeout(30)
            backoff = 1
            buf = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    raise ConnectionError("MSG socket closed")
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        text = line.decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                    if text:
                        update_from_msg(text)
        except Exception as e:
            # print(f"[MSG] reconnecting after error: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)


def build_aircraft_json():
    """
    Build a dump1090-like aircraft.json payload, with extra fields
    expected by our listener (type/category/vert_rate/timestamp).
    """
    now = time.time()
    now_ms = _now_ms()

    with lock:
        # purge stale
        stale = [hex_id for hex_id, d in aircraft.items()
                 if now - d.get("_last_any", 0) > AIRCRAFT_TTL_SEC]
        for hex_id in stale:
            del aircraft[hex_id]

        out_list = []
        for d in aircraft.values():
            last_any = d.get("_last_any", now)
            last_pos = d.get("_last_pos", None)

            item = {
                "hex": d["hex"],
                "messages": d.get("messages", 0),
                "seen": round(now - last_any, 1),

                # campi sempre presenti (anche vuoti) per compatibilità
                "type": d.get("type", ""),
                "category": d.get("category", ""),
            }

            # Copia sempre i campi che esistono nello stato
            for k in ("flight", "alt_baro", "gs", "track", "lat", "lon", "baro_rate", "squawk", "gnd"):
                if k in d:
                    item[k] = d[k]

            # heading indipendente dalla posizione
            if "track" in d:
                item["heading"] = d["track"]

            # seen_pos solo se abbiamo posizione
            if last_pos is not None:
                item["seen_pos"] = round(now - last_pos, 1)
            else:
                item["seen_pos"] = None

            # alias vert_rate (dump1090 a volte usa vert_rate)
            if "baro_rate" in item and "vert_rate" not in item:
                item["vert_rate"] = item["baro_rate"]

            # timestamp ISO (usa ultimo fix posizione se disponibile, altrimenti ultimo qualsiasi msg)
            item["timestamp"] = (
                d.get("last_pos_iso")
                or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(last_any))
            )

            out_list.append(item)

        payload = {
            "now": now_ms,
            "messages": messages_total,
            "aircraft": out_list
        }

    return payload


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/data/aircraft.json":
            self._send_json(build_aircraft_json())
            return

        # comodo: endpoint health
        if path == "/health":
            self._send_json({"ok": True, "ts": _now_ms()})
            return

        self._send_json({"error": "not found", "path": path}, status=404)

    def log_message(self, fmt, *args):
        # silenzioso
        return


def main():
    t = threading.Thread(target=msg_reader_loop, daemon=True)
    t.start()

    httpd = HTTPServer((HTTP_BIND, HTTP_PORT), Handler)
    print(f"[HTTP] Serving on http://{HTTP_BIND}:{HTTP_PORT}/data/aircraft.json")
    httpd.serve_forever()


if __name__ == "__main__":
    main()