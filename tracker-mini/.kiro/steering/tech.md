# Mini Tracker — Technical Stack

## Runtime

- **Platform**: Raspberry Pi (ARM, Debian-based OS)
- **Language**: Python 3 (backend), vanilla JavaScript (frontend)
- **Framework**: Flask (backend web server)
- **Map library**: Leaflet + Leaflet-Geoman
- **No frontend build step** — static HTML/CSS/JS served directly
- **No database server** — all persistence is JSON files and MBTiles (SQLite)

## Backend Entry Point

`backend/app.py` — Flask app on `0.0.0.0:5000`

## Key Dependencies (requirements.txt)

| Package | Purpose |
|---------|---------|
| Flask, Werkzeug | Web server and API |
| requests | Outbound HTTP (DSC, traffic sources) |
| psutil | System status (CPU, RAM, disk, network) |
| pymavlink | DS110 Remote ID MAVLink protocol |
| pyserial | Serial transport (DS110, Meshtastic) |
| gpsd-py3 | GPS data from gpsd |
| meshtastic | Meshtastic serial gateway control |
| PyPubSub | Meshtastic event subscription |
| RPLCD | 20x4 I2C character LCD |
| smbus2 | I2C bus access for LCD |
| gpiozero | Rotary encoder GPIO input |

## System Dependencies (not in requirements.txt)

- `lgpio` — provided by Raspberry Pi OS package, accessed via system-site-packages
- `gpsd` — system service on 127.0.0.1:2947
- `readsb` — ADS-B decoder, `readsb-local.service`
- `nmcli` / NetworkManager — Wi-Fi and Ethernet management
- `systemctl` — service control (readsb, tracker-mini)

## Threading Model

- Main thread: Flask request handling
- Daemon threads: DS110 worker, Meshtastic worker, DSC heartbeat, LCD refresh, map downloads
- Module-level `running` flags control worker lifecycle
- No formal dependency injection; modules import each other directly

## Configuration

- `config/settings.json` — loaded into memory at startup via `backend/config.py`
- `config/map_provider.json` — Thunderforest API key and provider
- Changes persisted via `save_settings()` which rewrites the JSON file

## Data Storage Locations

| Data | Path |
|------|------|
| Settings | `config/settings.json` |
| Map provider | `config/map_provider.json` |
| Offline maps | `maps/*.mbtiles` |
| Map catalog | `maps/maps_catalog.json` |
| Missions | `/home/pi/tracker-mini/missions/` |
| Update packages | `/home/pi/tracker-mini-updater/` |
| ADS-B decoder output | `/run/readsb/aircraft.json` |
| Application logs | In-memory deque (2000 entries, lost on restart) |

## Service Architecture Pattern

```
Dashboard (browser) → HTTP polling → Flask routes → Service modules → OS/Hardware/Files
```

No WebSocket. Frontend polls every 5 seconds for status, on-demand for traffic data.

## Virtual Environment

Must use `include-system-site-packages = true` in `venv/pyvenv.cfg` for `lgpio` access.

## Application Service

`tracker-mini.service` — systemd unit that runs the Flask backend.

## References

- `frontend/help/docs/developer/architecture.md`
- `frontend/help/docs/developer/services.md`
- `frontend/help/docs/developer/backend.md`
