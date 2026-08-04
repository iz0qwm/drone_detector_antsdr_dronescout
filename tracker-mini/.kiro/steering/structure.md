# Mini Tracker — Repository Structure

## Workspace Root

```
tracker-mini/
```

This directory is inside a larger Git repository (`drone_detector_antsdr_dronescout`), but only `tracker-mini/` is in scope for development.

## Top-Level Layout

| Path | Purpose |
|------|---------|
| `backend/app.py` | Flask application entry point |
| `backend/config.py` | Settings loader (`SETTINGS`, `save_settings()`) |
| `backend/routes/` | Flask Blueprint modules (thin JSON API layer) |
| `backend/services/` | Service modules (business logic, hardware, workers) |
| `backend/services/ui/lcd.py` | LCD display service |
| `frontend/index.html` | Dashboard single-page shell |
| `frontend/css/` | Stylesheets |
| `frontend/js/` | Frontend modules (no build step) |
| `frontend/js/air/` | ADS-B traffic layer |
| `frontend/js/drones/` | Remote ID traffic layer |
| `frontend/js/glider/` | OGN/FLARM traffic layer |
| `frontend/js/meshtastic/` | Meshtastic operator layer |
| `frontend/js/missions/` | Mission planning modules |
| `frontend/vendor/` | Leaflet, Leaflet-Geoman, plugins |
| `frontend/icons/` | UI icons |
| `frontend/help/docs/` | Documentation sources (MkDocs) |
| `frontend/help/mkdocs.yml` | MkDocs configuration |
| `frontend/help/site/` | Generated HTML documentation (do NOT edit) |
| `config/settings.json` | Runtime settings |
| `config/map_provider.json` | Map tile provider config |
| `maps/` | Offline MBTiles files + catalog |
| `CNSAS/` | Reference documents (CNSAS operations, costs) |
| `requirements.txt` | Python runtime dependencies |
| `AGENTS.md` | Agent development rules |
| `AI_HANDOFF.md` | Development status and handoff |
| `DOCUMENTATION.md` | Documentation writing rules |
| `.kiro/steering/` | Kiro steering files |

## Route Modules (backend/routes/)

Each module registers a Flask Blueprint with JSON API endpoints:

`air_local`, `air_network`, `ds110`, `dsc`, `gps`, `hardware`, `logs`, `maps`, `meshtastic`, `missions`, `network`, `network_manager`, `notifications`, `ogn_network`, `readsb`, `remoteid`, `services`, `settings`, `status`, `teams`, `update`

## Service Modules (backend/services/)

Implementation logic behind routes. Key services with background threads:

- `ds110.py` — Remote ID MAVLink worker
- `meshtastic_service.py` — Meshtastic serial worker
- `dsc_heartbeat.py` — DSC heartbeat loop
- `ui/lcd.py` — LCD refresh worker
- `map_downloader.py` — tile download workers

## Frontend Module Pattern

Each traffic source follows: `*-controller.js` (init), `*-layer.js` (Leaflet markers), `*-network.js` (API calls).

Mission modules: `mission-controller`, `mission-planning`, `mission-draw`, `mission-layer`, `mission-layer-properties`, `mission-network`, `mission-toolbar`, `mission-teams`, `mission-dsc`.

## Deployed Paths (on Raspberry Pi)

| Path | Content |
|------|---------|
| `/home/pi/tracker-mini/` | Application installation |
| `/home/pi/tracker-mini/missions/` | Mission data |
| `/home/pi/tracker-mini-updater/` | Update workflow data |
| `/run/readsb/aircraft.json` | ADS-B decoder output |

## Git Information

- Remote: `https://github.com/iz0qwm/drone_detector_antsdr_dronescout`
- Branch: `main`
- HEAD: `512e341` ("modifiche per kiro")
