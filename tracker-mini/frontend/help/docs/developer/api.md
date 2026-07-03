# API

### Part of the Mini Tracker Developer Documentation

---

## Purpose

This document provides a developer-oriented reference for the REST API exposed by Mini Tracker.

It groups endpoints by subsystem and documents method, purpose, request parameters and response format.

For the overall request flow and software architecture, refer to `developer/architecture.md`.

---

## API Model

Mini Tracker exposes local HTTP endpoints from the Flask backend.

Most API endpoints return JSON. The offline tile endpoint returns PNG tile bytes. Dashboard frontend modules call these APIs using `fetch()`.

There is no shared authentication, version prefix or global error schema in the current implementation.

---

## System Status

### `GET /api/status`

Returns basic host status.

Request parameters: none.

Response fields:

| Field | Description |
|----------|-------------|
| `hostname` | Hostname from the operating system. |
| `cpu` | CPU percent from psutil. |
| `ram` | RAM usage percent. |
| `disk` | Root filesystem usage percent. |
| `uptime` | Boot time from psutil. |

### `POST /api/system/restart`

Restarts the Mini Tracker application service.

Request parameters: none.

Response fields: `success`.

The backend executes `sudo systemctl restart tracker-mini.service`.

### `POST /api/system/reboot`

Requests an operating system reboot.

Request parameters: none.

Response fields: `success`.

The backend executes `sudo /usr/sbin/reboot`.

### `POST /api/system/shutdown`

Requests operating system shutdown.

Request parameters: none.

Response fields: `success`.

The backend executes `sudo /usr/sbin/shutdown -h now`.

### `GET /api/services`

Returns aggregated service indicator state for the Dashboard.

Request parameters: none.

Response fields:

| Field | Description |
|----------|-------------|
| `internet` | Internet connectivity state. |
| `ads_local` | Local ADS-B decoder freshness state. |
| `ads_network` | Network ADS-B availability based on Internet state. |
| `remote_id` | DS110 worker running state. |
| `meshtastic_enabled` | Meshtastic worker running state. |
| `meshtastic_alive` | Recent Meshtastic packet state. |
| `ogn` | OGN availability based on Internet state. |
| `dsc` | DSC availability based on Internet state. |

### `GET /api/hardware`

Returns aggregated hardware status.

Request parameters: none.

Response fields:

| Field | Description |
|----------|-------------|
| `wifi_client` | Whether `/sys/class/net/wlan1` exists. |
| `ds110` | Whether the configured DS110 device path exists. |
| `ds110_alive` | Whether DS110 heartbeat is recent. |
| `meshtastic` | Meshtastic worker running state. |
| `meshtastic_alive` | Recent Meshtastic packet state. |
| `adsb_receiver` | Whether `/run/readsb/aircraft.json` exists. |
| `adsb_decoder` | Whether the ADS-B decoder output is recent. |

---

## Network API

### `GET /api/network`

Returns network status.

Request parameters: none.

Response fields:

| Field | Description |
|----------|-------------|
| `admin_lan` | Object with `connected` and fixed `ip`. |
| `user_lan` | Object with `connected` and optional `ip`. |
| `wifi_ap` | Object with `connected`, `ip`, `ssid` and `ap_mode`. |
| `wifi_client` | Object with `connected`, `ip` and `ssid`. |
| `wifi` | Compatibility object for current frontend behavior. |
| `internet` | Internet connectivity state. |

### `GET /api/connections`

Returns NetworkManager connections.

Request parameters: none.

Response format: array of connection objects with `name`, `type` and `device`, or an object with `error`.

### `GET /api/wifi-scan`

Scans Wi-Fi networks on the Wi-Fi Client interface.

Request parameters: none.

Response format: array of networks.

Network fields:

| Field | Description |
|----------|-------------|
| `ssid` | Network SSID. |
| `signal` | Signal level as an integer. |
| `security` | Security string from `nmcli`. |
| `saved` | Whether a saved connection exists. |

### `POST /api/wifi/connect`

Connects the Wi-Fi Client interface to a network.

Request JSON:

| Field | Required | Description |
|----------|----------|-------------|
| `ssid` | Yes | Network SSID. |
| `password` | No | Network password. |

Response fields:

| Field | Description |
|----------|-------------|
| `success` | Operation result. |
| `message` | `nmcli` result or error message. |

If `ssid` is missing, the route returns HTTP 400.

### `POST /api/wifi/disconnect`

Disconnects the Wi-Fi Client interface.

Request parameters: none.

Response fields: `success`, `message`.

### `POST /api/ap/start`

Starts the Access Point.

Request parameters: none.

Response fields: `success`, `message`.

### `POST /api/ap/stop`

Stops the Access Point.

Request parameters: none.

Response fields: `success`, `message`.

### `GET /api/ap/status`

Returns Access Point status.

Response fields:

| Field | Description |
|----------|-------------|
| `active` | Whether the Access Point connection is active. |
| `ssid` | Active SSID or `null`. |

### `GET /api/lan/config`

Returns User LAN configuration.

Response fields:

| Field | Description |
|----------|-------------|
| `success` | Whether the configuration was read. |
| `ip` | User LAN IP if configured. |
| `mask` | Subnet mask. |
| `gateway` | Gateway value. |
| `message` | Error message when unsuccessful. |

### `POST /api/lan/config`

Updates User LAN configuration.

Request JSON:

| Field | Required | Description |
|----------|----------|-------------|
| `ip` | Yes | User LAN IP address. |
| `mask` | Yes | Subnet mask. |
| `gateway` | No | Gateway address. |

Response fields: `success`, `message`.

---

## Settings API

### `GET /api/settings`

Returns Dashboard settings.

Response fields:

| Field | Description |
|----------|-------------|
| `map` | Map configuration object from settings. |
| `ap_ssid` | Access Point SSID. |

### `GET /api/ds110/settings`

Returns DS110 configuration.

Response fields: `interface`, `device`, `baudrate`.

### `POST /api/ds110/settings`

Updates DS110 configuration.

Request JSON:

| Field | Required | Description |
|----------|----------|-------------|
| `interface` | No | `usb` or `uart`; defaults to `usb`. |
| `device` | No | Device path; defaults to `/dev/ttyACM0`. |
| `baudrate` | No | Baud rate; defaults to `115200`. |

Response fields: `success`.

### `GET /api/serial/ports`

Lists detected serial paths and current DS110 device.

Response fields:

| Field | Description |
|----------|-------------|
| `current` | Current configured DS110 device path. |
| `ports` | Sorted list of detected and configured serial paths. |

### `GET /api/dsc/settings`

Returns DSC settings.

Response format: DSC settings object from `SETTINGS["dsc"]`.

### `POST /api/dsc/settings`

Updates DSC settings.

Request JSON:

| Field | Required | Description |
|----------|----------|-------------|
| `node_name` | No | DSC node display name. |
| `position_source` | No | `manual` or `gps`. |
| `lat` | No | Manual latitude. |
| `lon` | No | Manual longitude. |
| `sync_enabled` | No | DSC synchronization flag. |

Response format: updated DSC settings object.

---

## Maps API

### `GET /tiles/<z>/<x>/<y>.png`

Returns an offline map tile.

Path parameters:

| Parameter | Description |
|----------|-------------|
| `z` | Zoom level. |
| `x` | Tile column. |
| `y` | Tile row. |

Response: PNG tile bytes, or HTTP 404 when no tile is found.

### `GET /api/maps`

Returns installed maps.

Response format: array of map objects.

Returned fields include `name`, `description`, `created`, `source`, `center_lat`, `center_lon`, `radius_km`, `min_zoom`, `max_zoom`, `size_mb`, `active` and `protected`.

### `GET /api/maps/storage`

Returns map storage usage for the maps directory.

Response fields: `total_gb`, `used_gb`, `free_gb`.

### `DELETE /api/maps/<map_name>`

Deletes a map file unless it is the protected base map.

Path parameters: `map_name`.

Response fields: `success`, `message`.

### `POST /api/maps/update-description`

Updates map description metadata.

Request JSON:

| Field | Required | Description |
|----------|----------|-------------|
| `name` | Yes | Map filename. |
| `description` | No | Description text. |

Response fields: `success`.

### `GET /api/maps/downloads`

Returns in-memory map download jobs.

Response format: object keyed by job ID. Job fields include `status`, `progress`, `current_tile`, `total_tiles`, `message`, `created` and optionally `result`.

### `POST /api/maps/download`

Starts a map download job.

Request JSON:

| Field | Required | Description |
|----------|----------|-------------|
| `lat` | Yes | Download center latitude. |
| `lon` | Yes | Download center longitude. |
| `radius` | Yes | Radius in kilometers. |
| `description` | No | Map description and filename basis. |

Response fields: `success`, `job_id`.

### `GET /api/maps/download-status/<job_id>`

Returns a map download job.

Path parameters: `job_id`.

Response format: job object, or `{}` when not found.

### `GET /api/maps/provider`

Returns map provider status.

Response fields:

| Field | Description |
|----------|-------------|
| `provider` | Provider name, defaulting to `thunderforest`. |
| `configured` | Whether an API key is present. |

### `POST /api/maps/provider`

Updates map provider settings.

Request JSON:

| Field | Required | Description |
|----------|----------|-------------|
| `provider` | No | Provider name; defaults to `thunderforest`. |
| `api_key` | No | Provider API key. |

Response fields: `success`.

### `POST /api/maps/set-active`

Updates a map active flag.

Request JSON:

| Field | Required | Description |
|----------|----------|-------------|
| `name` | Yes | Map filename. |
| `active` | Yes | Boolean active state. |

Response fields: `success`.

---

## Mission API

### `GET /api/missions`

Returns the mission index array.

Response fields per mission: `id`, `name`, `description`, `status`.

### `POST /api/missions/create`

Creates a mission.

Request JSON:

| Field | Required | Description |
|----------|----------|-------------|
| `name` | Yes | Mission name. |
| `description` | No | Mission description. |

Response fields: `success`, `mission`.

Missing mission name returns HTTP 400.

### `GET /api/missions/<mission_id>`

Returns a mission.

Path parameters: `mission_id`.

Response format: mission object, or HTTP 404 with `success: false` and `message`.

### `PUT /api/missions/<mission_id>`

Updates a mission.

Request JSON fields may include `name`, `description` and `status`.

Response fields: `success`, `mission`.

### `DELETE /api/missions/<mission_id>`

Deletes a mission directory and removes it from the index.

Response fields: `success`.

### `GET /api/missions/current`

Returns the currently selected mission object, or `null`.

### `POST /api/missions/select`

Selects the current mission.

Request JSON:

| Field | Required | Description |
|----------|----------|-------------|
| `mission_id` | Yes | Mission ID to select. |

Response fields: `success`.

### `POST /api/missions/import-geojson`

Imports a GeoJSON file as a mission layer.

Request form fields:

| Field | Required | Description |
|----------|----------|-------------|
| `mission_id` | Yes | Target mission ID. |
| `file` | Yes | Uploaded `.geojson` or JSON file. |

Response fields: `success`, `layer`.

Missing input or failed import returns HTTP 400.

### `GET /api/missions/<mission_id>/layers`

Returns mission layers.

Response format: array of layer objects, or HTTP 404 if the mission does not exist.

### `GET /api/missions/<mission_id>/layers/<layer_id>`

Returns one layer object, or HTTP 404 if missing.

### `POST /api/missions/<mission_id>/layers`

Creates a layer.

Request JSON fields may include `name`, `type`, `geometry`, `visible`, `locked`, `style`, `properties` and `geojson`.

Response fields: `success`, `layer`.

### `PUT /api/missions/<mission_id>/layers/<layer_id>`

Updates an existing layer by merging request JSON into the stored layer.

Response fields: `success`, `layer`.

### `DELETE /api/missions/<mission_id>/layers/<layer_id>`

Deletes a layer.

Response fields: `success`.

---

## Teams API

### `GET /api/teams`

Returns current mission team status with Meshtastic data.

Response fields:

| Field | Description |
|----------|-------------|
| `gateway` | Meshtastic gateway info. |
| `gateway_node` | Normalized gateway node if matched. |
| `operators` | Configured operators merged with matching nodes. |
| `external_nodes` | Meshtastic nodes not matched to configured operators. |
| `messages` | Current implementation returns an empty array. |

### `GET /api/teams/config`

Returns team configuration for the current mission.

Response fields: `operators`.

### `POST /api/teams/operator`

Adds an operator to the current mission team.

Request JSON:

| Field | Required | Description |
|----------|----------|-------------|
| `longName` | Yes | Operator long name. |
| `shortName` | Yes | Meshtastic short name used for matching. |

Response fields: `success`.

### `PUT /api/teams/operator/<operator_id>`

Updates an operator.

Request JSON fields: `longName`, `shortName`.

Response fields: `success`.

### `DELETE /api/teams/operator/<operator_id>`

Deletes an operator.

Response fields: `success`.

---

## Notifications API

### `GET /api/notifications`

Returns in-memory notification messages.

Request parameters: none.

Response fields:

| Field | Description |
|----------|-------------|
| `ok` | Request result. |
| `messages` | Notification entries in reverse chronological order. |

Notification entries include `id`, `timestamp`, `category`, `severity`, `source`, `target`, `target_node_id`, `text` and `status`. Failed send attempts may include `error`.

### `DELETE /api/notifications`

Clears in-memory notification messages.

Request parameters: none.

Response fields: `ok`.

### `POST /api/notifications/operator`

Sends a notification message to one operator node through the Notification Service.

Request JSON:

| Field | Required | Description |
|----------|----------|-------------|
| `node_id` | Yes | Meshtastic node ID used as the delivery target. |
| `text` | Yes | Message text. |
| `category` | No | Notification category; defaults to `manual`. |
| `severity` | No | Notification severity; defaults to `info`. |

Response fields:

| Field | Description |
|----------|-------------|
| `ok` | True when the notification status is `sent`. |
| `notification` | Notification record with final status. |

### `POST /api/notifications/all`

Sends a notification message to all configured online operators with an associated node ID.

Request JSON:

| Field | Required | Description |
|----------|----------|-------------|
| `text` | Yes | Message text. |
| `category` | No | Notification category; defaults to `manual`. |
| `severity` | No | Notification severity; defaults to `info`. |

Response fields:

| Field | Description |
|----------|-------------|
| `ok` | Request result. |
| `notifications` | Notification records created for the selected online operators. |

---

## Traffic API

### `GET /api/air/local`

Returns local ADS-B aircraft from readsb output.

Query parameters:

| Parameter | Required | Description |
|----------|----------|-------------|
| `minLat` | Yes | Bounds minimum latitude. |
| `maxLat` | Yes | Bounds maximum latitude. |
| `minLon` | Yes | Bounds minimum longitude. |
| `maxLon` | Yes | Bounds maximum longitude. |
| `showAll` | No | `"true"` disables the standard altitude filter. |

Response fields: `success`, `aircraft`.

Aircraft fields include `icao`, `callsign`, `lat`, `lon`, `altitude`, `speed`, `heading`, `category`, `isHelicopter`, `source` and `updatedAt`.

### `GET /api/air/network`

Returns network ADS-B aircraft from external sources.

Query parameters: same as `/api/air/local`.

Response fields:

| Field | Description |
|----------|-------------|
| `success` | Request result. |
| `sources` | Counts by external source. |
| `aircraft` | Merged aircraft array. |

### `GET /api/ogn/network`

Returns OGN / FLARM traffic.

Query parameters: `minLat`, `maxLat`, `minLon`, `maxLon`.

Response fields: `success`, `count`, `objects`.

Object fields include `id`, `callsign`, `lat`, `lon`, `alt_m`, `heading`, `speed`, `source`, `last_seen` and `updatedAt`.

### `GET /api/remoteid/aircraft`

Returns Remote ID aircraft currently held in memory by the DS110 service.

Response format: array of aircraft objects. Fields may include `source`, `serial`, `vendor`, `model`, `id_type`, `ua_type`, `lat`, `lon`, `altitude`, `height`, `speed`, `heading`, `operator_lat`, `operator_lon`, `operator_altitude`, `operator_id` and `last_seen`.

---

## Hardware Control API

### `GET /api/gps/status`

Returns GPS state.

Response fields include `available`, `fix`, `mode`, `lat`, `lon`, `alt`, `speed`, `track`, `satellites`, `hdop` and optionally `error`.

### `GET /api/ds110/status`

Returns DS110 worker state.

Response fields: `enabled`.

### `POST /api/ds110/enable`

Starts or stops the DS110 worker.

Request JSON:

| Field | Required | Description |
|----------|----------|-------------|
| `enabled` | No | Boolean; defaults to true. |

Response fields: `success`, `enabled`.

### `GET /api/readsb/status`

Returns whether `readsb-local.service` is active.

Response fields: `success`, `enabled`.

### `POST /api/readsb/enable`

Starts or stops `readsb-local.service`.

Request JSON:

| Field | Required | Description |
|----------|----------|-------------|
| `enabled` | No | Boolean; defaults to false when absent. |

Response fields: `success`, `enabled`.

### `GET /api/meshtastic/nodes`

Returns Meshtastic nodes.

Response fields: `ok`, `nodes`, `alive`.

### `GET /api/meshtastic/status`

Returns Meshtastic service status.

Response fields: `ok`, `enabled`, `alive`, `nodes_count`.

### `GET /api/meshtastic/gateway`

Returns Meshtastic gateway information.

Response fields: `ok`, `gateway`.

### `POST /api/meshtastic/enable`

Starts or stops the Meshtastic worker.

Request JSON:

| Field | Required | Description |
|----------|----------|-------------|
| `enabled` | No | Boolean; defaults to true. |

Response fields: `success`, `enabled`.

### `POST /api/meshtastic/nodes/reset`

Requests a complete Meshtastic radio NodeDB reset and clears the Mini Tracker node cache.

Request parameters: none.

Response fields: `ok`, or `ok: false` and `error` with HTTP 500 when the radio interface is not connected or the reset fails.

### `DELETE /api/meshtastic/nodes/<node_id>`

Requests removal of one node from the Meshtastic radio NodeDB and removes it from the Mini Tracker node cache.

Path parameters: `node_id`.

Response fields: `ok`, or `ok: false` and `error` with HTTP 500 when the radio interface is not connected or removal fails.

---

## Logs API

### `GET /api/logs`

Returns in-memory logs.

Response format: array of log entries.

Log fields: `time`, `timestamp`, `component`, `level`, `message`.

### `POST /api/logs/clear`

Clears in-memory logs.

Response fields: `status`.

---

## Update API

All update endpoints are registered under `/api/update`.

### `GET /api/update/status`

Returns updater API status.

Response fields: `service`, `status`.

### `POST /api/update/upload`

Uploads an update ZIP package.

Request form fields: `file`.

Response fields: `success`, `filename`, `size`, or `success: false` and `error`.

### `GET /api/update/backups`

Lists backup directories.

Response format: array of objects with `name`.

### `GET /api/update/package/<filename>`

Validates one uploaded package.

Response fields: `success`, `filename`, `type`, `backend`, `frontend`, or `success: false` and `error`.

### `GET /api/update/packages`

Lists uploaded ZIP packages.

Response format: array of objects with `filename`, `size` and `type`.

### `POST /api/update/backup`

Creates a backend and frontend backup.

Response fields: `success`, `backup`.

### `POST /api/update/extract/<filename>`

Extracts an uploaded package to staging.

Response fields: `success`, `filename`, `contents`, or `success: false` and `error`.

### `POST /api/update/test/backend`

Runs Python syntax checks on staged backend files.

Response fields: `success`, `checked`, `errors`.

### `POST /api/update/test/<filename>`

Runs package validation, extraction, structure checks and backend syntax checks when applicable.

Response fields include `success`, `package`, `type`, `backend`, `frontend`, `checked_files`, or failure details with `stage`.

### `POST /api/update/test-install`

Copies staged backend and frontend into the test install directory.

Response fields: `success`, `backend_files`, `frontend_files`, or `error`.

### `POST /api/update/test-import`

Tests importing the staged backend from the test install directory.

Response fields: `success`, or `success: false` and `error`.

### `GET /api/update/current`

Returns current installed package metadata.

Response fields include `installed_package`, `installed_at`, `backup`, `status` and optionally `rollback_reason`.

### `POST /api/update/restore/<backup_name>`

Restores a backup to the live tracker directory.

Response fields: `success`, or `success: false` and `error`.

### `POST /api/update/restore-test/<backup_name>`

Restores a backup to the test install directory.

Response fields: `success`, or `success: false` and `error`.

### `POST /api/update/pre-install/<filename>`

Runs package test, test install, backend import test and backup creation.

Response fields:

| Field | Description |
|----------|-------------|
| `success` | Result. |
| `ready` | Present when successful. |
| `backup` | Created backup name. |
| `steps` | Completed step list. |
| `stage` | Failure stage when unsuccessful. |
| `details` | Failure details when unsuccessful. |

### `POST /api/update/request-install/<filename>`

Creates an install request.

Request JSON:

| Field | Required | Description |
|----------|----------|-------------|
| `backup` | No | Backup name to associate with request. |

Response fields: `success`, `package`, `backup`, `status`, or `error`.

---

## Related Documentation

- `developer/architecture.md`
- `developer/backend.md`
- `developer/services.md`
- `developer/frontend.md`
- `developer/mission-storage.md`
- `developer/coding-guidelines.md`
