# MT-TRAFFIC-01 — Traffic Proximity Awareness

## Requirements (Revision 2)

---

## Feature Purpose

Provide Mini Tracker operators with real-time awareness of how close drones and aircraft are to each other, directly on the operational map.

The feature is **informational and non-certified**. It does not replace certified collision-avoidance systems (TCAS, ACAS) and must not be presented as such.

---

## Scope

The feature calculates and displays horizontal distance between detected drones (Remote ID) and nearby aircraft (ADSBRx, optional ADSBNet), highlights proximity states on the map, and provides a compact nearby-traffic summary.

### In Scope (MVP)

- Horizontal distance calculation between drones and aircraft
- Proximity state classification with configurable thresholds
- Visual emphasis on relevant aircraft markers
- Distance line for the highest-priority proximity pair
- Nearby-traffic summary (ordered by severity then distance)
- Stale-track detection and cleanup
- Hysteresis to prevent state flickering
- Configuration via existing settings architecture
- Graceful handling of missing or invalid data
- Approaching/diverging movement trend (when data quality supports it)
- Full offline operation with ADSBRx + Remote ID
- Optional ADSBNet enrichment when available
- Source provenance preserved and displayable
- Multiple-drone evaluation (all valid pairs)
- Duplicate aircraft detection and merge (ADSBRx + ADSBNet)
- Backend authoritative proximity engine with API
- Frontend rendering of backend-computed results

### Out of Scope (MVP)

- CPA / TCPA prediction (prepared architecturally, not implemented)
- Vertical separation warnings (altitude references are incompatible)
- Sound alerts
- Persistence of proximity events to disk
- Regulatory airspace integration
- Automatic avoidance recommendations
- Meshtastic proximity alert distribution (prepared via backend API)
- OGN/FLARM inclusion (deferred — see §OGN/FLARM Scope)

---

## Traffic Source Model

### ADSBRx

The local ADS-B reception subsystem. Receives aircraft directly through the RTL-SDR receiver and the locally running `readsb-local.service`.

Verified implementation:
- Backend service: `services/readsb.py` controls `readsb-local.service` via systemctl
- Data source: reads `/run/readsb/aircraft.json` (decoder output)
- Backend service: `services/air_local.py` normalizes and filters aircraft
- API: `GET /api/air/local?minLat&maxLat&minLon&maxLon&showAll`
- Enable/disable: `POST /api/readsb/enable` → starts/stops systemd service, persists to `settings.traffic.adsb_local_enabled`
- Health indicator: file exists AND mtime < 30s (`services/services.py: adsb_local_alive()`)
- Frontend toggle: checkbox `adsbLocalEnabled` → calls enable API
- Frontend label: "ADSB Rx" (LED and settings panel)
- Refresh: frontend polls every 15s

Operational characteristics:
- Local hardware source
- Works without Internet
- Primary native ADS-B source
- Must remain operational for Traffic Proximity Awareness even when ADSBNet is disabled or offline

### ADSBNet

The optional Internet-based aircraft-data subsystem.

Verified implementation:
- Backend service: `services/air_network.py` fetches from three providers:
  - SolarMonitor (`solarmonitor.kwos.org/api/adsb/aircraft.json`)
  - OGN-derived ADS-B (`solarmonitor.kwos.org/api/ogn/traffic`, filtered to source=ADSB)
  - OpenSky (`opensky-network.org/api/states/all`)
- API: `GET /api/air/network?minLat&maxLat&minLon&maxLon&showAll`
- Merge logic: `merge_aircraft(*lists)` — deduplicates by ICAO, keeps newest `updatedAt`
- Enable/disable: **frontend-only** via `localStorage.getItem("adsbNetworkEnabled")` — if "false", `air-network.js` returns empty without calling API
- No backend setting for ADSBNet enable — the backend always serves if asked
- Internet detection: `services/network.py: has_internet()` (TCP connect to 8.8.8.8:53)
- Provider failure: each provider silently returns empty list on timeout/error; others still contribute
- Health indicator: `services/services.py` reports `ads_network: internet` (available if Internet available)
- Frontend label: "ADSB Net" (LED and settings panel)
- Refresh: frontend polls every 15s (same timer as ADSBRx)

Operational characteristics:
- Optional, enabled by user preference (localStorage)
- Requires Internet connectivity
- Fails gracefully (each provider independent, returns [] on error)
- Must not be required for Traffic Proximity Awareness
- Currently NO backend-side enable/disable setting exists — purely frontend preference

### Remote ID (Drones)

Verified implementation:
- Backend service: `services/ds110.py` — MAVLink worker thread on configured serial device
- Coordinate validation: `is_valid_position()` rejects `(0.0, 0.0)` as sentinel (no GPS fix in ODID), validates -90≤lat≤90, -180≤lon≤180
- API: `GET /api/remoteid/aircraft` — returns list of aircraft dicts
- Enable/disable: `POST /api/ds110/enable` → starts/stops worker thread
- Health indicator: `is_alive()` — heartbeat within 30s
- Frontend label: "RID" (LED), "Remote ID" (settings)
- Frontend toggle: checkbox `droneNetworkEnabled`
- Refresh: frontend polls every 5s

### OGN/FLARM

Verified implementation:
- Backend service: `services/ogn_network.py` — fetches from `solarmonitor.kwos.org/api/ogn/traffic`
- Filters to sources: FLARM, SAFESKY, FREEFLIGHT, FANET (explicitly excludes ADSB source)
- API: `GET /api/ogn/network?minLat&maxLat&minLon&maxLon`
- Enable/disable: **frontend-only** via `localStorage.getItem("ognNetworkEnabled")`
- Requires Internet
- Frontend label: "OGN" (LED)
- Refresh: frontend polls every 10s
- Separate from ADSBNet — uses same server but different endpoint and filters different sources

---

## User Stories

1. **As an operator**, I want to see which aircraft are closest to detected drones so I can assess the traffic situation.
2. **As an operator**, I want visual emphasis on aircraft that are within concerning proximity of any drone so I can focus attention where needed.
3. **As an operator**, I want to know whether an aircraft is approaching or moving away from a drone so I can anticipate changes.
4. **As an operator**, I want stale or invalid proximity information to be clearly indicated and removed so I don't act on outdated data.
5. **As an operator**, I want to configure proximity thresholds so I can adjust awareness levels to the operational scenario.
6. **As an operator**, I want Traffic Proximity Awareness to work offline with only my local ADS-B receiver and Remote ID so I can operate in the field without Internet.
7. **As an operator**, I want to see the source of aircraft data (local receiver vs network) when diagnostically useful.
8. **As an operator**, I want proximity awareness to cover ALL visible drones, not just one.

---

## Functional Requirements

### FR-01: Offline Operation

The complete Traffic Proximity Awareness MVP must function with ADSBRx + Remote ID only, without Internet connectivity. ADSBNet is optional enrichment.

### FR-02: Distance Calculation

The backend must calculate horizontal (great-circle) distance in meters between each active drone and each active aircraft within the evaluation radius.

### FR-03: Multiple-Drone Evaluation

The system must evaluate ALL valid drone-aircraft pairs within the evaluation radius. It must not select a single reference drone and ignore others. The nearby-traffic panel displays the highest-severity pairs across all drones.

### FR-04: Proximity States

Each drone-aircraft pair is classified into:

| State | Meaning | Additional Indicator |
|-------|---------|---------------------|
| NORMAL | Distance exceeds all thresholds | — |
| MONITOR | Distance within outer threshold | Text "MON" + blue dashed line |
| CAUTION | Distance within middle threshold | Text "CTN" + orange dashed line |
| WARNING | Distance within inner threshold | Text "WRN" + red solid line + optional pulse |
| STALE | One or both tracks are stale | Text "STL" + gray dotted line |
| UNKNOWN | Insufficient data to calculate | Not displayed |

Every state uses text label + line pattern (not color alone) for accessibility.

### FR-05: Configurable Thresholds

Default thresholds (configurable):

| State | Entry Threshold | Exit Threshold |
|-------|-----------------|----------------|
| MONITOR | 3000 m | 3300 m |
| CAUTION | 1500 m | 1800 m |
| WARNING | 500 m | 700 m |

Exit thresholds provide hysteresis.

### FR-06: Source Health vs Track Freshness

The system must distinguish:
- Source availability (ADSBRx service running, ADSBNet reachable)
- Individual track freshness (last update age for a specific aircraft)

A disabled or unavailable ADSBNet must NOT make valid ADSBRx tracks stale. A provider outage must NOT generate false proximity-state transitions for tracks still available through ADSBRx.

### FR-07: ADSBRx + ADSBNet Duplicate Handling

The same aircraft (same ICAO) may be received from both ADSBRx and ADSBNet. The system must:
- Detect duplicates by ICAO address
- Merge into a single normalized target
- Never create duplicate markers, duplicate proximity pairs, or list the same aircraft twice
- Preserve source provenance (which sources contributed)

### FR-08: Stale Track Detection

| Source | Stale Timeout |
|--------|---------------|
| ADSBRx track | 30s since last decoder update |
| ADSBNet track | 30s since last provider update |
| Normalized target | Stale when ALL contributing sources are stale |
| Drone track | 15s since `last_seen` |

A target remains fresh when at least one accepted source provides a fresh update.

### FR-09: Map Visualization

- One distance line from the highest-priority pair (WARNING > CAUTION > MONITOR; ties broken by shortest distance)
- Proximity rings on up to 5 aircraft in proximity states
- Distance label on the line (meters or km)
- Color + line pattern + text label per state (accessibility: not color alone)
- Subtle pulse animation on WARNING only (configurable, not continuous flash)
- All proximity objects removed when pair exits proximity or tracks become stale

### FR-10: Nearby Traffic Panel

Compact panel listing up to 5 pairs ordered by severity then distance:
- Drone identifier (serial truncated or model)
- Aircraft identifier (callsign or ICAO)
- Horizontal distance
- Proximity state badge (text + color)
- Movement trend indicator
- Source label (RX / NET / RX+NET) when diagnostically useful

Panel visible when ≥1 drone active AND ≥1 aircraft within evaluation radius.

### FR-11: Movement Trend

Using ≥3 valid distance samples over 10-15s window:
- APPROACHING: distance consistently decreasing beyond deadband
- DIVERGING: distance consistently increasing beyond deadband
- STABLE: change within deadband
- UNKNOWN: insufficient data or inconsistent

Display: explicit text labels or horizontal arrows (NOT vertical arrows that could be confused with climb/descent).

### FR-12: Evaluation Radius

Only aircraft within configurable maximum radius (default: 10 km) from any active drone are evaluated.

### FR-13: Invalid Data Handling

- Missing lat/lon → skip target
- `(0.0, 0.0)` → treat as invalid (verified: ODID sentinel for no GPS fix)
- lat/lon outside ±90/±180 → skip target
- Non-finite values → skip target
- Missing speed/heading → disable movement analysis for that pair
- Implausible position jump > 50km between updates → mark history uncertain

### FR-14: Network Loss Transitions

When Internet is lost while ADSBNet was contributing:
- ADSBNet tracks that have no ADSBRx counterpart become stale normally (timeout)
- ADSBRx tracks remain unaffected
- No false state transitions for locally-received aircraft
- When Internet returns, ADSBNet resumes without application restart

### FR-15: Backend Authoritative Engine

The proximity engine runs in the backend. The frontend renders results from the backend API. Thresholds, stale rules, and state classification are NOT duplicated in JavaScript.

---

## Non-Functional Requirements

### NFR-01: Performance

- Backend calculation must complete within 100ms for 50 aircraft × 5 drones
- Backend proximity API response < 200ms
- Calculation cadence: every 5 seconds
- Frontend rendering must not cause visible lag on Raspberry Pi

### NFR-02: Offline Operation

Feature fully operational with ADSBRx + Remote ID only (no Internet).

### NFR-03: Resource Usage

- Proximity state is in-memory only; reset on backend restart
- No persistent storage required
- One additional lightweight computation cycle in the existing Flask process (no new threads)

### NFR-04: Compatibility

- Must not interfere with existing traffic layers or data structures
- Must preserve existing marker popups
- Must respect existing ADSBNet user preference (localStorage)
- Must work alongside mission layers and team markers

### NFR-05: Accessibility

- Every proximity state uses color + text label + line pattern
- Pulsing optional, limited to WARNING, not continuous flash
- Panel readable on small screens

### NFR-06: Performance Measurement

Measurable criteria:
- Baseline: 5-minute observation without proximity feature
- Feature-enabled: 5-minute observation with proximity feature
- Scenarios: local-only (5 aircraft, 1 drone), combined (20 aircraft, 2 drones)
- Backend CPU sustained increase < 5%
- Browser rendering latency increase imperceptible
- API response time `/api/proximity/status` < 200ms at p95

---

## Acceptance Criteria

1. With only ADSBRx + Remote ID active (no Internet), proximity states are correctly calculated and displayed.
2. When ADSBNet is disabled by the user, proximity operates normally using ADSBRx alone.
3. When ADSBNet is enabled but Internet is unavailable, proximity operates normally using ADSBRx alone without errors.
4. When the same aircraft is received from both ADSBRx and ADSBNet, only one target appears with no duplicate proximity pairs.
5. When Internet is lost, ADSBRx-tracked aircraft maintain their proximity states without interruption.
6. Multiple drones each generate proximity pairs with nearby aircraft.
7. The nearby-traffic panel shows pairs across all drones, ranked by severity then distance.
8. Only one distance line is drawn (highest-priority pair).
9. Proximity rings appear on up to 5 aircraft with non-NORMAL states.
10. Each proximity state is distinguishable by color, text label, AND line pattern (not color alone).
11. Movement trend shows APPROACHING/DIVERGING/STABLE/UNKNOWN using non-vertical indicators.
12. Stale tracks transition to STALE state and are cleaned up after grace period.
13. A fresh ADSBRx track is never marked stale due to ADSBNet unavailability.
14. Configuration changes take effect without application restart.
15. No UI element uses TCAS, ACAS, or collision-avoidance terminology.
16. Sustained backend CPU increase < 5% on Raspberry Pi.
17. Dashboard remains responsive during proximity calculations.

---

## Safety and Certification Wording

The feature must include the following notice (visible in the proximity panel or settings):

> **Traffic Proximity Awareness is informational only. It is not a certified collision-avoidance system and must not be used as the sole basis for separation decisions.**

Documentation must use "Traffic Proximity Awareness" or "Nearby Traffic" — never TCAS, ACAS, or collision avoidance.

---

## Excluded Behavior

- No sound alerts
- No automatic avoidance recommendations
- No CPA/TCPA prediction in MVP
- No vertical separation warnings
- No persistence of proximity events
- No regulatory airspace integration
- No Meshtastic proximity alert distribution (backend API prepared for future use)
- No OGN/FLARM in MVP (deferred)

---

## OGN/FLARM Scope Decision

**Option A selected**: OGN/FLARM is deferred to a later iteration.

Rationale:
- OGN requires Internet (same constraint as ADSBNet)
- OGN uses different identifiers (no ICAO for most targets)
- OGN deduplication with ADSBNet is complex (OGN-derived ADSB is already a provider within ADSBNet)
- Adding OGN increases scope without adding offline capability
- The proximity engine architecture supports adding OGN later

---

## Terminology

| Term | Meaning |
|------|---------|
| Traffic Proximity Awareness | The feature name |
| Nearby Traffic | Alternative label for UI elements |
| Proximity Warning | A visual indication that a pair is within a threshold |
| ADSBRx | Local ADS-B receiver subsystem |
| ADSBNet | Optional Internet-based ADS-B subsystem |
| Evaluation Radius | Maximum distance from any drone within which aircraft are evaluated |
| Normalized Target | A deduplicated aircraft representation with source provenance |

---

## Unresolved Product Decisions

| # | Decision | Recommendation | Requires Approval |
|---|----------|---------------|-------------------|
| 1 | Default threshold values (3000/1500/500m) | Use proposed values, configurable | Yes |
| 2 | Whether movement trend belongs in MVP | Include (≥3 samples over 10-15s window) | Yes |
| 3 | Subtle pulse on WARNING state | Enable by default, configurable | Yes |
| 4 | Movement trend display (text labels vs icons) | Use text: "APR" / "DIV" / "STB" / "—" | Yes |
| 5 | Panel position (bottom-right vs bottom-left) | Bottom-right (avoids drawer overlap) | Yes |
