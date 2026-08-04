# MT-TRAFFIC-01 — Traffic Proximity Awareness

## Requirements

---

## Feature Purpose

Provide Mini Tracker operators with real-time awareness of how close drones and aircraft are to each other, directly on the operational map.

The feature is **informational and non-certified**. It does not replace certified collision-avoidance systems (TCAS, ACAS) and must not be presented as such.

---

## Scope

The feature calculates and displays horizontal distance between detected drones (Remote ID) and nearby aircraft (ADS-B, OGN), highlights proximity states on the map, and provides a compact nearby-traffic summary.

### In Scope (MVP)

- Horizontal distance calculation between drones and aircraft
- Proximity state classification with configurable thresholds
- Visual emphasis on relevant aircraft markers
- Distance line between reference drone and nearest aircraft
- Nearby-traffic summary (ordered by distance)
- Stale-track detection and cleanup
- Hysteresis to prevent state flickering
- Configuration via existing settings architecture
- Graceful handling of missing or invalid data
- Approaching/diverging indicator (if data quality supports it)

### Out of Scope (MVP)

- CPA / TCPA prediction (prepared architecturally, not implemented)
- Vertical separation warnings (altitude references are incompatible across sources)
- Sound alerts
- Persistence of proximity events to disk
- Regulatory airspace integration
- Automatic avoidance recommendations

---

## User Stories

1. **As an operator**, I want to see which aircraft are closest to a detected drone so I can assess the traffic situation.
2. **As an operator**, I want visual emphasis on aircraft that are within concerning proximity of a drone so I can focus attention where needed.
3. **As an operator**, I want to know whether an aircraft is approaching or moving away from a drone so I can anticipate changes.
4. **As an operator**, I want stale or invalid proximity information to be clearly indicated and removed so I don't act on outdated data.
5. **As an operator**, I want to configure proximity thresholds so I can adjust awareness levels to the operational scenario.

---

## Functional Requirements

### FR-01: Distance Calculation

The system must calculate the horizontal (great-circle) distance in meters between each active drone and each active aircraft within an evaluation radius.

### FR-02: Reference Drone Selection

The system must operate using the **nearest visible drone** as the reference drone when multiple drones are present. If only one drone is visible, it is automatically the reference. The operator may select a specific drone as reference in a future iteration.

### FR-03: Proximity States

The system must classify each drone-aircraft pair into one of:

| State | Meaning |
|-------|---------|
| NORMAL | Distance exceeds all thresholds |
| MONITOR | Distance within outer threshold |
| CAUTION | Distance within middle threshold |
| WARNING | Distance within inner threshold |
| STALE | One or both tracks are stale |
| UNKNOWN | Insufficient data to calculate |

### FR-04: Configurable Thresholds

Default thresholds (configurable):

| State | Entry Threshold | Exit Threshold |
|-------|-----------------|----------------|
| MONITOR | 3000 m | 3300 m |
| CAUTION | 1500 m | 1800 m |
| WARNING | 500 m | 700 m |

Exit thresholds provide hysteresis to prevent flickering.

### FR-05: Stale Track Detection

A track is stale when:
- Aircraft: not updated for > 30 seconds (matches existing `AIRCRAFT_TTL_MS`)
- Drone: `last_seen` older than 15 seconds

Stale tracks must not generate proximity warnings. Existing proximity graphics for stale tracks must be marked as STALE and removed after a grace period.

### FR-06: Map Visualization

The system must display:
- A distance line from the reference drone to the nearest aircraft in MONITOR, CAUTION, or WARNING state
- A distance label on the line showing meters or km
- Visual emphasis (color ring or marker border) on aircraft in proximity states
- Color coding: MONITOR=blue, CAUTION=orange, WARNING=red

### FR-07: Nearby Traffic Summary

The system must display a compact panel (or popup) listing up to 5 nearest aircraft to the reference drone, ordered by distance, showing: callsign, distance, state, and approaching/diverging indicator when available.

### FR-08: Movement Analysis

When at least two valid position updates exist for both drone and aircraft within a 10-second window, the system must determine whether distance is:
- Decreasing (approaching) — shown with ↓ or arrow icon
- Increasing (diverging) — shown with ↑ or arrow icon
- Stable — no indicator

Movement analysis must not trigger from a single noisy update. Minimum 2 samples with > 3 second gap required.

### FR-09: Object Lifecycle

All proximity-related map objects (lines, labels, rings) must be:
- Created when a pair enters a proximity state
- Updated on each calculation cycle
- Removed when the pair returns to NORMAL, or either track becomes stale/disappears

### FR-10: Invalid Data Handling

The system must gracefully handle:
- Missing lat/lon (skip pair)
- lat=0, lon=0 (treat as invalid)
- Missing speed/heading (disable movement analysis for that pair)
- Missing altitude (do not calculate vertical separation)
- Implausible position jumps > 50km between updates (skip update, mark uncertain)

### FR-11: Evaluation Radius

To limit computation, only aircraft within a configurable maximum radius (default: 10 km) from any active drone are evaluated.

---

## Non-Functional Requirements

### NFR-01: Performance

- Calculation must complete within 50ms for up to 50 aircraft and 5 drones
- Map rendering must not cause visible lag on Raspberry Pi hardware
- Calculation cadence: every 5 seconds (aligned with drone refresh)

### NFR-02: Offline Operation

The feature must work with local ADS-B and local Remote ID only (no Internet required).

### NFR-03: Resource Usage

- No persistent storage required for proximity state
- In-memory state only; reset on backend/frontend restart
- No additional background threads in backend

### NFR-04: Compatibility

- Must not interfere with existing traffic layers
- Must not modify existing aircraft or drone data structures
- Must preserve existing marker popup content
- Must work alongside mission layers and team markers

### NFR-05: Browser Support

Must work in modern Chromium-based browsers (used on operator devices).

---

## Acceptance Criteria

1. When a drone and an aircraft are within 3000m, the aircraft marker shows a blue proximity indicator.
2. When within 1500m, the indicator changes to orange.
3. When within 500m, the indicator changes to red.
4. A distance line is drawn from the reference drone to the nearest proximity aircraft.
5. The line includes a distance label in meters (or km if > 1000m).
6. When an aircraft moves beyond the exit threshold, the state downgrades after hysteresis.
7. When a drone or aircraft track becomes stale, proximity graphics are marked and removed.
8. The nearby-traffic panel lists aircraft ordered by distance.
9. Movement indicators show approaching (↓) or diverging (↑) when data supports it.
10. No proximity calculation occurs for tracks with invalid coordinates.
11. The feature operates correctly with only local ADS-B and Remote ID (offline).
12. CPU usage increase on Raspberry Pi is < 5% during normal operation.
13. The Dashboard remains responsive during proximity calculations.
14. Configuration changes take effect without application restart.
15. No UI element uses TCAS, ACAS, or collision-avoidance terminology.

---

## Safety and Certification Wording

The feature must include the following notice in the Dashboard UI (visible in the proximity panel or settings):

> **Traffic Proximity Awareness is informational only. It is not a certified collision-avoidance system and must not be used as the sole basis for separation decisions.**

Documentation must use the term "Traffic Proximity Awareness" or "Nearby Traffic" — never TCAS, ACAS, or collision avoidance.

---

## Excluded Behavior

- No sound alerts
- No automatic avoidance recommendations
- No CPA/TCPA prediction in MVP
- No vertical separation warnings (altitude references are incompatible)
- No persistence of proximity events
- No regulatory airspace integration
- No alert sent to DSC or Meshtastic

---

## Terminology

| Term | Meaning |
|------|---------|
| Traffic Proximity Awareness | The feature name |
| Nearby Traffic | Alternative label for UI elements |
| Proximity Warning | A visual indication that a pair is within a threshold |
| Reference Drone | The drone used as the origin for distance calculations |
| Evaluation Radius | Maximum distance from any drone within which aircraft are evaluated |

---

## Unresolved Product Decisions

| # | Decision | Recommendation | Requires Approval |
|---|----------|---------------|-------------------|
| 1 | Reference drone behavior (nearest vs selected) | Start with nearest visible drone | Yes |
| 2 | Whether approaching/diverging belongs in MVP | Include if data quality allows (≥2 samples in 10s) | Yes |
| 3 | Whether pulsing animation should be enabled for WARNING state | Subtle pulse only on WARNING, not continuous flash | Yes |
| 4 | Default threshold values (3000/1500/500) | Use proposed values, configurable | Yes |
| 5 | Whether vertical separation should be prepared but hidden | Prepare architecture, do not display in MVP | Yes |
