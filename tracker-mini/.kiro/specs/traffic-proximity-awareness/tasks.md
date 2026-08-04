# MT-TRAFFIC-01 — Traffic Proximity Awareness

## Implementation Tasks

---

## Task 1: Test Fixtures and Synthetic Traffic

**Objective**: Create test infrastructure for proximity calculations.

**Files affected**:
- `tests/test_proximity_calc.py` (new)
- `tests/conftest.py` (new)
- `tests/fixtures/` (new directory)

**Dependencies**: None

**Work**:
- Set up `pytest` in the workspace with minimal configuration
- Create synthetic aircraft and drone position fixtures
- Create test scenarios: approaching, crossing, departing, stationary
- Generate known-distance pairs for haversine validation

**Tests**:
- Haversine returns correct distances for known coordinates
- Fixture data is self-consistent

**Completion criteria**: `pytest` runs successfully with passing distance calculation tests.

---

## Task 2: Distance Calculation Module

**Objective**: Implement the core haversine and pair-evaluation logic.

**Files affected**:
- `frontend/js/proximity/proximity-calc.js` (new)

**Dependencies**: Task 1 (for validation reference)

**Work**:
- Implement `haversineMeters(lat1, lon1, lat2, lon2)`
- Implement bounding-box pre-filter `isWithinRadius(droneLat, droneLon, acLat, acLon, radiusM)`
- Implement `evaluatePairs(drones, aircraft, radiusM)` returning sorted distance pairs
- Implement coordinate validation (`isValidPosition`)
- Handle edge cases: lat=0/lon=0, NaN, null, undefined

**Tests**:
- Haversine accuracy (Rome→Milan ≈ 477km, known point pairs)
- Pre-filter correctly includes/excludes
- Invalid coordinates are skipped
- Empty inputs return empty results

**Completion criteria**: All distance calculations produce correct results with known inputs.

---

## Task 3: Track Validation and Stale Handling

**Objective**: Implement freshness checking for aircraft and drone tracks.

**Files affected**:
- `frontend/js/proximity/proximity-calc.js` (extend)

**Dependencies**: Task 2

**Work**:
- Implement `isTrackFresh(track, type, config)` for aircraft and drone
- Parse drone `last_seen` (ISO 8601) into epoch ms for comparison
- Use aircraft `updatedAt` (already epoch ms) for freshness check
- Implement implausible-jump detection (>50km between updates)
- Track validation: reject lat=0/lon=0, NaN, out-of-range

**Tests**:
- Fresh track accepted
- Stale aircraft (>30s) detected
- Stale drone (>15s) detected
- Implausible jump flagged
- Invalid coordinates rejected

**Completion criteria**: Stale and invalid tracks are correctly identified and excluded from active evaluation.

---

## Task 4: Configuration

**Objective**: Add proximity configuration to the settings system.

**Files affected**:
- `config/settings.json` (add `proximity` section)
- `backend/routes/settings.py` (add proximity endpoints)
- `frontend/js/proximity/proximity-state.js` (new, reads config)

**Dependencies**: None (can be done in parallel with Tasks 2-3)

**Work**:
- Add default `proximity` object to `config/settings.json`
- Add `GET /api/settings/proximity` route
- Add `POST /api/settings/proximity` route with validation
- Frontend config loader that falls back to defaults if API unreachable
- Store proximity preferences in `localStorage` for UI toggles

**Tests**:
- GET returns current config
- POST updates and persists config
- Invalid thresholds rejected (entry must be < exit)
- Missing config section returns defaults

**Completion criteria**: Proximity configuration is readable and writable through the API, persisted to settings.json.

---

## Task 5: Proximity State Machine

**Objective**: Implement state classification with hysteresis.

**Files affected**:
- `frontend/js/proximity/proximity-state.js` (new)

**Dependencies**: Tasks 2, 3, 4

**Work**:
- Implement state enum: NORMAL, MONITOR, CAUTION, WARNING, STALE, UNKNOWN
- Implement `classifyPair(distance, currentState, config)` with hysteresis
- Maintain per-pair state: `{ pairId, state, enteredAt, distance, history[] }`
- Implement state transitions with entry/exit thresholds
- Handle stale transition: when either track goes stale, pair → STALE
- Implement stale grace period (10s) before removal
- Implement pair cleanup when tracks disappear

**Tests**:
- NORMAL → MONITOR at entry threshold
- MONITOR → NORMAL only at exit threshold (not at entry+1m)
- State doesn't flicker when distance oscillates near boundary
- Stale transition removes pair after grace
- History buffer limited to 3 entries

**Completion criteria**: State machine correctly classifies pairs with hysteresis, handles stale gracefully.

---

## Task 6: Frontend Integration (Controller)

**Objective**: Wire proximity calculation into the Dashboard lifecycle.

**Files affected**:
- `frontend/js/proximity/proximity-controller.js` (new)
- `frontend/js/dashboard.js` (add proximity initialization)
- `frontend/index.html` (add script tags)

**Dependencies**: Tasks 2, 3, 4, 5

**Work**:
- Implement `PROXIMITY.start(map)` and `PROXIMITY.stop()`
- Create 5-second timer that:
  - Reads current drones from `DRONES.markers`
  - Reads current aircraft from `markersByIcao` (via `window.AIR` access)
  - Extracts position data from marker state
  - Calls evaluation pipeline
  - Updates layer and panel
- Select reference drone (nearest to any aircraft)
- Add `window.PROXIMITY` global
- Add proximity start in `dashboard.js` after traffic modules start
- Add script tags in `index.html` for new modules
- Create `traffic-proximity` pane (z-index 670)
- Respect `proximity.enabled` setting and localStorage toggle

**Tests**:
- Controller starts/stops cleanly
- Timer fires at configured interval
- No errors when no drones are present
- No errors when no aircraft are present
- Controller reads markers without modifying them

**Completion criteria**: Proximity evaluation runs in the Dashboard every 5 seconds without interfering with existing traffic.

---

## Task 7: Map Rendering (Proximity Layer)

**Objective**: Draw proximity lines, labels, and rings on the map.

**Files affected**:
- `frontend/js/proximity/proximity-layer.js` (new)
- `frontend/css/proximity.css` (new)

**Dependencies**: Tasks 5, 6

**Work**:
- Implement distance line (dashed polyline) from reference drone to nearest proximity aircraft
- Implement distance label (Leaflet tooltip or DivIcon at line midpoint)
- Implement proximity ring (circle marker around aircraft marker)
- Color-code by state (blue/orange/red/gray)
- Implement object creation, update, and removal lifecycle
- Implement stale visual (gray, dashed)
- Implement subtle pulse CSS animation for WARNING state (configurable)
- Limit: 1 distance line, up to 5 rings, 1 label

**Tests**:
- Line appears when pair enters MONITOR
- Line color changes on state transition
- Line removed when pair returns to NORMAL
- Ring appears on aircraft marker
- All objects removed when drone disappears
- Stale objects grayed out then removed

**Completion criteria**: Map correctly shows proximity visualization that updates and cleans up.

---

## Task 8: Nearby Traffic Panel

**Objective**: Display a compact aircraft-distance summary panel.

**Files affected**:
- `frontend/js/proximity/proximity-panel.js` (new)
- `frontend/css/proximity.css` (extend)
- `frontend/index.html` (add panel container div)

**Dependencies**: Tasks 5, 6

**Work**:
- Create floating panel (bottom-right, semi-transparent)
- Show reference drone identity (model or serial truncated)
- List up to 5 nearest aircraft: callsign, distance (m or km), state badge, trend arrow
- Show/hide based on: drone present AND at least 1 aircraft within evaluation radius
- Panel updates every calculation cycle
- Compact enough for Mini Tracker display (small screen)
- Click on aircraft entry centers map on that aircraft

**Tests**:
- Panel appears when drone + aircraft are present
- Panel disappears when no drone
- List sorted by distance
- Distance format switches to km at 1000m
- Panel does not overlap critical UI elements

**Completion criteria**: Panel provides useful at-a-glance proximity information.

---

## Task 9: Hysteresis and Edge Cases

**Objective**: Validate and harden state transitions under realistic conditions.

**Files affected**:
- `frontend/js/proximity/proximity-state.js` (refine)
- `tests/test_proximity_state.py` (new)

**Dependencies**: Tasks 5, 6, 7

**Work**:
- Test hysteresis with oscillating distances (boundary ± small delta)
- Test rapid state escalation (NORMAL → WARNING in one step)
- Test de-escalation path
- Handle aircraft that appears directly in WARNING zone
- Handle drone track loss during active WARNING
- Handle configuration change while proximity is active
- Verify no orphaned map objects after edge-case sequences

**Tests**:
- 20-cycle oscillation around 3000m does not cause more than 1 state change
- Direct NORMAL → WARNING works (aircraft appears close)
- WARNING → removal works when drone disappears
- Config change mid-operation applies cleanly

**Completion criteria**: No state flickering under realistic conditions; no orphaned map objects.

---

## Task 10: Movement Analysis

**Objective**: Implement approaching/diverging determination.

**Files affected**:
- `frontend/js/proximity/proximity-calc.js` (extend)
- `frontend/js/proximity/proximity-state.js` (extend)
- `frontend/js/proximity/proximity-panel.js` (add arrow indicator)

**Dependencies**: Tasks 5, 6, 8

**Work**:
- Maintain distance history per pair (last 3 values with timestamps)
- Determine trend: approaching (↓), diverging (↑), stable/unknown (—)
- Require ≥2 samples spanning >3 seconds
- Reject if any implausible jump in history
- Display trend arrow in nearby-traffic panel
- Display trend arrow color: red=approaching, green=diverging, gray=unknown

**Tests**:
- Consistent decrease → approaching
- Consistent increase → diverging
- Mixed direction → unknown
- Single sample → unknown
- Implausible jump → unknown
- History limited to 3 entries (no memory leak)

**Completion criteria**: Movement analysis provides useful approaching/diverging indication when data quality allows.

---

## Task 11: Diagnostics and Logging

**Objective**: Add operational diagnostic information.

**Files affected**:
- `frontend/js/proximity/proximity-controller.js` (extend)

**Dependencies**: Tasks 6, 7, 8

**Work**:
- Log proximity state changes to console: `[PROXIMITY] DRONE_serial → AC_icao: MONITOR (1234m)`
- Log stale transitions
- Log configuration loads
- Add proximity status to system diagnostic panel (optional: number of pairs evaluated, current state)
- Log performance: calculation time per cycle (if > 20ms, warn)

**Tests**:
- State change produces console log
- No excessive logging in NORMAL state (log only transitions)
- Performance warning fires when threshold exceeded

**Completion criteria**: Developer and maintainer can observe proximity behavior from browser console.

---

## Task 12: Documentation

**Objective**: Update user and developer documentation.

**Files affected**:
- `frontend/help/docs/user/traffic-monitoring.md` (extend)
- `frontend/help/docs/developer/services.md` (extend with proximity note)
- `frontend/help/docs/developer/frontend.md` (extend with proximity modules)

**Dependencies**: Tasks 6-11 complete

**Work**:
- Add "Traffic Proximity Awareness" section to traffic-monitoring user doc
- Describe what the operator sees and how to configure
- Include informational/non-certified notice
- Add proximity module description to developer frontend doc
- Note that proximity is frontend-only, no backend computation

**Tests**: MkDocs build succeeds with new content.

**Completion criteria**: Documentation accurately describes the implemented feature.

---

## Task 13: Simulated Integration Tests

**Objective**: Validate complete proximity flow with synthetic data.

**Files affected**:
- `tests/test_proximity_integration.py` (new)
- `tests/fixtures/scenarios/` (new)

**Dependencies**: Tasks 1-11

**Work**:
- Create test scenarios with known outcomes:
  - Aircraft approaching stationary drone at known speed
  - Aircraft crossing at 1000m distance
  - Drone disappearing mid-WARNING
  - Multiple aircraft at various distances
  - Noisy position data
- Validate full pipeline: input → calculation → state → output
- Validate cleanup after scenario end

**Tests**:
- Each scenario produces expected state sequence
- No memory leaks (history buffers bounded)
- Cleanup leaves no orphaned state

**Completion criteria**: Integration tests pass for all defined scenarios.

---

## Task 14: Raspberry Pi Validation

**Objective**: Validate performance and behavior on physical hardware.

**Files affected**: None (read-only validation)

**Dependencies**: Tasks 1-13

**Work**:
- Record stable starting commit
- Deploy to staging installation
- Run with real ADS-B traffic + DS110 (or simulator)
- Measure CPU and memory impact
- Verify Dashboard responsiveness
- Verify stale cleanup
- Verify no interference with existing layers
- Verify panel readability on target display
- Restore stable service
- Record results in `AI_HANDOFF.md`

**Tests**: Physical observation and measurement.

**Completion criteria**: Feature operates correctly on Raspberry Pi without degrading system performance.

---

## Task 15: Final Review

**Objective**: Confirm feature completeness and push.

**Files affected**:
- `AI_HANDOFF.md` (update feature status)

**Dependencies**: Tasks 1-14

**Work**:
- Review all code for consistency
- Verify no orphaned TODO or debug code
- Verify all acceptance criteria from requirements
- Update feature status in AI_HANDOFF.md
- Verify documentation
- Final commit and push

**Completion criteria**: Feature complete, tested, documented, pushed.

---

## Task Summary

| # | Task | Type | Dependencies |
|---|------|------|-------------|
| 1 | Test fixtures and synthetic traffic | Infrastructure | None |
| 2 | Distance calculation module | Core logic | 1 |
| 3 | Track validation and stale handling | Core logic | 2 |
| 4 | Configuration | Backend + Frontend | None |
| 5 | Proximity state machine | Core logic | 2, 3, 4 |
| 6 | Frontend integration (controller) | Integration | 2, 3, 4, 5 |
| 7 | Map rendering (proximity layer) | Frontend | 5, 6 |
| 8 | Nearby traffic panel | Frontend | 5, 6 |
| 9 | Hysteresis and edge cases | Quality | 5, 6, 7 |
| 10 | Movement analysis | Core logic | 5, 6, 8 |
| 11 | Diagnostics and logging | Quality | 6, 7, 8 |
| 12 | Documentation | Docs | 6-11 |
| 13 | Simulated integration tests | Testing | 1-11 |
| 14 | Raspberry Pi validation | Hardware | 1-13 |
| 15 | Final review | Release | 1-14 |
