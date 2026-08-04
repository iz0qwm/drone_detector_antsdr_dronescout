# MT-TRAFFIC-01 — Traffic Proximity Awareness

## Implementation Tasks (Revision 2)

---

## Task 1: Test Infrastructure and Fixtures

**Objective**: Set up pytest and create synthetic traffic fixtures.

**Files affected**:
- `tests/conftest.py` (new)
- `tests/test_proximity_calc.py` (new)
- `tests/fixtures/` (new directory)
- `pytest.ini` or `pyproject.toml` (new, minimal config)

**Dependencies**: None

**Work**:
- Install pytest as dev dependency (add `tests/requirements-dev.txt`)
- Create pytest configuration
- Create synthetic aircraft fixtures (ADSBRx format, ADSBNet format)
- Create synthetic drone fixtures (DS110 format)
- Create known-distance coordinate pairs for haversine validation
- Create scenarios: approaching, crossing, departing, stationary, stale

**Test commands**:
```bash
cd backend
py -3 -m pytest tests/ -v
```

**Tests**:
- Haversine returns correct distances for known coordinates
- Fixtures are self-consistent

**Completion criteria**: `pytest` runs successfully with basic tests passing.

---

## Task 2: Distance Calculation Module

**Objective**: Implement haversine and pair-evaluation logic.

**Files affected**:
- `backend/services/proximity/__init__.py` (new)
- `backend/services/proximity/calc.py` (new)
- `tests/test_proximity_calc.py` (extend)

**Dependencies**: Task 1

**Work**:
- Implement `haversine_meters(lat1, lon1, lat2, lon2)`
- Implement bounding-box pre-filter `is_within_radius()`
- Implement coordinate validation `is_valid_position(lat, lon, source_type)`
  - Drones: reject (0.0, 0.0) as ODID sentinel
  - Aircraft: accept (0.0, 0.0) (readsb omits fields for missing, doesn't use sentinel)
- Implement `evaluate_pairs(drones, targets, radius_m)` returning distance pairs

**Tests**:
- Haversine accuracy (Rome→Milan ≈477km, short distances verified)
- Pre-filter includes/excludes correctly
- (0,0) rejected for drones, accepted for aircraft
- Invalid coordinates (None, NaN, out-of-range) skipped
- Empty inputs return empty results

**Completion criteria**: Distance calculations correct for all validation cases.

---

## Task 3: Aircraft Normalization and Source Merge

**Objective**: Implement the normalized target model with source provenance and deduplication.

**Files affected**:
- `backend/services/proximity/normalize.py` (new)
- `tests/test_proximity_normalize.py` (new)

**Dependencies**: Task 2

**Work**:
- Define `NormalizedTarget` dataclass
- Implement `normalize_adsbrx_aircraft(aircraft_list)` → list of normalized targets
- Implement `normalize_adsbnet_aircraft(aircraft_list)` → list of normalized targets
- Implement `merge_targets(adsbrx_targets, adsbnet_targets)` with:
  - ICAO-based deduplication
  - Source precedence (fresh local > fresh net > stale local > stale net)
  - Never replace newer position with older
  - Preserve both source timestamps
  - Track `primary_source` and `sources` list
- Implement `SourceHealth` tracking

**Tests**:
- ADSBRx-only normalization correct
- ADSBNet-only normalization correct
- Duplicate ICAO merged correctly (ADSBRx preferred when fresh)
- Source timestamps preserved independently
- Stale ADSBRx + fresh ADSBNet → ADSBNet primary
- Fresh ADSBRx + stale ADSBNet → ADSBRx primary
- Missing ICAO creates synthetic ID (no merge possible)
- Callsign conflict → prefer ADSBRx

**Completion criteria**: Normalized targets correctly represent merged state with provenance.

---

## Task 4: Track Freshness and Stale Handling

**Objective**: Implement per-source and per-target staleness detection.

**Files affected**:
- `backend/services/proximity/normalize.py` (extend)
- `backend/services/proximity/state.py` (new, partial)
- `tests/test_proximity_stale.py` (new)

**Dependencies**: Task 3

**Work**:
- Implement source-track freshness check (per source, per target)
- Implement normalized-target staleness (stale when ALL sources stale)
- Implement target retention (keep stale target for `target_retention_ms`)
- Implement drone freshness check (parse ISO 8601 `last_seen`)
- Implement implausible-jump detection (>50km between updates)

**Tests**:
- Fresh ADSBRx track → target fresh
- Stale ADSBRx + fresh ADSBNet → target fresh
- Both sources stale → target stale
- Stale target retained within retention window
- Target removed after retention expires
- Drone stale after 15s since `last_seen`
- Implausible jump flagged

**Completion criteria**: Staleness correctly determined per-source and per-target.

---

## Task 5: Configuration

**Objective**: Add proximity configuration to settings system.

**Files affected**:
- `config/settings.json` (add `proximity` section)
- `backend/services/proximity/config.py` (new)
- `backend/routes/proximity.py` (new, config endpoints)
- `tests/test_proximity_config.py` (new)

**Dependencies**: None (parallel with Tasks 2-4)

**Work**:
- Add default `proximity` section to `config/settings.json`
- Implement `get_proximity_config()` with fallback defaults
- Implement `update_proximity_config(data)` with validation
- Add routes: `GET /api/proximity/config`, `POST /api/proximity/config`
- Validate: entry < exit thresholds, positive radii, reasonable ranges
- Respect existing patterns (`SETTINGS`, `save_settings()`)

**Tests**:
- GET returns current config (or defaults if section missing)
- POST updates and persists
- Invalid thresholds rejected
- Missing section → defaults used (no crash)
- `adsb_net_enabled` respects Internet availability at runtime

**Completion criteria**: Proximity configuration readable, writable, validated.

---

## Task 6: Proximity State Machine

**Objective**: Implement state classification with hysteresis.

**Files affected**:
- `backend/services/proximity/state.py` (new/extend)
- `tests/test_proximity_state.py` (new)

**Dependencies**: Tasks 2, 4, 5

**Work**:
- Define state enum: NORMAL, MONITOR, CAUTION, WARNING, STALE, UNKNOWN
- Implement `ProximityPair` with: pair_id, drone_id, target_id, state, distance, history, entered_at
- Implement `classify_pair(distance, current_state, config)` with hysteresis
- Implement direct escalation (aircraft appears directly in WARNING range)
- Implement stale transition (either track stale → pair STALE)
- Implement pair cleanup (stale grace → removal)
- Implement ranking: severity first, distance second, deterministic tie-break

**Tests**:
- NORMAL → MONITOR at entry threshold
- MONITOR → NORMAL only at exit threshold (hysteresis)
- Direct NORMAL → WARNING when aircraft appears close
- Oscillation around boundary doesn't flicker (20 cycles at boundary ±10m)
- Stale transition works
- Pair removed after grace
- Ranking: WARNING pairs before CAUTION before MONITOR

**Completion criteria**: State machine correctly classifies all pairs with hysteresis.

---

## Task 7: Movement Trend Analysis

**Objective**: Implement approaching/diverging determination.

**Files affected**:
- `backend/services/proximity/trend.py` (new)
- `tests/test_proximity_trend.py` (new)

**Dependencies**: Task 6

**Work**:
- Implement per-pair distance history (circular buffer, 4 entries)
- Determine trend: APPROACHING, DIVERGING, STABLE, UNKNOWN
- Require ≥3 samples spanning ≥10s
- Apply 50m deadband
- Handle implausible jump → UNKNOWN + reset history
- Handle insufficient data → UNKNOWN

**Tests**:
- Consistent decrease → APPROACHING
- Consistent increase → DIVERGING
- Change within deadband → STABLE
- Mixed directions → UNKNOWN
- Only 2 samples → UNKNOWN
- Jump in history → UNKNOWN + history reset
- Buffer limited to 4 entries

**Completion criteria**: Movement trend correctly determined when data quality allows.

---

## Task 8: Proximity Engine Integration

**Objective**: Wire all components into the main evaluation loop.

**Files affected**:
- `backend/services/proximity/engine.py` (new)
- `backend/routes/proximity.py` (extend with status endpoint)
- `backend/app.py` (register blueprint, start engine timer)
- `tests/test_proximity_engine.py` (new)

**Dependencies**: Tasks 2-7

**Work**:
- Implement `ProximityEngine` class:
  - Reads ADSBRx via `air_local.get_local_aircraft()` (wide bounds)
  - Reads optional ADSBNet via `air_network.get_network_aircraft()` (if enabled + Internet)
  - Reads drones via `ds110.get_aircraft()`
  - Normalizes and merges aircraft
  - Evaluates all pairs
  - Manages state lifecycle
  - Updates trend history
  - Returns ranked results
- Implement 5-second calculation timer (threading.Timer or loop in main tick)
- Implement `GET /api/proximity/status` returning current results
- Register `proximity_bp` in `app.py`
- Start engine timer in `app.py` startup sequence

**Tests**:
- Engine produces correct results with ADSBRx + drones
- Engine works without ADSBNet (disabled)
- Engine works with ADSBNet enabled
- Engine handles no drones gracefully
- Engine handles no aircraft gracefully
- Cycle completes within 100ms with 50 aircraft × 5 drones

**Completion criteria**: Backend proximity engine runs autonomously, serves correct API responses.

---

## Task 9: Frontend Proximity Controller

**Objective**: Poll backend API and manage frontend lifecycle.

**Files affected**:
- `frontend/js/proximity/proximity-controller.js` (new)
- `frontend/js/dashboard.js` (add proximity init + pane)
- `frontend/index.html` (add script tags + panel container)

**Dependencies**: Task 8

**Work**:
- Implement `PROXIMITY.start(map)` and `PROXIMITY.stop()`
- Create `traffic-proximity` pane (z-index 670)
- Poll `GET /api/proximity/status` every 5s
- Pass results to layer and panel modules
- Respect `proximity.enabled` from settings
- Add `window.PROXIMITY` global
- Add script tags in `index.html`
- Start proximity after traffic modules in `dashboard.js`

**Tests**:
- Controller starts/stops cleanly
- Polling works at configured interval
- No errors when API returns empty pairs
- Pane created at correct z-index

**Completion criteria**: Frontend polls backend and passes results to rendering.

---

## Task 10: Map Rendering (Proximity Layer)

**Objective**: Draw proximity lines, labels, and rings.

**Files affected**:
- `frontend/js/proximity/proximity-layer.js` (new)
- `frontend/css/proximity.css` (new)

**Dependencies**: Task 9

**Work**:
- Implement distance line (1 line, highest-priority pair)
  - Dashed (MONITOR, CAUTION), solid (WARNING), dotted (STALE)
  - Colored by state
- Implement distance label (DivIcon at midpoint)
- Implement proximity rings on up to 5 aircraft
  - Color + pattern matching state
- Implement pulse CSS animation for WARNING (configurable)
- Implement complete cleanup on each cycle (remove old, add new from API)
- All objects in `traffic-proximity` pane
- Accessibility: color + pattern + text label

**Tests**:
- Line appears for WARNING pair
- Line color/pattern changes on state change
- All objects removed when no pairs in API response
- Rings appear on correct aircraft positions
- Pulse animation only on WARNING (if enabled)

**Completion criteria**: Map correctly visualizes proximity with accessible indicators.

---

## Task 11: Nearby Traffic Panel

**Objective**: Display compact proximity summary.

**Files affected**:
- `frontend/js/proximity/proximity-panel.js` (new)
- `frontend/css/proximity.css` (extend)
- `frontend/index.html` (panel container already added in Task 9)

**Dependencies**: Task 9

**Work**:
- Floating panel, bottom-right, semi-transparent
- Show up to `max_panel_entries` pairs from API response
- Each entry: drone label → aircraft label, distance, state badge, trend label
- Source label (RX/NET/RX+NET) visible but not prominent
- Show/hide: visible when pairs exist, hidden otherwise
- Compact for Mini Tracker display
- No continuous error displayed when ADSBNet is offline
- Click entry → center map on that aircraft

**Tests**:
- Panel appears when pairs exist
- Panel hidden when no pairs
- Entries sorted correctly (from API, pre-sorted)
- Distance format: meters if <1000, km if ≥1000
- Source labels shown correctly

**Completion criteria**: Panel provides clear at-a-glance proximity information.

---

## Task 12: Source Switching and Edge Cases

**Objective**: Validate all source-transition scenarios.

**Files affected**:
- `tests/test_proximity_source_switching.py` (new)
- `backend/services/proximity/engine.py` (bug fixes if needed)

**Dependencies**: Tasks 3, 4, 8

**Work**:
- Test all 10 network-loss/source-switching scenarios from design §16
- Verify pair identity preserved across source switches
- Verify no duplicate targets after source transitions
- Verify fresh ADSBRx never made stale by ADSBNet loss
- Verify graceful degradation when Internet disappears
- Verify ADSBNet recovery without restart
- Verify ADSBRx restart handling

**Tests (from requirements)**:
1. ADSBRx-only offline operation
2. ADSBNet disabled by user
3. ADSBNet enabled but Internet unavailable
4. ADSBNet provider failure
5. Same aircraft from both sources
6. Fresh ADSBRx + stale ADSBNet
7. Stale ADSBRx + fresh ADSBNet
8. Source switch without duplicate
9. Internet loss during WARNING
10. ADSBNet enabled during runtime
11. ADSBNet disabled during runtime
12. Multiple drones + multiple aircraft
13. Network aircraft outside local reception
14. Local receiver restart
15. No aircraft from any source
16. Invalid source timestamps
17. Conflicting identifiers/callsigns
18. Full offline (ADSBRx + Remote ID only)

**Completion criteria**: All source transitions pass without false stale, duplicates, or lost pairs.

---

## Task 13: Hysteresis Hardening

**Objective**: Validate state machine under realistic edge conditions.

**Files affected**:
- `tests/test_proximity_hysteresis.py` (new)
- `backend/services/proximity/state.py` (fixes if needed)

**Dependencies**: Tasks 6, 8

**Work**:
- Test 30-cycle oscillation at each threshold boundary (±10m, ±50m)
- Test rapid escalation (NORMAL→WARNING in one step)
- Test de-escalation path
- Test drone track loss during WARNING
- Test config change mid-operation
- Verify no orphaned state after edge-case sequences

**Tests**:
- Oscillation produces max 1 state change
- Direct WARNING entry works
- Config update applies to next cycle cleanly
- No orphaned pairs in engine state

**Completion criteria**: No flickering, no orphaned state under adversarial conditions.

---

## Task 14: Diagnostics and Logging

**Objective**: Add operational diagnostics.

**Files affected**:
- `backend/services/proximity/engine.py` (extend with logging)
- `frontend/js/proximity/proximity-controller.js` (extend with console logging)

**Dependencies**: Tasks 8, 9

**Work**:
- Backend: log state transitions via `services.logger.log("PROXIMITY", ...)`
- Backend: log cycle performance when > 50ms
- Backend: log source health transitions
- Frontend: console log `[PROXIMITY]` on state changes and errors
- Include `calculation_time_ms` in API response for monitoring
- No excessive logging in steady state (transitions only)

**Tests**:
- State change produces backend log entry
- Performance warning fires when threshold exceeded
- No log spam in normal operation

**Completion criteria**: Developers and maintainers can observe proximity behavior from logs.

---

## Task 15: Documentation

**Objective**: Update user and developer documentation.

**Files affected**:
- `frontend/help/docs/user/traffic-monitoring.md` (extend)
- `frontend/help/docs/developer/services.md` (extend)
- `frontend/help/docs/developer/api.md` (extend)

**Dependencies**: Tasks 8-11 complete

**Work**:
- Add "Traffic Proximity Awareness" section to traffic-monitoring user doc
- Describe operator experience and configuration
- Include informational/non-certified notice
- Add proximity engine description to developer services doc
- Add proximity API to developer API doc
- Follow `DOCUMENTATION.md` rules

**Tests**: MkDocs build succeeds.

**Completion criteria**: Documentation accurately describes implemented feature.

---

## Task 16: Simulated Integration Tests

**Objective**: End-to-end scenarios with synthetic data.

**Files affected**:
- `tests/test_proximity_integration.py` (new)
- `tests/fixtures/scenarios/` (new)

**Dependencies**: Tasks 1-14

**Work**:
- Full pipeline scenarios with known outcomes
- Verify: input → normalize → merge → evaluate → state → API response
- Scenarios: approaching drone-aircraft, crossing, multiple drones, stale data
- Verify cleanup leaves no orphaned state
- Verify performance under load (50 aircraft × 5 drones)

**Tests**:
- Each scenario produces expected ranked pair list
- No memory leaks (history buffers bounded)
- Cleanup complete after scenario end
- Performance within 100ms

**Completion criteria**: Integration tests validate the complete pipeline.

---

## Task 17: Raspberry Pi Physical Validation

**Objective**: Validate on real hardware.

**Files affected**: None (read-only validation)

**Dependencies**: Tasks 1-16

**Work**:

### Offline Local Mode
- Disconnect Internet (do not change ADSBNet config permanently)
- Verify ADSBRx continues receiving
- Verify proximity works with ADSBRx + Remote ID
- Verify no repeated network errors disrupt Dashboard

### Combined Mode
- Enable ADSBNet, verify Internet available
- Receive local and network aircraft
- Verify deduplication (no duplicate markers or pairs)
- Verify source provenance in panel

### Network-Loss Mode
- Start with ADSBNet operational
- Remove Internet connectivity
- Verify ADSBRx tracks remain active
- Verify local proximity warnings continue
- Restore Internet, verify ADSBNet recovers

### Performance
- 5-minute baseline (no proximity)
- 5-minute feature-enabled (local-only: ~5 aircraft, 1 drone)
- 5-minute feature-enabled (combined: ~20 aircraft, 2 drones)
- Measure: backend CPU, browser responsiveness, API response time, memory

Record results in `AI_HANDOFF.md`. Restore stable service after testing.

**Completion criteria**: Feature operates correctly on Raspberry Pi in all modes without degrading performance.

---

## Task 18: Final Review and Push

**Objective**: Confirm feature completeness.

**Files affected**:
- `AI_HANDOFF.md` (update feature status)

**Dependencies**: Tasks 1-17

**Work**:
- Review all code for consistency
- Verify no orphaned TODO or debug code
- Verify all acceptance criteria met
- Update AI_HANDOFF.md feature status
- Final commit and push

**Completion criteria**: Feature complete, tested, documented, pushed.

---

## Task Summary

| # | Task | Type | Dependencies |
|---|------|------|-------------|
| 1 | Test infrastructure and fixtures | Infrastructure | None |
| 2 | Distance calculation module | Core logic | 1 |
| 3 | Aircraft normalization and source merge | Core logic | 2 |
| 4 | Track freshness and stale handling | Core logic | 3 |
| 5 | Configuration | Backend | None (parallel) |
| 6 | Proximity state machine | Core logic | 2, 4, 5 |
| 7 | Movement trend analysis | Core logic | 6 |
| 8 | Proximity engine integration | Backend | 2-7 |
| 9 | Frontend proximity controller | Frontend | 8 |
| 10 | Map rendering (proximity layer) | Frontend | 9 |
| 11 | Nearby traffic panel | Frontend | 9 |
| 12 | Source switching and edge cases | Testing | 3, 4, 8 |
| 13 | Hysteresis hardening | Testing | 6, 8 |
| 14 | Diagnostics and logging | Quality | 8, 9 |
| 15 | Documentation | Docs | 8-11 |
| 16 | Simulated integration tests | Testing | 1-14 |
| 17 | Raspberry Pi physical validation | Hardware | 1-16 |
| 18 | Final review and push | Release | 1-17 |
