# Mini Tracker AI Development Status

Last updated: 2026-08-04

## Purpose

This file records the current development status of the Mini Tracker project during AI-assisted development.

For the current development phase, Kiro is the primary development agent for the Mini Tracker repository.

This file must be read before starting any implementation task and updated after every meaningful development phase.

It provides:

* current priorities;
* active feature ownership;
* branch and deployment status;
* architectural decisions;
* physical Mini Tracker validation results;
* known limitations;
* pending work;
* rollback information.

This file does not replace:

* `AGENTS.md`;
* `DOCUMENTATION.md`;
* Kiro steering files;
* feature specifications;
* Git history;
* the official product documentation under `frontend/help/docs/`.

---

## Current Ownership

### Mini Tracker repository

Primary development agent: Kiro

Kiro is the only development agent working on Mini Tracker during this phase.

Kiro may:

* inspect the entire Mini Tracker workspace under tracker-mini;
* create and update Feature Specs;
* modify backend and frontend code;
* add and update tests;
* update documentation;
* commit directly on the current branch;
* push completed and tested work to the current remote branch;
* connect to the physical Mini Tracker through the LAN;
* deploy committed versions to the staging installation;
* execute hardware validation.

Raffaello remains responsible for:

* approving requirements and design decisions;
* approving potentially destructive system operations;
* validating operational behavior;
* deciding when a feature is ready for operational use.

---

## Required Reading

Before starting work, read:

* `AGENTS.md`;
* `AI_HANDOFF.md`;
* `DOCUMENTATION.md`;
* `.kiro/steering/`;
* the relevant Feature Spec;
* the relevant source code;
* the corresponding documentation under `frontend/help/docs/`.

The current source code is the primary technical source of truth.

The physical Mini Tracker installation is the primary source of truth for deployed hardware configuration and runtime behavior.

Differences between repository code, documentation and deployed behavior must be reported explicitly.

---

## Current Development Phase

### Phase: Repository and Device Onboarding

Status: **In Progress** — local workspace inspection complete, physical device inspection pending SSH credentials.

Completed:

* Local workspace structure inspected
* Backend architecture understood (Flask + service modules + threaded workers)
* Frontend architecture understood (static JS + Leaflet + polling)
* Configuration system documented
* Hardware integration points identified from source code
* Six steering files created under `.kiro/steering/`
* Git repository state verified
* Mini Tracker reachable on LAN (ping OK to 192.168.1.115)

Pending:

* SSH access to physical device for runtime inspection
* Comparison of deployed code vs repository
* Physical hardware verification (serial devices, I2C, GPIO, services)
* Staging environment verification

Expected outputs:

* `.kiro/steering/product.md` ✓
* `.kiro/steering/tech.md` ✓
* `.kiro/steering/structure.md` ✓
* `.kiro/steering/workflow.md` ✓
* `.kiro/steering/hardware.md` ✓
* `.kiro/steering/documentation.md` ✓
* repository and deployment comparison — pending SSH
* recommended staging workflow — documented in workflow.md
* recommended rollback workflow — documented in workflow.md
* first Feature Spec scope — Traffic Proximity Awareness

---

## Planned Development Sequence

The current feature order is:

1. Traffic Proximity Awareness
2. Meshtastic Operational Network
3. DSC Operational Area Synchronization

Only one major Feature Spec should be actively implemented at a time unless the work is explicitly divided into independent components.

---

## Feature Status

### MT-TRAFFIC-01 — Traffic Proximity Awareness

Status: **Specified (Revision 3 — Final)** — awaiting review
Owner: Kiro
Specification: `.kiro/specs/traffic-proximity-awareness/` (requirements.md, design.md, tasks.md)
Working branch: Current repository branch (main)
Starting commit: To be recorded at implementation start
Latest commit: Not started
Push status: Not started

Key design decisions (Revision 3 — Final):
- Authoritative proximity engine in backend (Python), not frontend
- All valid drone-aircraft pairs evaluated (not single reference drone)
- ADSBRx = primary local source, works offline
- ADSBNet = optional enrichment; ONE unified authoritative backend setting (migrated from localStorage)
- ADSBNet snapshot cache: network providers fetched at 15s interval, proximity engine reads cache (no blocking)
- Managed daemon thread worker with idempotent start/stop (follows existing DS110/Meshtastic pattern)
- Normalized target model with source provenance and ICAO deduplication
- Timestamp-aware source precedence with configurable tie window (3s default, ADSBRx preferred on tie)
- Provider health based on execution success, not aircraft count (empty sky ≠ failure)
- OGN/FLARM deferred from MVP
- Source health tracked separately from individual track freshness
- Movement trend: ≥3 samples, 10-15s window, 50m deadband, text labels (not vertical arrows); speed/heading NOT required
- Backend exposes `GET /api/proximity/status` — fast, non-blocking, returns latest snapshot
- Stale pairs remain in API during grace period, removed after expiry; frontend trusts API lifecycle
- Panel hidden when no non-NORMAL pairs exist (no continuous "no aircraft" message)
- Accessibility: color + line pattern + text label (not color alone)
- Coordinate validation: (0,0) rejected for drones only (ODID sentinel), accepted for aircraft
- Test framework: pytest at workspace root, platform-independent commands

Goal:

Provide an operational visualization of proximity between drones and aircraft.

Expected capabilities:

* calculate horizontal distance between drones and aircraft;
* identify stale traffic data;
* identify approaching and diverging tracks;
* display proximity information on the map;
* use configurable attention levels;
* change map rendering according to proximity state;
* provide clear visual warnings without excessive flashing;
* prepare support for closest-point calculations;
* prepare support for CPA and TCPA;
* handle missing or incompatible altitude references safely.

The initial implementation must not present itself as a certified TCAS or collision-avoidance system.

The interface should use terminology such as:

* Traffic Proximity Awareness;
* Traffic Awareness;
* Proximity Warning.

The feature must be described as informational and non-certified.

---

### MT-MESH-02 — Meshtastic Operational Network

Status: Planned
Owner: Kiro
Specification: Not created
Working branch: Current repository branch
Starting commit: To be recorded
Latest commit: Not started
Push status: Not started

Goal:

Improve operational communication between field operators and the Mini Tracker control center through Meshtastic.

Expected capabilities:

* versioned message envelope;
* heartbeat and node presence;
* operator and team status;
* text messages;
* operational tasks;
* acknowledgements;
* emergency messages;
* traffic proximity alerts;
* message priority;
* TTL and expiration;
* duplicate detection;
* controlled retries;
* persistent inbox and outbox;
* recovery after process or device restart;
* peer last-seen state;
* bandwidth-aware message handling;
* rate limiting;
* degraded-network testing.

The design must account for:

* limited airtime;
* delayed packets;
* packet loss;
* duplicate packets;
* out-of-order delivery;
* temporary disconnection;
* tracker restart;
* Meshtastic node restart.

---

### MT-DSC-03 — DSC Operational Area Synchronization

Status: Planned
Owner: Kiro
Specification: Not created
Working branch: Current repository branch
Starting commit: To be recorded
Latest commit: Not started
Push status: Not started

Goal:

Integrate Mini Tracker with Drone Sky Check so an operational area can be published and displayed during field activities.

Example activities include:

* exercises;
* search and rescue;
* missing-person recovery;
* civil protection operations;
* technical tests;
* coordinated UAS operations.

Expected capabilities:

* Mini Tracker device identification;
* authenticated communication with DSC;
* operational session creation;
* operational area geometry;
* activity type and description;
* start and expected end time;
* heartbeat;
* active, stale, ended and expired states;
* offline queue;
* controlled retry;
* session closure;
* public and operator-only information;
* explicit distinction from regulatory airspace restrictions.

Operational areas must be presented as advisory information.

They must not visually or semantically resemble:

* prohibited areas;
* official UAS geographical zones;
* NOTAM restrictions;
* controlled airspace;
* regulatory limitations.

The Mini Tracker–DSC data contract must be reviewed before implementation begins.

---

## Repository Workflow

All source code changes must be made in the local development clone of the repository.

Do not use the physical Mini Tracker as the primary code-editing environment.

### Before beginning a development task

1. Verify the current branch.
2. Verify the current commit.
3. Record the commit as the stable starting point.
4. Verify that there are no unrelated local modifications.
5. Pull the latest changes from the current remote branch.
6. Confirm that the local branch is synchronized with the remote branch.

### For every feature or meaningful task

1. Read the relevant Feature Spec.
2. Implement the work incrementally.
3. Create small and focused commits.
4. Run all relevant available tests.
5. Perform physical Mini Tracker validation when required.
6. Update documentation when required.
7. Update `AI_HANDOFF.md`.
8. Push the completed and tested commits to the current remote branch.

A separate feature branch is not required. Kiro works directly on the current checked-out branch.

Raffaello retrieves completed work using a normal `git pull`.

### Git safety

The following are strictly prohibited:

* `git push --force` or any force-push variant;
* rewriting, rebasing or amending already pushed commits;
* deleting remote history;
* staging files outside `tracker-mini`;
* committing credentials, passwords, tokens or device-specific secrets;
* mixing unrelated changes in one commit;
* using repository-wide `git add -A` or `git commit -a`.

Because `tracker-mini` is inside a larger Git repository, always use workspace-scoped commands:

```bash
git status --short -- .
git diff -- .
git diff --cached -- .
git add -- .
```

Before every commit and push, verify that every changed or staged file belongs to `tracker-mini`.

---

## Physical Mini Tracker Access

The physical Mini Tracker may be accessed through the LAN using SSH.

During repository onboarding, access must remain read-only.

Before executing changes on the physical tracker, identify:

* deployment directory;
* Git branch;
* deployed commit;
* active Python environment;
* active system services;
* startup mechanism;
* serial devices;
* I2C devices;
* GPIO usage;
* network interfaces;
* access point configuration;
* relevant logs;
* local persistent data;
* available disk space.

Do not expose or store:

* passwords;
* tokens;
* private keys;
* Firebase credentials;
* API credentials;
* Wi-Fi credentials;
* private device configuration.

---

## Deployment Layout

The stable Mini Tracker installation should remain separate from development testing whenever practical.

Recommended layout:

```text
/home/pi/tracker-mini
/home/pi/tracker-mini-staging
```

Stable installation:

```text
/home/pi/tracker-mini
```

Development and integration testing:

```text
/home/pi/tracker-mini-staging
```

The exact paths must be verified on the physical device before use.

Do not assume these paths already exist.

---

## Hardware Test Rules

Only one process may access an exclusive serial, GPIO or hardware interface at a time.

Before testing a staging version:

1. identify the stable service using the hardware;
2. record its current state;
3. verify the rollback procedure;
4. stop only the required service;
5. start the staging version;
6. perform the test;
7. collect logs;
8. stop the staging version;
9. restore the stable service;
10. verify normal operation.

Development-machine tests do not prove correct hardware operation.

Mocked tests must be reported as mocked tests.

Physical validation must identify:

* device used;
* interface used;
* test conditions;
* observed result;
* logs collected;
* known limitations.

---

## Restricted Operations

The following operations require explicit approval before execution on the physical Mini Tracker:

```text
sudo commands that modify the system
package installation or removal
systemctl enable or disable
network configuration changes
access point configuration changes
firewall changes
serial configuration changes
GPIO reassignment
I2C configuration changes
filesystem deletion
git reset --hard
database deletion
credential changes
operating system upgrades
firmware changes
```

Read-only inspection commands do not require separate approval unless they expose secrets or private data.

---

## Rollback Requirements

Before every physical deployment, record:

* current branch;
* stable commit (the commit running on the device before deployment);
* deployment commit (the commit being deployed);
* services that will be stopped;
* services that will be started;
* configuration files affected;
* databases affected;
* rollback commands;
* expected restoration checks.

Rollback is complete only when:

* the stable service is running;
* the expected hardware devices are available;
* the dashboard is reachable;
* critical services report the expected state;
* no new persistent error remains in the logs.

The physical staging installation (`/home/pi/tracker-mini-staging`) may be used when safe and practical for hardware testing. It does not require a separate Git branch. A committed version from the current branch may be deployed to the staging installation.

---

## Documentation

All documentation changes must follow `DOCUMENTATION.md`.

Documentation sources are stored under:

```text
frontend/help/docs/
```

MkDocs configuration is stored under:

```text
frontend/help/mkdocs.yml
```

Never edit generated documentation under:

```text
frontend/help/site/
```

Documentation must:

* be written in English;
* reflect verified implementation;
* preserve existing terminology;
* update existing documents where possible;
* avoid duplicate documents;
* use existing screenshots when appropriate;
* keep diffs focused;
* update the documentation status when required.

---

## Current Architectural Decisions

The following decisions are currently approved:

* Kiro is the only Mini Tracker development agent during this phase.
* Kiro works directly on the current repository branch.
* Separate feature branches are not required.
* Work is divided through Feature Specs and focused commits.
* Completed and tested work is pushed to the current remote branch.
* Raffaello synchronizes through a normal `git pull`.
* Pushed history must never be rewritten.
* Major features must use separate Feature Specs.
* Features will be developed sequentially.
* Development changes are made in the local Git clone.
* The physical Mini Tracker is used for integration and hardware validation.
* Direct source-code editing on the physical tracker is discouraged.
* Hardware configuration changes require explicit review.
* Offline operation must be considered for every network-dependent feature.
* Stale, delayed and duplicate traffic data must be handled explicitly.
* Traffic proximity warnings are informational and non-certified.
* DSC operational areas are advisory and not regulatory airspace restrictions.
* Documentation must follow `DOCUMENTATION.md`.

---

## Current Deployment Status

Repository remote: `https://github.com/iz0qwm/drone_detector_antsdr_dronescout`
Development branch: `main`
Development commit: `512e341` ("modifiche per kiro")
Physical deployment path: `/home/pi/tracker-mini` (to be verified via SSH)
Physical branch: To be verified via SSH
Physical commit: To be verified via SSH
Python version: To be verified via SSH
Operating system: Raspberry Pi OS (Debian-based, to be confirmed via SSH)
Service manager: systemd (`tracker-mini.service`)
Staging environment: Not yet created

---

## Test Status

Automated test framework: **None** — no test files exist in the repository
Backend tests: Not present
Frontend tests: Not present
Hardware mocks: Not present
Physical integration tests: Not started
Traffic simulation tools: Not present
Meshtastic test support: Not present
DSC integration test support: Not present

---

## Known Constraints

* Mini Tracker may operate without Internet access.
* Internet connectivity may be intermittent.
* Meshtastic bandwidth and airtime are limited.
* Hardware devices may not exist on development computers.
* Multiple services may compete for exclusive hardware interfaces.
* ADS-B, Remote ID and GPS data may use different altitude references.
* Traffic data may be delayed, incomplete, duplicated or stale.
* Physical tracker configuration may differ from repository defaults.
* Device-specific configuration must not be committed.
* Operational warnings must avoid creating a false impression of certification.
* DSC operational areas must remain clearly distinct from official airspace data.
* No automated tests exist — all validation is manual.
* Application logs are in-memory only (lost on restart).
* Flash storage is the single point of persistence (power loss risk).

---

## Active Work Record

```text
Task: Remote ID Map Visibility Fix
Feature: Remote ID dashboard rendering
Owner: Codex
Working branch: main
Starting commit: 0c59f29
Latest commit: Pending
Push status: Pending
Status: Local fix complete, physical Mini Tracker validation pending
Started: 2026-08-04
Last updated: 2026-08-04

Observed issue:
  - Mini Tracker logs showed DS110 receiving Dronetag Beacon 1596A34EE1D16FD with valid coordinates.
  - The drone was sent to DSC, proving backend decoding and DS110 ingestion were working.
  - The marker did not appear on the Dashboard map.

Root cause:
  - Dashboard Remote ID polling could be blocked by browser-local `localStorage("droneNetworkEnabled") == "false"` even when the backend DS110 service was active.
  - `initTrafficSettings()` updated the Remote ID checkbox from `/api/ds110/status`, but did not start drone map polling after discovering that DS110 was already enabled.
  - `DRONES.stopDroneTraffic()` recursively called itself instead of clearing the drone layer.

Files modified:
  frontend/js/dashboard.js
  frontend/js/drones/drone-controller.js
  frontend/help/docs/developer/frontend.md

Implementation:
  - Remote ID map polling now starts from backend DS110 status (`/api/ds110/status`) during map initialization.
  - When traffic settings load and DS110 is already enabled, drone polling is started if the map is ready.
  - Removed dependence on stale browser-local `droneNetworkEnabled` for Remote ID display.
  - Fixed `DRONES.stopDroneTraffic()` to clear the drone layer instead of recursing.
  - Updated developer frontend documentation to describe Remote ID as backend-state driven.

Tests/checks:
  - Passed: `node --check frontend/js/dashboard.js`
  - Passed: `node --check frontend/js/drones/drone-controller.js`

Known limitations:
  - Real marker display must be validated on the physical Mini Tracker with an active Remote ID source.
  - Local workspace ZIP files (`tracker-mini.zip` removed, `mini-tracker.zip` added) appear user-managed and were not modified by this task.
```

```text
Task: ADSBNet Multi-Provider Update
Feature: Network ADS-B hardening
Owner: Codex
Working branch: main
Starting commit: 0c59f29
Latest commit: Pending
Push status: Pending
Status: Local implementation complete, physical Mini Tracker validation pending
Started: 2026-08-04
Last updated: 2026-08-04

Files created:
  tests/test_air_network.py

Files modified:
  backend/services/air_network.py
  frontend/help/docs/hardware/ads-b.md
  frontend/help/docs/developer/services.md
  frontend/help/docs/developer/architecture.md
  frontend/help/docs/developer/api.md

Implementation:
  - Added direct backend network ADS-B provider support for Airplanes.live and ADSB.lol.
  - Uses provider point APIs directly from Mini Tracker backend; no browser proxy is required.
  - Derives point-query center and radius from Dashboard map bounds, caps provider radius at 250 NM, then filters returned aircraft back to map bounds.
  - Keeps provider failures isolated with per-provider error handling.
  - Fetches active ADS-B network providers in parallel to avoid sequential provider delays.
  - Normalizes readsb-compatible provider data into the existing Mini Tracker aircraft schema.
  - Merges by ICAO and preserves combined source provenance in the `source` field.
  - SolarMonitor ADS-B feed is intentionally paused and is not called by the active provider list.
  - OGN-derived ADS-B and OpenSky remain active network ADS-B sources.

Documentation:
  - Updated ADS-B hardware documentation to list active network ADS-B providers.
  - Updated developer services, architecture and API docs to reflect active source counts.
  - Did not edit generated `frontend/help/site/`.

Tests/checks:
  - Passed: Python syntax compile for `backend/services/air_network.py` and `tests/test_air_network.py`
    Command: bundled Python `-m py_compile backend/services/air_network.py tests/test_air_network.py`
  - Not run: full pytest suite. Local `python` and `py` are unavailable in PATH; bundled Python does not have pytest; project `.venv` Python runs but does not have pytest installed.

Known limitations:
  - External provider reachability and real aircraft display must be validated on the physical Mini Tracker after deployment with Internet access.
  - Network ADS-B provider rate limits and real response variability are not validated by local mocked tests.
  - Existing `tracker-mini.zip` is locally modified by Raffaello and was not updated by this task.
```

```text
Task: MT-TRAFFIC-01 Local Hardening Pass
Feature: MT-TRAFFIC-01
Owner: Kiro
Working branch: main
Starting commit: 0cb7af0
Latest commit: Pending
Push status: Pending
Specification: .kiro/specs/traffic-proximity-awareness/
Status: Local hardening complete, 106 tests passing, ready for physical validation
Started: 2026-08-04
Last updated: 2026-08-04

Files created:
  tests/test_proximity_flask.py (17 Flask integration tests)

Files modified:
  frontend/js/dashboard.js (ADSBNet localStorage→backend migration)
  frontend/help/docs/user/traffic-monitoring.md (Traffic Proximity Awareness section)
  frontend/help/docs/developer/services.md (proximity engine in service tables)
  frontend/help/docs/developer/api.md (proximity API documentation)
  tests/test_proximity_engine.py (performance threshold correction)
  .kiro/steering/lessons-learned.md (performance test pattern)

Application integration: Complete
  - proximity_bp registered
  - proximity_engine started in app.py
  - All 3 proximity API routes reachable (verified by Flask integration tests)

Frontend: Complete
  - proximity-controller.js polls /api/proximity/status every 5s
  - proximity-layer.js renders distance line + rings
  - proximity-panel.js shows Nearby Traffic panel
  - ADSBNet migration logic in dashboard.js

Tests: 106 total (89 unit + 17 Flask integration), all passing
  - Command: python -m pytest tests -v
  - Duration: 0.85s
  - Failed: 0, Skipped: 0

Documentation: Updated
  - user/traffic-monitoring.md: Traffic Proximity Awareness section added
  - developer/services.md: proximity modules in service table
  - developer/api.md: proximity API endpoints documented
  - MkDocs build: NOT executed (mkdocs not installed; source updated, build pending)

Performance statement (corrected):
  - Windows development machine: test cycle ~132ms (variable, mock overhead)
  - Raspberry Pi: UNKNOWN until physical validation
  - Acceptance threshold: must be evaluated on physical device
  - No performance requirement weakened without measured RPi evidence

ADSBNet preference migration:
  - Implemented in dashboard.js (migrateAdsbNetPreference function)
  - Reads localStorage once on load, POSTs to backend, removes localStorage key
  - Does not silently re-enable for users who disabled it
  - Backend setting becomes authoritative after migration

Runtime package verification:
  - All runtime files under backend/ and frontend/
  - No dependency on tests/, .kiro/, pytest.ini, or config/settings.json distribution
  - Feature works after deploying only backend/ and frontend/

Physical validation: PENDING (requires Raffaello approval)
Rollback reference: 0cb7af0
Next action: Physical Mini Tracker validation
```

---

## Completed Work

No summer development feature has been completed yet.

---

## Update Requirements

Kiro must update this file after:

* repository onboarding;
* physical tracker inspection;
* creation of a Feature Spec;
* approval of a technical design;
* completion of an implementation task group;
* deployment to staging;
* physical hardware testing;
* discovery of an architectural inconsistency;
* modification of a shared contract;
* completion or suspension of a feature.

Updates must record facts and test results.

Do not use this file for speculative design details that belong in a Feature Spec.
