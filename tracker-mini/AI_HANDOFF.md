# Mini Tracker AI Development Status

Last updated: 2026-08-06

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
* inspect runtime state when explicitly authorized and required.

Raffaello remains responsible for:

* approving requirements and design decisions;
* approving potentially destructive system operations;
* pulling pushed repository changes;
* installing the Mini Tracker software package through the System Update functionality;
* testing the installed package manually on the physical Mini Tracker;
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

### Installation and validation boundary

Kiro must not attempt to install the Mini Tracker software on the physical device.

The software is installed and tested manually by Raffaello using the package installation flow exposed through the Mini Tracker System Update functionality.

Kiro's delivery responsibility ends at pushing completed, locally checked work to the GitHub repository and recording clear validation notes in `AI_HANDOFF.md`.

After Kiro pushes to GitHub, Raffaello pulls the repository changes, creates or uses the appropriate installation package, installs it through System Update, and performs the physical Mini Tracker validation.

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

Application-level checks after development can only be validated by Raffaello after installing the tested package on the Raspberry Pi Mini Tracker through System Update.

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

Kiro must not deploy code or install packages into the stable or staging physical Mini Tracker installation. If staging validation is needed, Raffaello performs the pull, package installation through System Update, and physical validation.

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
* Kiro must not attempt to install the software on the physical Mini Tracker.
* Raffaello performs manual package installation and testing through the Mini Tracker System Update functionality.
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
* Post-development application controls can only be proven by Raffaello after package installation through System Update on the Raspberry Pi Mini Tracker.
* Application logs are in-memory only (lost on restart).
* Flash storage is the single point of persistence (power loss risk).

---

## Active Work Record

```text
Task: Installation Responsibility Clarification
Feature: Repository workflow and physical validation boundary
Owner: Codex
Working branch: main
Starting commit: Pending due local repository safe-directory restriction
Latest commit: Pending
Push status: Pending
Status: Handoff guidance updated
Started: 2026-08-06
Last updated: 2026-08-06

Files modified:
  AI_HANDOFF.md

Implementation:
  - Clarified that Kiro must not attempt to install the Mini Tracker software on the physical device.
  - Documented that Raffaello pulls pushed repository changes, installs the package through System Update, and manually tests the installed package.
  - Updated repository workflow, ownership, validation boundary, rollback/staging guidance, architectural decisions and known constraints to use the manual System Update installation workflow.

Tests/checks:
  - Not run: documentation-only handoff update.

Known limitations:
  - Git status could not be verified locally because the parent repository is blocked by Git safe-directory ownership protection for the sandbox user.
```

```text
Task: Remote ID and Meshtastic Marker Lifecycle Settings
Feature: Traffic and team marker freshness
Owner: Codex
Working branch: main
Starting commit: Pending due local repository safe-directory restriction
Latest commit: Pending
Push status: Pending
Status: Local implementation complete, physical Mini Tracker validation pending
Started: 2026-08-06
Last updated: 2026-08-06

Observed issue:
  - Remote ID drone markers disappeared too quickly for DJI drones that transmit less frequently than Dronetag devices.
  - Meshtastic operator markers did not use the same stale/fade/removal lifecycle as Remote ID drone markers.
  - Meshtastic operator lastSeen was being updated by team refresh/binding rather than reflecting radio last seen timing.

Files created:
  tests/test_meshtastic_operator_freshness.py

Files modified:
  config/settings.json
  backend/services/ds110.py
  backend/services/meshtastic_service.py
  backend/services/teams.py
  frontend/js/drones/drone-layer.js
  frontend/js/meshtastic/meshtastic-controller.js
  frontend/js/meshtastic/meshtastic-layer.js
  frontend/js/missions/mission-teams.js
  tests/test_remoteid_stale.py
  frontend/help/docs/developer/api.md
  frontend/help/docs/hardware/remote-id.md
  frontend/help/docs/hardware/meshtastic.md
  frontend/help/docs/user/traffic-monitoring.md
  frontend/help/docs/user/teams.md
  frontend/help/docs/user/settings.md
  AI_HANDOFF.md

Implementation:
  - Added Remote ID marker lifecycle settings under SETTINGS["remoteid"]: marker_stale_ms=45000 and marker_retention_ms=180000.
  - DS110 Remote ID API freshness metadata now includes stale_ms and retention_ms for each returned drone.
  - Remote ID frontend marker fade/removal now uses API-provided stale/retention values with conservative defaults matching settings.json.
  - Added Meshtastic operator lifecycle settings under SETTINGS["meshtastic"]: operator_stale_ms=600000 and operator_retention_ms=1800000.
  - Meshtastic node last_seen now derives from Meshtastic lastHeard when available, rather than being refreshed every polling cycle.
  - /api/teams now annotates operators with updatedAt, age_ms, stale, expired, stale_ms and retention_ms, and includes operator_freshness settings.
  - Meshtastic operator markers now fade to grayscale after stale_ms and disappear after retention_ms, using the same style pattern as Remote ID drone markers.
  - Teams panel now displays radio last_seen when available.

Tests/checks:
  - Passed: bundled Python -m json.tool config/settings.json.
  - Passed: bundled Python -m py_compile backend/services/ds110.py backend/services/meshtastic_service.py backend/services/teams.py tests/test_remoteid_stale.py tests/test_meshtastic_operator_freshness.py.
  - Passed: node --check frontend/js/drones/drone-layer.js.
  - Passed: node --check frontend/js/meshtastic/meshtastic-layer.js.
  - Passed: node --check frontend/js/meshtastic/meshtastic-controller.js.
  - Passed: node --check frontend/js/missions/mission-teams.js.
  - Passed: frontend/help/docs image reference check.
  - Passed: mkdocs build --strict --config-file frontend/help/mkdocs.yml using the project .venv after sandbox escalation.
  - Passed: git diff --check -- .
  - Not run: pytest tests/test_remoteid_stale.py tests/test_meshtastic_operator_freshness.py. Bundled Python does not have pytest installed; project .venv pytest remains unavailable in the sandbox due access denied.

Known limitations:
  - Local checks do not validate real DJI Remote ID transmission intervals or real Meshtastic radio timing.
  - Raffaello must validate on the Mini Tracker hardware: DJI Remote ID marker fade/removal timing, Dronetag marker behavior, Meshtastic stationary-operator marker fade/removal, and operator last seen display.
```

```text
Task: User Manual Image Integration and Missing Page Completion
Feature: Mini Tracker documentation
Owner: Codex
Working branch: main
Starting commit: Pending due local repository safe-directory restriction
Latest commit: Pending
Push status: Pending
Status: Local documentation update complete
Started: 2026-08-06
Last updated: 2026-08-06

Observed issue:
  - Several newly added screenshots under frontend/help/docs/images/ were not referenced by the manual.
  - MkDocs navigation referenced user/settings.md, user/troubleshooting.md, user/faq.md and glossary.md, but those source files did not exist.

Files created:
  frontend/help/docs/user/settings.md
  frontend/help/docs/user/troubleshooting.md
  frontend/help/docs/user/faq.md
  frontend/help/docs/glossary.md

Files modified:
  frontend/help/docs/index.md
  frontend/help/docs/user/teams.md
  frontend/help/docs/user/traffic-monitoring.md
  frontend/help/docs/user/mission-planning.md
  AI_HANDOFF.md

Implementation:
  - Added Teams screenshots for gateway status, mission operators, external nodes, operator map marker, direct message dialog and sent/received Messages section.
  - Added Traffic Monitoring screenshots for ADS-B traffic and Traffic Proximity Awareness MON/CAUTION examples.
  - Added Mission Planning screenshots for Drone Sky Check import action, import panel and imported zone display.
  - Added a new Settings user-guide page covering system status, traffic source controls, hardware status, DS110 settings, network settings and system update workflow.
  - Added operator-level Troubleshooting, FAQ and Glossary pages to satisfy existing MkDocs navigation entries.
  - Updated documentation status entries for settings, troubleshooting, FAQ and glossary.

Tests/checks:
  - Passed: local image reference check across frontend/help/docs Markdown files.
  - Passed: mkdocs build --strict --config-file frontend/help/mkdocs.yml using the project .venv after sandbox escalation. Build succeeded in 0.88s.
  - Passed: git diff --check -- frontend/help/docs.
  - Cleaned generated frontend/help/site changes after the build so only documentation sources remain modified.

Known limitations:
  - Documentation remains in English per DOCUMENTATION.md. Italian manual generation requires an approved documentation policy and structure change.
  - Screenshot filenames added by the user were left unchanged, including Italian names and existing typos, to avoid moving user-managed files.
```

```text
Task: Meshtastic Message Direction and Incoming Text Fix
Feature: Meshtastic operational control
Owner: Codex
Working branch: main
Starting commit: Pending due local repository safe-directory restriction
Latest commit: Pending
Push status: Pending
Status: Local implementation complete, Raspberry deployment validation pending
Started: 2026-08-05
Last updated: 2026-08-05

Observed issue:
  - Direct Message from one operator card did not reach the configured operator, while Send message to all reached the only configured operator.
  - Gateway-sent messages appeared in the Mission Teams Messages list as `tracker` plus text only, without clear source, destination or status.
  - Meshtastic text messages sent by an operator to the Mini Tracker gateway were logged as packets but were not shown in the Messages list.

Files created:
  tests/test_meshtastic_messages.py
  tests/test_notification_service.py

Files modified:
  backend/services/meshtastic_service.py
  backend/services/notification_service.py
  frontend/js/missions/mission-teams.js
  frontend/help/docs/user/teams.md
  frontend/help/docs/hardware/meshtastic.md
  frontend/help/docs/developer/api.md
  AI_HANDOFF.md

Implementation:
  - Incoming Meshtastic TEXT_MESSAGE_APP packets are now recorded through the Notification Service when they are not sent by the local gateway.
  - Notification records now include direction, source node ID, target label and transport metadata while preserving existing basic fields.
  - Outgoing operator messages now identify the source as Gateway and include the operator label when available.
  - Send message to all now sends only to online operators with a Meshtastic nodeId, avoiding fallback to the mission operator numeric id.
  - The Mission Teams Messages list now displays source -> destination, status, timestamp and message text.
  - The single-operator Message button refreshes live team status before selecting the target nodeId.
  - External node removal now uses the backend-provided nodeId instead of a missing node.id field.

Tests/checks:
  - Passed: bundled Python -m py_compile backend/services/meshtastic_service.py backend/services/notification_service.py tests/test_meshtastic_messages.py tests/test_notification_service.py.
  - Passed: node --check frontend/js/missions/mission-teams.js.
  - Passed: git diff --check -- .
  - Not run: pytest tests/test_meshtastic_messages.py tests/test_notification_service.py tests/test_meshtastic_routes.py. System python is unavailable, bundled Python does not have pytest installed, and .venv/Scripts/python.exe fails with access denied in the sandbox.

Known limitations:
  - Local tests use mocked Meshtastic packet/interface behavior and do not prove delivery over the real radio.
  - Raffaello must validate on the Raspberry Pi Mini Tracker with the connected T-Beam: direct operator message, send-to-all, operator-to-gateway inbound text, and message list labeling.
```

```text
Task: Meshtastic Enable Persistence Fix
Feature: Meshtastic operational control
Owner: Codex
Working branch: main
Starting commit: Pending due local repository safe-directory restriction
Latest commit: Pending
Push status: Pending
Status: Local implementation complete, Raspberry deployment validation pending
Started: 2026-08-05
Last updated: 2026-08-05

Observed issue:
  - After adding the proximity section to config/settings.json, Meshtastic appeared disabled at app.py startup.
  - Enabling Meshtastic from the Dashboard checkbox did not start the connection to the local T-Beam.
  - config/settings.json was valid JSON and the meshtastic section was still readable.
  - Root cause found in the enable flow: /api/meshtastic/enable called meshtastic_service.start(), but start() refused to run while SETTINGS["traffic"]["meshtastic_enabled"] remained false.

Files created:
  tests/test_meshtastic_routes.py

Files modified:
  backend/routes/meshtastic.py
  frontend/js/dashboard.js
  frontend/help/docs/developer/api.md
  AI_HANDOFF.md

Implementation:
  - /api/meshtastic/enable now persists SETTINGS["traffic"]["meshtastic_enabled"] before starting or stopping the Meshtastic worker.
  - /api/meshtastic/status now returns both configured persistent state and current worker running state.
  - Dashboard Meshtastic checkbox now starts or stops the frontend Meshtastic polling layer immediately after the backend enable request.
  - Developer API documentation now describes the persistent Meshtastic enable behavior.
  - AI_HANDOFF.md now explicitly states that post-development application-level controls can only be validated by Raffaello after deployment on the Raspberry Pi Mini Tracker.

Tests/checks:
  - Passed: PowerShell ConvertFrom-Json validation for config/settings.json.
  - Passed: bundled Python -m json.tool config/settings.json.
  - Passed: bundled Python -m py_compile backend/routes/meshtastic.py tests/test_meshtastic_routes.py.
  - Passed: project .venv Python -m py_compile backend/routes/meshtastic.py tests/test_meshtastic_routes.py.
  - Passed: node --check frontend/js/dashboard.js.
  - Not run: tests/test_meshtastic_routes.py under pytest. The project .venv starts but does not have pytest installed, and the bundled Python does not have pytest or Flask installed.

Known limitations:
  - This local validation does not prove Meshtastic serial hardware operation.
  - Raffaello must deploy to the Raspberry Pi Mini Tracker and test the Dashboard checkbox against the local T-Beam to validate the application behavior.
  - If the T-Beam path differs on the deployed device, backend logs should show the configured device path used by meshtastic_service.
```

```text
Task: Remote ID Stale Marker Lifecycle and Popup Details
Feature: Remote ID dashboard usability
Owner: Codex
Working branch: main
Starting commit: 0c59f29
Latest commit: Pending
Push status: Pending
Status: Local implementation complete, physical Mini Tracker validation pending
Started: 2026-08-05
Last updated: 2026-08-05

Observed issue:
  - Remote ID drone markers could remain on the Dashboard map for minutes after packets stopped.
  - The DS110 API returned all in-memory Remote ID aircraft without a map-facing freshness lifecycle.
  - The drone marker popup showed only model, vendor, serial and source.

Files created:
  tests/test_remoteid_stale.py

Files modified:
  config/settings.json
  backend/services/ds110.py
  frontend/js/drones/drone-layer.js
  frontend/help/docs/hardware/remote-id.md
  frontend/help/docs/user/traffic-monitoring.md
  frontend/help/docs/developer/api.md
  frontend/help/docs/developer/frontend.md
  frontend/help/docs/developer/services.md
  AI_HANDOFF.md

Implementation:
  - Fixed `config/settings.json` JSON syntax by restoring the missing comma between `proximity` and `meshtastic`.
  - Remote ID API responses now include computed freshness metadata: `updatedAt`, `age_ms` and `stale`.
  - Remote ID tracks are considered stale using the existing proximity `drone_stale_ms` setting, defaulting to 15 seconds.
  - Expired Remote ID tracks are removed from the DS110 in-memory cache after the stale threshold plus the retention grace window, defaulting to about 75 seconds total.
  - Dashboard drone markers now fade and turn grayscale while stale, then disappear after the retention window.
  - Drone popup details now include altitude, height, speed, heading and last packet age when available.

Tests/checks:
  - Passed: PowerShell `ConvertFrom-Json` validation for `config/settings.json`
  - Passed: bundled Python `-m json.tool config/settings.json`
  - Passed: `node --check frontend/js/drones/drone-layer.js`
  - Passed: `node --check frontend/js/drones/drone-controller.js`
  - Passed: `node --check frontend/js/drones/drone-network.js`
  - Passed: bundled Python `-m py_compile backend/services/ds110.py tests/test_remoteid_stale.py`
  - Passed: direct bundled-Python Remote ID freshness assertions for fresh, stale and expired tracks.
  - Not run: pytest. System `python` is unavailable in PATH, bundled Python does not have pytest installed, and `.venv/Scripts/python.exe` failed with access denied in the sandbox.

Known limitations:
  - Raspberry Pi updater `test_import` must be rerun after deploying this corrected package.
  - Remote ID stale/fade timing must be visually validated in the deployed Mini Tracker browser with a real DS110 source.
  - Development-machine checks do not validate DS110 hardware reception.
```

```text
Task: ADS-B Popup Source Label Cleanup
Feature: ADS-B dashboard popup usability
Owner: Codex
Working branch: main
Starting commit: 0c59f29
Latest commit: Pending
Push status: Pending
Status: Local UI cleanup complete, physical Mini Tracker validation pending
Started: 2026-08-05
Last updated: 2026-08-05

Files modified:
  frontend/js/air/air-layer.js

Implementation:
  - Added frontend source-label normalization for ADS-B aircraft popups.
  - Network ADS-B provider source combinations such as `AIRPLANES_LIVE+ADSB_LOL+OGN_ADSB+OPENSKY` now display as `Internet`.
  - Local ADS-B displays as `RTL-SDR`.
  - Mixed local/network provenance displays as `RTL-SDR + Internet`.
  - Backend `source` values remain unchanged for diagnostics and merge/proximity logic.

Tests/checks:
  - Passed: `node --check frontend/js/air/air-layer.js`

Known limitations:
  - Visual popup result must be validated in the deployed Mini Tracker browser.
```

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
