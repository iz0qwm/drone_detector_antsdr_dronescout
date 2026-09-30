# Mini Tracker AI Development Status

Last updated: 2026-09-23

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
* compiling documentation with MkDocs when documentation output must be regenerated;
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

Kiro and Codex must not compile the documentation with MkDocs as part of normal development or validation.

Raffaello manually runs MkDocs and regenerates documentation output when needed.

Do not request sandbox escalation only to run a MkDocs build. Documentation validation by AI agents should be limited to source inspection, link/image reference checks when useful, and `git diff --check`.

Documentation must:

* keep English as the canonical/default language;
* allow Italian localized documentation when explicitly requested by Raffaello or when implementing the multilingual manual;
* follow `DOCUMENTATION.md` for mkdocs-static-i18n structure and language rules;
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
* Kiro and Codex must not compile documentation with MkDocs; Raffaello runs MkDocs manually when needed.
* English remains the canonical documentation language, but Italian localized pages are allowed for the multilingual manual using `mkdocs-static-i18n` guidance in `DOCUMENTATION.md`.
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

### Field Operations final UI polish / rehearsal — 23 September 2026

Status: **FIELD OPERATIONS UI POLISH — IMPLEMENTED**. Explicit Product Owner request supersedes MF6-C's Esci dal replay control. Spec: `.kiro/specs/maker-faire-demo-cut/field-operations-ui-polish.md`. Report: `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_FIELD_OPERATIONS_UI_POLISH_20260923.md`.

- Removed explicit Replay exits in 2D/Google and duplicate body Chiudi. Header X is the session close; accepted/pending Replay locks mode switches in UI and controller. Google Field Operations X delegates to the same owner lifecycle. Chiudi 3D/Escape remain view-only and leave 2D playback active.
- Close disposes Replay driver, timers and generation, clears scenes/interpolation/imports, closes Google, and waits for explicit reopen. A previously exported recorder buffer is released so cleared export flags cannot resurrect it as unsaved. No disk file deletion; unsaved-entry gates unchanged.
- Polished segmented selector, persistent mode/paused/completed/recording status, Italian DEMO state copy, compact awareness summary, hierarchy, duplicate counters/source labels, model-status copy, dark/light buttons and responsive bounds. Labels OFF and all domain/timing/model behavior preserved.
- Modified DSC production files: field-node-card.js, field-map.js, record-replay-controls.js, scene-motion.js (presentation only), replay-source.js (error copy only), mission3d-bridge.js, field-operations.css, mission3d.js/html/css, mission3d-model.js and mission3d-renderer.js (copy only). Seven test suites and three test-host files updated; exact 22-file inventory in report. Tracker files: this additive handoff entry and the new polish spec only; pre-existing changes preserved.
- Verification: **530 tests passed**, zero failed/cancelled/skipped/todo; 517 retained plus 13 new. After final copy adjustment, **37 affected tests passed** again. **18 JavaScript syntax checks** and scoped whitespace checks passed; final lifecycle/control edits rechecked.
- Actual Chromium 153 rehearsal: LIVE record/stop/export initiation, complete DEMO, all 62 canonical Replay frames with 2D/Google terminal ACKs, pause/resume, restart/loop, 2D/Google X closure, and clean reopened mode selection. Native GLB scale 4/6 and terminal 35 m presentation preserved. Dark/light at 390/768/1280, 2D drag, compact and keyboard controls reviewed; no observed horizontal overflow or page errors.
- Limitations: synthetic source/access host, no authenticated deployed-account/hardware claim. Native filechooser automation timed out; canonical test-host button passed the unchanged fixture through the real worker/importer. Native picker and full-app/device acceptance remain manual checks. Existing Google LIVE altitude limitations unchanged.

No deployment, commit, push, Mini Tracker runtime, dependencies, transport/services, offline package or MkDocs changes.

### MF-DEMO-6C Replay and mode separation — 23 September 2026

Status: **MF-DEMO-6C — REPLAY + MODE SEPARATION IMPLEMENTED**. Explicit PO authority and MF6 Design Lock revision 2; LIVE-only product recording supersedes earlier control presentation. Spec: `.kiro/specs/maker-faire-demo-cut/mf-demo-6c.md`. Report: `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_MF_DEMO_6C_REPLAY_IMPLEMENTATION_20260923.md`.

DSC adds replay-source.js and two test suites. Twenty-one narrow updates cover parent/map/controls/shared presentation/bridge, Google child/model/renderer, entry HTML/CSS, existing tests and isolated browser rehearsal host. The report gives the full 24-file inventory. Mini Tracker changed only this handoff and the bounded spec; prior local changes were preserved. Domain core/profile/geometry, MF6 format/import/recorder modules, fixture and model assets are unchanged.

Parent explicitly owns LIVE/DEMO/REPLAY and gates both UI and handlers. LIVE records; DEMO operates the unchanged rescue scenario; REPLAY imports validated frozen snapshots and plays at 1x. No autoplay on import, nested recordings, current-UTC aging or domain recalculation. Active/unsaved recording prevents source switching until Stop and Save/Discard. Monotonic driver preserves sequence/equal-time frames/static tail, bounds work at 100 frames/8 ms, pauses on hidden tab/backlog, invalidates stale callbacks and resets presentation generations. Bridge v2 retains exact origin/window/channel guards; healthy consumers settle/acknowledge terminal sequence within 2 s before loop. Closed/loading/failed Google cannot block 2D. Exit clears replay/views and restores selection only; explicit reopen starts fresh source. Badges/provenance remain visible in compact panels.

Verification: **517 passed**, zero failures/cancellations/skips (174 focused MF6 including 36 additions, 343 retained regressions); 19 JavaScript syntax checks and scoped whitespace/scope checks passed. Instrumented replay invokes source/motion/core ingestion/evaluation zero times. Canonical tests consume all 62 frames, preserve narrative/awareness and versioned 600-to-35 m descent, stable model identities, camera and scales 4/6. Tests cover malformed/stale bridge messages, old UTC freshness, backlog and terminal-ACK failure.

Actual Chromium 153 browser workflow recorded/exported a new LIVE synthetic-source file (187 frames, 62248 ms, 1030789 bytes), reimported with native picker and played all frames in 2D. Native Google canonical run delivered 62 ordered frames; both 2D and Google acknowledged terminal seq 61. Pause/resume, restart/loop, Google close/reopen, compact badges/390 px map container, exit/fresh LIVE reopen and LIVE recording controls while Google is open were reviewed. Final Google run: two native models scales 4/6, terminal altitude 35 m, no page errors. Canonical import 55 ms; parent frame delivery p95 1.8 ms/max 3.3 ms; one observed 65 ms task around Google initialization. No GPU/FPS/heap claim. Temporary tab/server closed.

Ready for broader **MF6-D online acceptance**, with PO acceptance pending. Browser evidence uses production modules and actual Google on the existing authorized localhost origin, with synthetic source/access callbacks; it is not deployed-account or Mini Tracker/RF hardware acceptance. Existing Google LIVE altitude limitation remains. Final polish candidates are recorded for after C/D acceptance; no premature polish automation. No deploy, commit, push, Mini Tracker runtime edits, dependencies, offline package or MkDocs build.

### MF-DEMO-6B JSON export/import and validation worker — 23 September 2026

Status: **MF-DEMO-6B — JSON EXPORT / IMPORT IMPLEMENTED**. Explicit PO authorization to consume approved MF6-A/Design Lock without playback or offline implementation. Spec: `.kiro/specs/maker-faire-demo-cut/mf-demo-6b.md`. Report: `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_MF_DEMO_6B_EXPORT_IMPORT_IMPLEMENTATION_20260923.md`.

DSC additions: replay-import-validator.js, replay-import-worker.js, recording-files.js, record-replay-controls.js; two new test suites, worker/fixture/browser-server helpers, isolated browser rehearsal host and recorder-generated rescue-arrival-v1.json. Narrow updates: scene-recorder.js (stopped serialized export parts), field-node-card.js (file owner/privacy lifecycle), field-map.js (controls/compact indicator), field-operations.css, index.html and recorder test/helper. The report lists all 12 new and 7 modified DSC files. Mini Tracker changed only this handoff and the bounded spec; prior local changes were preserved.

Export uses retained chunks, JSON UTF-8 Blob, metadata-only UTC/UUID filename, bounded Object URL cleanup and retryable buffer. Import size gate precedes File read; shared lexical/schema validator runs in a local terminable worker with 15 s deadline. Duplicate/prototype/credential keys, invalid UTF-8, token/depth/string/byte violations and cross-frame inconsistencies reject atomically. Only supported normalized fields reach a deeply frozen session. Large sanitized transfers are batched after whole-file validation. No core calculation, remote recording fetch, persistence or source replacement. Unsaved recorder buffers require Save/Discard; ordinary close retains data, authorization/private-context loss clears buffer/session/worker.

Verification: **481 passed**, zero failures/cancellations/skips: 138 focused (33 format + 24 recorder + 67 import + 14 controls) and 343 requested regressions; 82 additions relative to MF6-A. Syntax checks passed on 15 JavaScript files; scoped whitespace/scope checks passed. Canonical real-recorder fixture: 62 frames, 60000 ms, 334415 bytes, COMPLETE/SUSPENDED/WARNING/INSIDE/0 m, full semantic round-trip equality through actual worker code.

Actual Chromium 153 browser run: 60 s scenario recorded/exported through controls; disk JSON verified (63 frames, 77240 ms including prelude/static tail, 339518 bytes), native-picker reimport accepted without leaving DEMO. Canonical fixture and 16.5 MB valid sample also imported; six hostile samples rejected preserving source/session/buffer. Compact indicator and 390 px layout verified, zero page errors/warnings. Final canonical import 127.2 ms total / 66.8 ms worker; 16504483-byte valid input 1659 ms total / 1309.6 ms worker with zero observed >=50 ms main-thread tasks after batch transfer. Exact 32 MiB hostile token input rejected by Node worker in 444.2631 ms. Environment: Windows x64, Ryzen 5 8645HS, Node v24.11.1. No browser heap claim.

Ready for **MF6-C Replay source + Play/Pause/Restart/Loop** via immutable validatedRecording. Known limits: browser evidence is isolated synthetic-source host using real production modules, not authenticated deployed-account/Google visual or hardware acceptance; full maximum-shape heap behavior unmeasured. Browser event automation timed out for download notification, but actual disk file was independently verified. Test server/tab closed after verification. No deploy, commit, push, Mini Tracker runtime edits, dependency additions, replay playback, offline package or MkDocs.

### MF-DEMO-6A recording format and recorder core — 23 September 2026

Status: **MF-DEMO-6A — RECORDER CORE IMPLEMENTED**. Product Owner explicitly requested DSC implementation of the approved MF-DEMO-6 Design Lock revision 2. Bounded spec: `.kiro/specs/maker-faire-demo-cut/mf-demo-6a.md`. Report: `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_MF_DEMO_6A_RECORDER_IMPLEMENTATION_20260923.md`.

New DSC modules: awareness-schema.js (existing strict Google validator, calculation-free), replay-format.js (V1 allowlist/limits/metadata/equality) and scene-recorder.js (detached full snapshots, monotonic timeline, bounded atomic capture). Narrow edits: scene-store.js, scene-motion.js, operational-source.js, field-node-card.js, public/index.html and mission3d.js/html. New tests: replay-format.test.js, scene-recorder.test.js, helpers/recording-fixtures.js; existing mission3d.test.js and geoawareness-controller.test.js loaders updated. Full paths and responsibilities are in the report. Only this handoff and the bounded spec changed in Mini Tracker; pre-existing local changes were preserved.

Parent now commits after motion/awareness/logical freshness; recorder errors are isolated from store/2D/3D/core/motion. Existing freshness thresholds and DEMO holds are shared unchanged. Scenario version RESCUE_ARRIVAL_DEMO_20260922_V1 is stable through all controls. Normal close retains SOURCE_CLOSED recording in authorized page memory; revocation/logout/disposal clear it. Hidden developer API only; no network/persistence. Counts, UTF-8 budgets and empty-only team are enforced before append; failures retain valid prefix. Source timestamps/revisions and equal-time terminal/restart ordering are preserved.

Verification: 56 focused + 343 required regression tests = 399 passed, zero failures/cancellations/skips. Syntax: 13 JavaScript files passed. Scoped whitespace/hash checks passed; core/profile/geometry, 2D map, Google model/renderer and model scales 4/6 unchanged. Real 60 s rescue capture: 62 frames, 60000 ms, 334415 JSON UTF-8 bytes, COMPLETE/SUSPENDED and WARNING/INSIDE/CURRENT/0 m. Existing terminal callback and subsequent publish differ in observation timestamps, so both are retained at t=60000.

Performance (Node v24.11.1, Windows x64, AMD Ryzen 5 8645HS): p50 0.3883 ms, p95 0.7939 ms, max 3.3525 ms; largest frame 5411 bytes; no measured call >=50 ms. Ten-minute ordinary 601-frame size projection: 3236861 bytes. These are Node measurements, not browser main-thread/memory or hardware evidence. Browser Long Tasks and fresh visual/hardware acceptance remain unperformed; MEMORY_PRESSURE is explicitly signaled, not automatically detected.

Ready for **MF6-B JSON export/import + validation worker**. Full untrusted-file validation, export/error UI, replay source/controls, offline package and browser acceptance remain future work. No deploy, commit, push, Mini Tracker runtime change, dependencies or MkDocs build.

### MF-DEMO-5C-4B DEMO helicopter descent refinement — 22 September 2026

PO visually accepted rescue story, authorized gradual helicopter ground approach. Google adapter-local height now falls 600 -> 35 m relative to ground during 55–59 s, only for designated DEMO rescue target after a CURRENT area INSIDE/BOUNDARY 0 m relation; last second settles before COMPLETE. Source 2400 ft MSL, target samples, route and horizontal awareness unchanged. Same AGL presentation height as drone; no touchdown/real control. Existing interpolator smoothly carries only the adapter-local field to marker/model; pause and restart covered. No 2D behavior change.

Three DSC files: mission3d-model.js, scene-motion.js, mission3d.test.js. Existing spec/report updated with superseding descent refinement. 115 focused and 370 regression tests passed, zero failures/skips; syntax and scoped whitespace/hash checks passed. Actual native Google run [1,6] completed: 600 m at 33 s, ~183.6 m at 58 s, both targets 35 m at 60 s; two models scales 4/6; no captured errors/warnings. No deploy/commit/push/Mini Tracker runtime changes. Existing 5C-4B report remains authoritative with its appended refinement; MF6 recording inputs unaffected.

### MF-DEMO-5C-4B rescue helicopter operational story — 22 September 2026

Status: IMPLEMENTED, ready for PO visual review. PO accepts the prior models, animation and helicopter scale 6; this supersedes the outstanding scale-review qualification below. Spec: `.kiro/specs/maker-faire-demo-cut/mf-demo-5c-4b.md`. Addendum: `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_GEOAWARENESS_5C4B_RESCUE_STORY_20260922.md`.

Replaced synthetic fly-through with 60 s rescue arrival. RID patrol within canonical 200 m circle; operator decision at 25 s, return to explicit DEMO waiting point at 30 s, stationary/suspended outside circle at 38 s. No touchdown, no automatic command; 35 m AGL unchanged. Elicottero 01 retains identity/class and source altitude, approaches from 3600 m east to center. Rescue role exists only in explicit DEMO narrative. Area MONITOR 5 s, CAUTION 25 s, WARNING 50 s, INSIDE/0 m 55–60 s; independent UAS WARNING only at 55 s. All computed by unchanged core. Final trend APPROACHING, not forced DIVERGING. Arbitrary LIVE geometry remains truthful and has no guaranteed canonical timeline.

Nine DSC files: public/js/field-operations/{scene-motion,field-map}.js; public/css/field-operations.css; public/mission3d/{mission3d.js,mission3d.html,mission3d.css}; functions/test/{scene-motion,mission3d,field-operations-map}.test.js. Shared presentation-only motion.scenario is included in terminal/normal samples; both views use identical narrative text above collapsed/scrolled bodies. Google panel body scrolls independently to retain the operator story. No core/profile/geometry/controller/bridge/renderer/model/GLB, transport or Mini Tracker runtime changes; scoped pre-task hashes confirmed. Prior unrelated local work preserved.

Tests: 182 focused across four suites, 368 required 13-suite regression, all passed, zero failures/skips; five new tests plus updated scenario oracles. Six JS syntax and nine-file whitespace checks passed. Independent haversine patrol/arrival checks, natural WARNING/0 m, UAS hold, phase isolation, pause/resume/restart/loop reset, same bridge/Google story, compact 2D and stable native model identities/scales/camera verified.

Actual browser: uninterrupted complete 2D [1,2] and native Google [1,3] runs, all 13 checkpoints each matched oracle. Interaction [1,4] tested 2D drag/compact/zoom/pause, 390 px no horizontal overflow, resumed same phase in 3D, orbit/zoom/compact/normal. Loop [1,4]->[1,5] preserved terminal INSIDE checkpoint then reset to ACTIVE/distant NORMAL/UNKNOWN; two native models at scales 4/6, labels OFF. Settled PAUSED model poses were identical across separated reads, resume verified. Google story remains visible while scrolling controls. No rendering stop/new renderer errors observed. One source-less MutationObserver error at 03:48:12 UTC on review page is disclosed; no cause assigned. Local DEMO only, no real auth/hardware/FPS guarantee.

MF-DEMO-6 can proceed after PO review: normalized target/core inputs preserved; additive motion.scenario is presentation metadata and no recorder was implemented. Future replay should identify this revised scenario version. No deploy, commit, push, MkDocs or dependency changes.

### MF-DEMO-5C-5 visual scale refinement — 22 September 2026

PO accepted model fluidity/animation. Helicopter presentation scale changed 3 -> 6 after evaluating 5; drone stays 4. Production edit only public/mission3d/mission3d-model.js; existing scale assertion updated in functions/test/mission3d.test.js. No other renderer, lifecycle, loading, heading, pose, altitude, awareness, camera, 2D, scene-motion, GLB or Mini Tracker runtime change. Existing Feature Spec and implementation report updated, no new slice.

Browser: scale-6 run [1,22] completed 60 s, pause/resume stable, terminal NORMAL/DIVERGING; restart [1,23] verified, two models with scales 4/6. Checked initial/Ricentra, moderate zoom out, closer view, orbit, labels OFF and panel readability. Scale 6 improves tail/fuselage visibility without covering the area; wide overview still favors markers, so unconditional silhouette recognition there is not claimed and needs PO review. No renderer stop/new rendering errors; existing source-less MutationObserver error disclosed. No FPS benchmark. Focused 81 tests passed (0 failed/skipped), model syntax and two-line before/after scope check passed. Earlier 363 suite result was not rerun for this constant-only refinement. No deploy/commit/push.

### MF-DEMO-5C-5 native Google target models — 22 September 2026

Status: MF-DEMO-5C-5 — GOOGLE 3D TARGET MODELS IMPLEMENTED. Current request visually accepts 5C-4, superseding its historical review-pending status below. Feature Spec: `.kiro/specs/maker-faire-demo-cut/mf-demo-5c-5.md`. Report: `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_GEOAWARENESS_5C5_GOOGLE_3D_MODELS_20260922.md`. Product Owner has accepted models and animation; helicopter scale refinement is recorded below.

Six DSC files: `public/mission3d/mission3d-model.js`, `mission3d-renderer.js`, `mission3d.js`, `mission3d.html`, `mission3d.css`, `functions/test/mission3d.test.js`. Optional native Model3DElement mapping: UAS -> existing drone.glb, ROTORCRAFT -> existing helicopter.glb; UNKNOWN/malformed classes and unevaluated fixed wing remain marker-only. Source assets verified (490912 bytes/8348 triangles; 3676360 bytes/313140 triangles; fixed wing inventory 578792 bytes/31068 triangles) and unchanged.

Original native visual calibration (helicopter scale superseded below): drone heading correction 0/tilt 90/roll 0/scale 4; helicopter heading +90/tilt 0/roll 0/scale 3. Both checked at cardinal headings and 359/1, with no geographic correction or source heading/altitude changes. Existing pose/interpolation retained (35 m AGL RID, helicopter +600 m presentation). Per-target stable model lifecycle, maximum two enhancements, cached availability check with 10 s timeout, no per-frame retry, class failure latch and late-work invalidation. Native Google has no documented load-success event: marker, labels and selection always remain as conservative fallback. No hidden timer-based marker removal. Session selector defaults Solo simboli, optional Drone/Elicottero/both. No persistent operational data. Model status preserves failure notice when another class succeeds.

Tests: 81 focused (60 existing + 21 new), 363 required 13-suite regression, all passed, 0 failed/0 skipped. Mapping/cardinal/wrap/malformed inputs; lifecycle and ten simulated loops; pause/resume/restart/removal/re-entry/class changes; failure cases and late completions; bounds; awareness/line/label/camera independence. Six-file whitespace and four JS syntax checks; scope baseline six changes among 169 DSC files. All 97 Mini Tracker runtime files unchanged; locked awareness, 2D, scene-motion and binary assets unchanged.

Actual browser stress completed: ten full 60 s cycles with both calibrated models, run IDs [1,8] through [1,17], all terminal checkpoints NORMAL/DIVERGING for both references. Two models/one map at observations, working orbit/zoom/Ricentra/selection/compact/labels, frozen poses during pause and resumed movement; clean close/reopen to current scene and models OFF. No new captured renderer errors during stress. Separate marker/drone/helicopter comparisons are detailed in the report. Local review uses production Google/Leaflet/core/motion with explicit DEMO access context, not authenticated private LIVE data. Native assets rendered and direction/up-axis were checked on a separate temporary same-origin calibration page. Original port 8767 attempt was blocked by existing referrer restriction; moved to approved 8766 without changing credentials/security. Default markers remain clearer in whole-route overview; models improve close views. The two simultaneous meshes remain optional, never a Maker Faire dependency.

Limitations: no FPS/memory/load-duration benchmark, dense traffic, hardware or external-browser validation. Native asynchronous model failure without a documented event is handled conservatively by never removing the marker. Fixed wing not visually evaluated/enabled; no optimization or binary edits. One source-less MutationObserver exception in cumulative logs predates the first model load. Normalized scene/recorder architecture unchanged, ready for MF-DEMO-6 after PO acceptance. No deploy, commit, push, Mini Tracker runtime change, Cesium, new dependency or MkDocs build.

Final per-asset evaluation: drone.glb ACCEPT and helicopter.glb ACCEPT as optional native enhancements; airplane_low.glb NOT EVALUATED and disabled. After the ten-loop combined stress, separate complete 60 s drone-only [1,19] and marker-only [1,20] runs reached the same terminal NORMAL/DIVERGING states. Default remains Solo simboli. Final report saved and copy hash verified at the requested DSC+ path.

### MF-DEMO-5C-4 helicopter geoawareness scenario — 22 September 2026

Status: MF-DEMO-5C-4 — HELICOPTER GEOAWARENESS SCENARIO IMPLEMENTED; ready for Product Owner review. The current request explicitly accepts 5C-3 visually, superseding its historical review-pending status below. Feature Spec: `.kiro/specs/maker-faire-demo-cut/mf-demo-5c-4.md`. Report: `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_GEOAWARENESS_5C4_HELICOPTER_SCENARIO_20260922.md`.

Seven DSC files: `public/js/field-operations/scene-motion.js`, `fixture-source.js`, `public/mission3d/mission3d-model.js`, and `functions/test/{scene-motion,mission3d,geoawareness-controller,field-operations-map}.test.js`. Stable `demo:aircraft-01`, display `Elicottero 01`, explicit ROTORCRAFT/EXPLICIT_DEMO; existing RID identity/patrol retained. Full 60 s continuous east-to-west flyby north of the operation; one authoritative sampler, including READY/final checkpoint. Explicit fixture circle radius 200 m; LIVE geometry and observations untouched. Only Google presentation adaptation: include motion bounds in initial/explicit-recenter framing, with no automatic camera movement on updates.

Real core oracle: area MONITOR at 5 s, UAS MONITOR at 10 s, both CAUTION at 25–40 s, MONITOR at 45–55 s and NORMAL at 60 s. Trends UNKNOWN at 0/5, APPROACHING at 10–30, UNKNOWN at 35/40, DIVERGING at 45–60. Actual independent distances locked to 0.05 m; full table in spec/report. No profile/core/geometry/controller/bridge/renderer semantic changes. A large LIVE footprint is explicitly tested to produce truthful WARNING/INSIDE instead of forcing the canonical story.

Tests: 28 scene-motion and 60 mission3d focused tests (88 total); 342 across the required 13 suites; all passed, 0 failed/0 skipped. Seven syntax and whitespace/conflict-marker checks passed. Hash comparison: exactly seven changed of 169 DSC baseline files; 97 Mini Tracker runtime files unchanged. Prior unrelated local modifications preserved.

Browser: full uninterrupted 60 s Leaflet and actual Google 3D runs in localhost:8766 explicit DEMO harness matched all 13 checkpoints; untouched cameras remained fixed. Verified labels OFF/ON/OFF, selection of both references, ground line/area outline, 2D SVG, Google terrain/native markers, compact panels, actual 2D panel drag/pan/zoom, Google orbit/zoom, pause with frozen positions and revision, resume, restart, two loop boundaries with NORMAL/UNKNOWN reset, close/reopen current run and stable element counts. No captured browser warnings/errors. Final 60000 callback ordering/no end-start interpolation additionally covered across three test loops. No visual acceptance claim on behalf of PO.

Limitations: browser uses local DEMO, not authenticated LIVE areas/hardware; mixed provenance and geometry are unit-tested. Arbitrary real areas have no guaranteed threshold timeline. Flyby is time-compressed, not aircraft performance simulation; native Google target markers remain until 5C-5. Browser smoothness is observational, not an FPS benchmark. Scenario is technically ready for 5C-5 after PO acceptance. No deploy/commit/push/MkDocs/dependency/Mini Tracker runtime change; no GLB or Cesium integration.

### MF-DEMO-5C-3 Google 3D Traffic Awareness — 22 September 2026

Status: MF-DEMO-5C-3 — GOOGLE 3D TRAFFIC AWARENESS IMPLEMENTED; ready for Product Owner visual review. Scope authorized after acceptance of 5C-1/5C-2. Feature Spec: `.kiro/specs/maker-faire-demo-cut/mf-demo-5c-3.md`. Report: `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_GEOAWARENESS_5C3_IMPLEMENTATION_20260922.md`.

Six DSC files: `public/mission3d/mission3d-model.js`, `mission3d-renderer.js`, `mission3d.js`, `mission3d.html`, `mission3d.css`, `functions/test/mission3d.test.js`. Integrated compact/normal Traffic Awareness card consumes exact validated core pairs; bounded selection in core order, safe 2D Italian terminology, independent freshness/provenance. Ground measured-endpoint line, removable area outline (zero INSIDE/BOUNDARY: outline only), subtle target heading emphasis, native interactive markers and labels default OFF. Never recalculate domain semantics or move camera on awareness. Revision/selection updates independent of RAF; iframe teardown preserves parent motion/core/2D. Actual Google rejected empty marker label; OFF now removes the attribute, covered by native-contract fake assertion.

Tests: 58 focused (28 previous + 30 new), 66 including controller, 328 across required 13 regression suites; all 0 failed/0 skipped. Four JS syntax checks and six-file whitespace/conflict-marker check passed. Hash comparison: exactly six changed of 169 DSC files; locked core/profile/geometry/controller/bridge/motion unchanged; 97 Mini Tracker runtime baseline files unchanged. Prior local changes preserved.

Actual Google browser rendered terrain/volume in IAB. Local explicit DEMO harness uses production 2D, bridge, 3D, core/controller and motion. Verified UAS/area ground paths, colors/states, approaching/diverging, STALE last-valid gray and independent freshness, UNKNOWN, zero area outline, rotorcraft fixture, labels ON/OFF, details selection, camera invariance, compact updates, pause/resume/restart, full loop and close/reopen current run, light/dark and 390 px. Stable camera heading 85/range 1575 over ~40 s; one map/one line/no duplicates. Multi-minute observation had no rendering halt after correction; not a GPU/FPS benchmark. Harness timing is parent 2D only and background RAF gaps reached ~1 s.

Limitations: external Edge unavailable to tools; product page on localhost:8766 is unauthenticated and old 8765 server unreachable, so no authenticated real-area browser validation. Mixed DEMO/LIVE and PROVISIONAL tested in Node, not private live browser data. No dense/hardware/physical touch validation. Generic native markers retained; no GLB or new helicopter scenario. Next: PO review then 5C-4; optional 5C-5 later. No deploy/commit/push/MkDocs/dependency/Mini Tracker runtime change.

### Field Operations 2D UI polish — 22 September 2026

Status: FIELD OPERATIONS 2D UI POLISH — IMPLEMENTED. Product Owner requested target label visibility and movable/compact panel after accepting 5C-2. Details and commands are recorded in `.kiro/specs/maker-faire-demo-cut/mf-demo-5c-2.md`, UI polish addendum.

DSC files: `public/js/field-operations/field-map.js`, `public/css/field-operations.css`, `functions/test/field-operations-map.test.js`, `functions/test/helpers/field-map-fakes.js`; one close-selector adjustment in `functions/test/field-nodes-ui.test.js` for the new header/body structure. Labels default OFF, checkbox toggles only owned target labels, with proper Leaflet repositioning when shown. Choice stays in the view instance through updates/close/reopen; no browser storage. Icons/popups/halo/relationships unaffected. Header-only pointer drag, map-bounded clamp, map-drag state restored on termination, cleaned handlers/observer. <=700 px map uses safe bottom-left anchoring. Compact preserves header/restore/close and live animation/awareness; close resets panel placement and compact state.

Tests: 60 focused passed (45 previous + 15 new), 0 failed, 0 skipped; full 13-suite regression 298 passed, 0 failed, 0 skipped (includes focused). Four JavaScript syntax checks, five-file whitespace check and scoped documentation diff check passed. SHA-256: only five expected changes among 141 DSC files; 97 checked Mini Tracker runtime files unchanged. Core/profile/geometry/controller, scenario/motion and 3D unchanged; existing local modifications preserved.

Actual local Leaflet browser review: target labels OFF/ON, icons/popups and awareness retained, rotorcraft fixture, all four panel corners clamped, camera unchanged during panel dragging and normal map pan/zoom afterward, reduced bar with original animation/core revisions continuing, restore/compact close/reopen, dark/light, 390 px safe re-anchoring. Physical touchscreen/pen and authenticated/hardware paths not tested. Cumulative browser logs include one isolated MutationObserver exception without source URL, no observed UI failure; source unconfirmed. Impact on 5C-3: NONE. No deploy, commit, push, new dependency/storage, Mini Tracker runtime change or Google 3D change; no MkDocs build.

### MF-DEMO-5C-2 Traffic Awareness 2D — 22 September 2026

Status: MF-DEMO-5C-2 — TRAFFIC AWARENESS 2D IMPLEMENTED. Product Owner authorized the 2D presentation slice only. Report: `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_GEOAWARENESS_5C2_IMPLEMENTATION_20260922.md`; Feature Spec: `.kiro/specs/maker-faire-demo-cut/mf-demo-5c-2.md`.

Modified DSC `public/js/field-operations/field-map.js`, `public/css/field-operations.css`, `functions/test/field-operations-map.test.js` and `functions/test/helpers/field-map-fakes.js`. Added bounded keyed Traffic Awareness rows driven only by scene.awareness; translated state/trend/reasons, distinct UAS/area horizontal distances, independent freshness and snapshot provenance. Preserved core order and subdued NORMAL; maximum five rows, or first four plus surviving explicit selection outside the first five, with omitted count. Dedicated selected relationship line uses measured endpoints; area INSIDE/BOUNDARY zero uses an independent outline and no zero line. Original geometry/style untouched. Small pixel halo, no geographic safety radius. Explicit ROTORCRAFT SVG, original RID/aircraft preserved; no production scenario rewrite. Awareness update is revision/context/selection driven, never RAF computation. Existing owner handles authorization teardown.

Validation: 45 focused map tests passed (12 existing + 33 new), 0 failed, 0 skipped. Full required 13-suite regression: 283 passed, 0 failed, 0 skipped, including focused tests. Added a direct immutable-core-snapshot integration check. Three JavaScript syntax checks, four-file whitespace check and tracker-scoped documentation diff check passed. SHA-256 baseline of 141 DSC files: only the four listed files changed; 97 Mini Tracker runtime files unchanged. Pre-existing local modifications preserved.

Real local browser/Leaflet review completed with explicit DEMO harness separately from Node: states/trends, UAS and area measured lines, zero outline, stale/unknown, rotorcraft fixture, keyboard selection, dark/light, desktop and 390 px viewport, original scene-motion with real geoawareness-controller, pause/resume, restart, run loop, close/reopen. Observed final view update p95 about 1.8–2.3 ms on this development machine; not a hardware or dense-core guarantee. Authenticated live session/logout not repeated in browser; owner/access regression covers logout/context/denial and late callbacks. Cumulative browser logs included one isolated MutationObserver exception without source URL; subsequent visual review showed no recurrence or rendering halt. Remaining dense-core performance risk unchanged from 5C-1.

5C-3 can render the same unchanged scene.awareness without recalculation. Deferred: Google 3D awareness and models, explicit helicopter scenario, real ADS-B/RID transport, record/replay, Preview3D/Airspace3D migration. No Mini Tracker runtime, profile/geometry/core/controller, scenario, owner, bridge or 3D visual change; no dependency, network/storage feature, deploy, commit, push or MkDocs build.

### MF-DEMO-5C-1 geoawareness core implementation — 21 September 2026

Status: MF-DEMO-5C-1 — GEOAWARENESS CORE IMPLEMENTED. Raffaello explicitly authorized implementation of the approved Design Lock and subsequently authorized the minimum 3D receiver adjustment needed for V03 validation/revision handling. Authoritative implementation report: `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_GEOAWARENESS_5C1_IMPLEMENTATION_20260921.md`. Feature Spec: `.kiro/specs/maker-faire-demo-cut/mf-demo-5c-1.md`.

Created DSC `public/js/field-operations/geoawareness-{profile,geometry,core,controller}.js`, four `functions/test/geoawareness-*.test.js` suites and `functions/test/helpers/geoawareness-fixtures.js` (including a reproducible benchmark). Narrowly changed DSC `public/index.html`, `field-node-card.js`, `scene-motion.js` and the authorized `public/mission3d/mission3d.js` receiver. The latter validates and forwards the immutable awareness snapshot, rejects stale/altered revisions, separates existing scene and derived-awareness size budgets, and imports no proximity core. Actual Leaflet/Google renderer implementations, bridge, styles and visible scenario are unchanged.

Implemented exact MT_HORIZONTAL_V1 asymmetric transitions; four-sample trend; trusted observation watermarks and future quarantine; independent reference/target freshness, provisional/last-valid stale explanations, grace expiry/tombstones and recovery; explicit scope; encoded identities; canonical geometry/revisions; global point/Circle and regional spherical Polygon/holes; immutable, bounded, ranked snapshots. Parent controller attaches `scene.awareness` before one store publication, uses the existing one-second freshness tick, and samples the existing frozen route at five-second checkpoints. Final 60000 ms checkpoint is emitted before loop reset. Logout/denial/context cleanup and late callbacks are covered. Demo classification remains UNKNOWN for the existing generic aircraft; no inferred helicopter/rescue label.

Validation: 68 focused tests passed, 0 failed, 0 skipped; all 53 Design Lock matrix identifiers present. Required nine regression suites: 182 passed, 0 failed, 0 skipped. Node syntax checks passed for 12 changed/new JavaScript files; those files have no trailing whitespace; tracker-scoped `git diff --check` passed. SHA-256 comparison: 149 existing DSC files checked, only the four authorized existing application files changed; 97 Mini Tracker runtime files remained unchanged. Existing local modifications were preserved.

Performance (Node v24.11.1 on Windows, injected clocks, 60 iterations; not browser acceptance): normal two-relation workload combined p95 0.265 ms; 512 relations/simple areas 18.010 ms; 512 relations/8000 vertices 54.056 ms combined p95, 12/60 combined iterations >=50 ms. Dense evaluation alone p95 49.874 ms, 3/60 >=50 ms; initial dense preparation 96.266 ms. Exact cap pruning, topology sweep and unchanged-geometry caching reduced the initial heavy workload substantially without reducing scope or precision. Maximum-density responsiveness remains an explicit Edge validation/performance risk; do not claim universal <50 ms or visual acceptance from these unit tests.

5C-2 can consume `scene.awareness` without recalculation. The 3D receiver can now receive/validate the same snapshot for 5C-3; awareness UI is deferred. Also deferred: GLBs, visible helicopter scenario, real ADS-B/RID transport, recording/replay storage/UI and legacy viewers. No Mini Tracker runtime change, new dependency, endpoint, Firestore write, credential, browser persistence, deploy, commit, push or MkDocs build.

### MF-DEMO-5C-1 geoawareness design lock — 21 September 2026

Raffaello approved the Geoawareness Discovery and requested an implementation-ready design lock, explicitly without application implementation. Created `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_GEOAWARENESS_5C1_DESIGN_LOCK_20260921.md` with the required 25 sections. Verdict: 5C-1 DESIGN LOCK — READY FOR IMPLEMENTATION.

Locked MT_HORIZONTAL_V1 exact/asymmetric transitions and trend/deadband precedence; defined corrected observation acceptance, non-latching stale recovery, separate LIVE/DEMO/reference clocks, stable encoded identities, local geometry revision, Circle/Polygon/holes geometry and unsupported-geometry behavior, immutable snapshot/ranking, parent ownership, bounded performance and privacy cleanup. Included the approved 60-second scenario oracle and implementation test matrix. Discovery was not repeated; inspection was limited to current parent/motion/iframe integration seams. Current owner is DSC field-node-card.js; deterministic motion sampling and awareness-only iframe updates are explicit future integration requirements.

Checks: 25-section validation, JSON parse/relationship-key/count checks, existing local-link checks and unchanged SHA-256 comparison of 130 inspected-subsystem source/asset files. No new core/runtime tests, application code, source contracts, assets, thresholds, transport, scenario, deployments, commits, pushes or MkDocs. Files changed: requested external Design Lock and this handoff entry only. The approved discovery and pre-existing local changes were preserved. Future implementation must use this contract and the existing Maker Faire Feature Spec; this documentation task did not implement 5C-1, 5C-2 or 5C-3.

### DSC+ geoawareness discovery — 21 September 2026

Discovery/design only, requested by Raffaello. Report: `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_GEOAWARENESS_DISCOVERY_20260921.md`. Status: READY FOR PRODUCT OWNER DESIGN REVIEW; no implementation authorized by this report. Google 3D remains authoritative for Field Operations; legacy viewers remain untouched.

Inspected actual proximity engine, providers, state/trend/freshness/UI, relevant configuration and documentation, DSC scene/geometry/Google adapters and seven existing GLBs. Documented exact threshold boundaries and asymmetric hysteresis, timestamp renewal limitations, sticky STALE recovery, unused configuration, missing normalized-aircraft retention and the world-bounds/point-provider query mismatch. Proposed shared renderer-independent awareness with separate aircraft–UAS and aircraft–area references; design details remain in the external report.

Checks: 12 in-memory characterization assertions against existing Python methods passed; a 13-row/two-reference 60-second scenario was evaluated using the existing classifier/trend methods. Seven GLBs inspected as binary/JSON metadata only, not rendered. Existing six-module pytest run could not execute: bundled Python lacks pytest; workspace venv Python failed to start with access denied. No packages installed, hardware/provider calls, application changes, asset additions, threshold/transport changes, deployment, commit/push or MkDocs. GLB rendering/performance and future live observation timestamps require separate validation after design approval. Files changed in this discovery: requested external report and this handoff entry only; pre-existing local changes preserved.

### Field Operations Google 3D — 21 September 2026

Latest PO validation: after deploying to the website and addressing the key's domain restriction, Raffaello tested the integrated view in his external Edge and reports it works perfectly, is fluid and has no blocking. Record this as a positive PO-reported real-browser product test, distinct from the earlier spike. Exact runtime, completed loop count, Edge version/GPU and individual control checklist were not supplied; do not claim the specific 10 x 60-second stress passed. The agent's internal-browser WebGL2 failure below remains an environment limitation, not evidence that the deployed Google renderer fails in Edge. No additional application changes or test reruns in this documentation-only update. Updated DSC report, external Demo Cut/report and both Google/MF5 specs with this evidence; deployment was performed by PO, not by the agent.

Authoritative Product Owner decision: Field Operations uses Google Maps 3D only. MF1/2 accepted; MF3 physical acceptance passed; MF4 visual acceptance passed; MF5A 2D animation accepted; MF5B Cesium animation ABANDONED for Maker Faire. The following Cesium investigation and previous delivery notes are historical, not instructions to resume debugging. Its unresolved RangeError / updateFrustums / createPotentiallyVisibleSet is not claimed fixed.

DSC `public/mission3d/` now loads Maps JavaScript/maps3d with an environment-supplied, browser-restricted key in Git-ignored google-maps-config.json (optional mapId). Replaced the engine-specific renderer/config, removed its CDN/styles/icon imports, terrain sampling and frustum/context workarounds. Native ground tracker, Polygon footprint/holes and 120 m relative extrusion, stable RID/aircraft marker and world-space heading arrows. Google innerPaths is absent without holes, including Circle and extrusion. One map; the existing shared interpolator consumes normalized samples; camera only changes initially or by explicit user interaction. Async loader shutdown and failure latching protect close/logout/context changes.

Preserved normalized scene/store/MF5 logical source, private iframe protocol, all public/private Mini Tracker transport, ingest/read/device-token/storage and 2D code. Baseline SHA-256 comparison: all 14 files under Preview3D/Airspace3D unchanged; all existing public/test files outside mission3d and its test unchanged for this migration. Airspace3D is only a future Google migration candidate; Preview3D remains Cesium until a product reason exists. No common 3D framework.

Validation: 194 DSC Node tests passed (mission3d, google3d-spike, scene-motion, field-operation-service, field-nodes-ui, field-operations-map, pilot-workspace-refactor, operational-place-ui, mission-workspace-ui, access-resolver). All four mission3d JS syntax checks and scoped whitespace checks passed. Ten accelerated logical loops test identity/counts/contracts, NOT GPU stability. Actual integrated iframe/2D commands verified for start/pause/resume/restart/loop, current-state reopen, close preserving 2D and simulated logout clearing the scene.

Visual limitation: PO reports the corrected isolated spike fluid and visually correct in their initial test. The agent's configured product attempts in the in-app browser fail before rendering with Google's explicit 'WebGL2 is not available on this browser'. Zero complete rendered loops; the preferred 10 x 60-second stress and Google real-area visual acceptance are not passed. External Chrome is not available to tools; user subsequently allowed the internal browser. After PO login, the actual localhost DSC flow was verified: Vescovio - Esercitazione, two real areas (Area delle operazioni / Area di ricerca) retained as STALE, tracker LIVE/MANUAL but OFFLINE, both areas received by the iframe as LIVE/MINI_TRACKER alongside DEMO motion. This authenticates real data handoff, not successful GPU drawing. Do not fabricate freshness or call mock tests visual acceptance.

Files in this slice: DSC mission3d six existing JS/HTML/CSS files, google-maps-config.example.json, ignored local config, .gitignore, functions/test/mission3d.test.js, developer report; Mini Tracker this handoff, MF5 spec status and new google3d-field-operations.md spec only. External Demo Cut/report updated. No Mini Tracker runtime changes, deployment, commit/push, device install or MkDocs. MF6 readiness is normalized sample compatibility only; recorder/replay not implemented.

### MF-DEMO-5 crash investigation — historical, abandoned for Maker Faire

The Product Owner confirms MF5 2D works correctly, but reports Cesium 1.114 Invalid array length immediately after the first drone movement. This remains an unresolved 3D defect, not an accepted browser-only explanation.

The failing official Cesium code is updateFrustums, called by createPotentiallyVisibleSet: an invalid computed frustum count reaches an array length assignment. Which scene element contaminates that calculation has NOT been identified. A temporary CPU check using the actual official Cesium 1.114 unminified bundle evaluated 5,551 target frames over the 60-second scenario: real ECEF positions/quaternions, Entity model matrices and ArcType.NONE PolylineGeometry remained finite. Terrain was synthetic in that CPU check; it does not validate WebGL, real terrain or model drawing.

Confirmed and fixed a separate error-lifecycle race in public/mission3d/mission3d-renderer.js: rendering failure now invalidates pending terrain updates, cancels the animation frame, disables the render loop and rejects later updates/camera actions. A regression reproduces failure while terrain sampling is pending and verifies that completion cannot restart rendering. Retained bounded local console diagnostics for camera validity/frustum limits and invalid command bounds; no coordinates, scene payloads, target IDs or tokens are logged. These diagnostics are for identifying the original defect once rendering is available, not evidence of its cause.

Validation: 21/21 mission3d tests, followed by the existing nine-file MF5 regression command: 175/175 passed, zero failed/skipped. Both changed JavaScript files passed node --check; scoped git diff --check passed. No 2D/source/transport/device implementation changed in this investigation.

Browser blocker independently confirmed on a temporary minimal canvas page, outside Cesium: BOTH webgl2 and webgl context creation fail with GL_VENDOR = Disabled, GL_RENDERER = Disabled, BindToCurrentSequence failed. New tab and subsequent retry after PO reported reopening still failed before scene drawing. The diagnostic page was removed. No browser security/GPU settings were modified. Remaining: a functioning WebGL browser, capture of the first-motion diagnostic, root-cause correction and the full 3D acceptance cycle. No claim that the original RangeError is fixed. No deploy, commit, push or device installation.


### MF-DEMO-5 — animated targets implemented; final WebGL review pending (2026-09-20)

- Current PO status: MF-DEMO-1/2 PRODUCT OWNER ACCEPTED; MF-DEMO-3 PRODUCT OWNER PHYSICAL ACCEPTANCE PASSED; MF-DEMO-4 PRODUCT OWNER VISUAL ACCEPTANCE PASSED in the real DSC application (terrain, ground tracker, synchronized real area, DEMO volume/targets, camera and 2D return). Do not reopen discovery or redesign the accepted view.
- MF5 code is implemented, but NOT yet marked ready for visual acceptance: the mandatory complete 3D animation/control cycle could not finish because the in-app browser lost WebGL. Early actual 3D frames showed TERRAIN OK, moving low RID/higher aircraft and an orbit; later Cesium reported Invalid array length and subsequent context creation failed even in a fresh tab. PO confirms WebGL also failed in the accepted static scene on zoom-out and normally needs a restart. Cause is not proven and fixes must not be described as verified resolution. Resume actual WebGL validation after browser recovery.
- New DSC public/js/field-operations/scene-motion.js: deterministic 60-second scenario anchored to current first valid area, else tracker. RID waits 3 s, traverses 4 legs, turns near 35 s, returns at 50 s; aircraft enters 15 s, crosses, removed at 45 s. Anchor stays fixed during the run while real area edits still render. Start/Pause/Resume/Restart/optional Loop controls in both views. Only designated DEMO IDs move; real tracker/areas/team and other targets are retained.
- Source samples at 1 Hz (plus explicit control events), monotonic elapsed clock, observedAt and motion {state, elapsedMs, runId, sampleAt, loop, durationMs, bounds}. Existing store is sole normalized logical scene; controller.subscribeScene allows MF6 to capture {elapsedMs, scene} before presentation interpolation. No recorder/history/export/import. Shared presentation-only bounded interpolation, shortest heading turn, no extrapolation; stale >3 s, expired >10 s. Explicit paused/completed DEMO holds. Newly opened 3D receives current logical sample and joins interpolation on the next sample (up to one interval of initial visual offset).
- 2D updates stable marker objects/popups/tooltips; 3D retains Viewer, terrain, area/volume groups, target entities/model graphics and dynamic guide property. Camera stays free except explicit Ricentra. RID remains DEMO 35 m AGL, aircraft preserves source 2400 ft MSL and separate display terrain +600 m. Volume stays DEMO 120 m AGL. Rendering interpolation never writes to store. Real heartbeat/private area updates recompose moving targets without restarting elapsed time.
- Lifecycle: 3D close preserves ongoing 2D motion. Field map close, logout/context/source change/private denial stop source timers. Close/reopen returns READY. Exact-window/origin/version and allowlisted commands gate iframe motion actions. Added synchronous same-origin child disposal before iframe removal, explicit graphics-context release, one-attempt WebGL failure latch, stopped rendering after render errors and dynamic straight vertical guides. These resource/graphics mitigations passed unit tests but still require actual 3D revalidation after browser recovery. Temporary diagnostic logs were removed.
- DSC files: new scene-motion.js and functions/test/scene-motion.test.js; modified field-map.js, field-node-card.js, mission3d-bridge.js, index.html, field-operations.css, mission3d/{mission3d.html,mission3d.css,mission3d.js,mission3d-model.js,mission3d-renderer.js}, tests field-nodes-ui.test.js/mission3d.test.js/helpers/field-map-fakes.js. MT runtime unchanged; only this handoff and new .kiro/specs/maker-faire-demo-cut/mf-demo-5.md. Existing MF4 doc/spec changes preserved.
- Tests from DSC/functions: node --test test/scene-motion.test.js test/mission3d.test.js test/field-operation-service.test.js test/field-nodes-ui.test.js test/field-operations-map.test.js test/pilot-workspace-refactor.test.js test/operational-place-ui.test.js test/mission-workspace-ui.test.js test/access-resolver.test.js — 174 passed, none failed/skipped (23 added MF5 tests). Eleven changed/new JavaScript files passed node --check; scoped whitespace checks passed. MF3 source/transport/rules and Preview3D/Airspace3D verified unchanged. No MT hardware/receiver/emulator tests rerun for this frontend-only slice.
- Actual 2D browser: isolated explicit DEMO scene, simulated auth/Workspace; progressive marker movement, orientation, pause held position/time, resume, truthful RID popup surviving source updates, restart/loop with single target markers and free zoom checked. Browser testing does not establish real private-cloud/hardware integration. Local temporary server remains available at http://127.0.0.1:8765/__field-node-review (localhost alias used in final 2D checks); use desktop width for filming.
- Delivery/status: external DSC_PLUS_MF_DEMO_5_IMPLEMENTATION_20260920.md, updated Demo Cut and MF4 acceptance note. Remaining: recovered WebGL full cycle, pause/resume/restart, orbit/zoom/Ricentra, loop and 3D close/reopen; then PO 18-step review on real area. No trails/follow camera, recorder/export/import, RF transport, generic POI, Mission V3, Flight Plan, messaging or provisioning. No commit/push/deploy, device install or MkDocs.

### MF-DEMO-4 — Field Operations 3D implemented (2026-09-20)

- Current Product Owner status: MF-DEMO-1/2 PRODUCT OWNER ACCEPTED; MF-DEMO-3 PRODUCT OWNER PHYSICAL ACCEPTANCE PASSED. The PO reports real Polygon create/update/delete, Circle, private remote synchronization, continued local operation offline, retained stale DSC geometry and latest-state recovery after Internet restoration. Excluded Point/POI "Punto di decollo" is expected; generic POI sync remains SHOULD HAVE / POST MF-DEMO-4, requiring separate authorization.
- Current MF-DEMO-4 status: PRODUCT OWNER VISUAL ACCEPTANCE PASSED in the real DSC application, as reported in the MF5 request. The following implementation/browser/release notes describe the original MF4 delivery before that acceptance. Dedicated DSC public/mission3d/{mission3d.html,mission3d.css,mission3d-config.js,mission3d-model.js,mission3d-renderer.js,mission3d.js}; new public/js/field-operations/mission3d-bridge.js. Modified DSC public/index.html, public/css/field-operations.css, field-map.js and field-node-card.js only for the CTA/overlay/lifecycle. Added functions/test/mission3d.test.js and extended field-nodes-ui.test.js. Mini Tracker changes in that slice were this handoff and .kiro/specs/maker-faire-demo-cut/mf-demo-4.md; pre-existing runtime/ZIP changes were outside it.
- Vista 3D passes the current normalized 2D scene in memory through a versioned same-origin/source-checked iframe handshake. No scene URL/storage, independent private reads, endpoints or MF3 transport changes. Logout/context/source changes, access denial and closing the 2D scene close/destroy private 3D state. Close/reopen leaves the accepted Leaflet scene intact.
- Reuses Preview3D Cesium 1.114/World Terrain asset 1/public client configuration and sampling concepts, plus existing Airspace3D icon helpers and existing drone/aircraft GLB assets. Existing viewer directories, assets, backend/rules and operational-source.js are unchanged. No shared framework or new dependency version. Static target entities update in-place by stable ID, ready for MF5 without rebuilding Viewer.
- Terrain samples ground the tracker and Polygon/Circle perimeter. Circle centre/radius stay canonical; discretization is rendering-only. Synthetic walls/roof/edges are labelled VOLUME DEMO 120 m AGL, not an authoritative Mini Tracker altitude. DEMO RID uses terrain + 35 m. Aircraft details preserve 2400 ft MSL, while separate DEMO display height is terrain + 600 m; no MSL/ellipsoid conversion or certified separation claim.
- Actual browser: Codex in-app Chromium/WebGL, localhost isolated preview, 1440x900. TERRAIN OK with real terrain and imagery; Polygon and test Circle (220 m), tracker, transparent volume/roof, drone and aircraft rendered. Dark/light chrome, orbit, zoom, recenter, selection/details, repeated close/reopen and intact 2D return verified. Initial iframe camera/layout rendering failure was fixed by resize-before-framing/render-loop startup; subsequent reload/reopens rendered successfully. Aircraft asset origin compensated per instance. Real account/private-cloud 3D review is not claimed: auth/Workspace were simulated and geometry/targets were explicitly DEMO. Existing MF3 physical evidence is PO-reported, independent of this browser test.
- Validation from DSC/functions: node --test test/mission3d.test.js test/field-operation-service.test.js test/field-nodes-ui.test.js test/field-operations-map.test.js test/pilot-workspace-refactor.test.js test/operational-place-ui.test.js test/mission-workspace-ui.test.js test/access-resolver.test.js — 151/151 passed. Includes 16 new 3D tests and lifecycle integration tests; late terrain failures cannot downgrade newer scenes. Nine new/changed JavaScript files passed node --check. Scoped whitespace checks passed. Mini Tracker runtime tests, physical hardware and Firestore emulator were not rerun in MF4 because those execution paths were unchanged. Existing viewer files were verified unchanged; no new end-to-end browser test of their separate workflows.
- Delivery: external DSC_PLUS_MF_DEMO_4_IMPLEMENTATION_20260920.md and updated Demo Cut; local review http://127.0.0.1:8765/__field-node-review (temporary isolated server). Terrain fallback is visible and unit-tested but does not count as terrain acceptance. Cesium/CDN/imagery/terrain require network; Mini Tracker offline operation remains independent. Remaining: PO authenticated real-area 18-step visual review, then MF5 animation. No target animation, recording/replay, real RF target transport, POI, Mission V3, Flight Plan or messaging. No MF4 deploy, commit/push, device installation or MkDocs.

### MF-DEMO-3 — real saved operational areas implemented (2026-09-20)

- Current status: PRODUCT OWNER PHYSICAL ACCEPTANCE PASSED (scope/evidence above). The following implementation/test/release notes describe the original MF3 delivery, before the subsequent PO physical validation; they are not current deployment-state claims.
- Added backend/services/field_scene.py and field_sender.py. Read selected saved mission/layers in-process, exclude imports/POIs, normalize Polygon/Rectangle/Circle and small FeatureCollections. Preserve names, optional color/radius and stable serial:mission:layer:index IDs. No invented altitude or RF data. Missing storage, malformed/oversize geometry and selection races emit ERROR without clearing the last good remote projection; valid deletion/no selection publishes an empty array.
- app.py imports the sender and starts it only under normal script execution. System Update's existing import-app verification cannot start a second private uploader. Existing public heartbeat, RID bridge, receiver/config/storage implementations and local UI are unchanged.
- Independent collector and uploader daemon threads, 5-second sampling, latest-only buffer, one request in flight, 2/3-second connect/read timeout, no redirects/body download, 5–30-second retry backoff. Disabled without explicit private configuration; existing dsc-node02 identity required. Configuration is outside the update tree at /etc/tracker-mini/field-operations.json (enabled/token); optional FIELD_OPERATIONS_CONFIG path override. Cloud secret name FIELD_OPERATIONS_DEVICE_TOKEN. No secret created or committed.
- DSC files: functions/fieldOperations/{field-operation-service,field-operation-functions}.js, functions/index.js, firestore.rules; new public/js/field-operations/operational-source.js; existing field-node-card.js composition/lifecycle; field-map.js optional color only; public/index.html load order. Private latest doc fieldOperationsLatest/MTRK26-0001 stores bounded projection JSON (Firestore nested-array limitation), content hash/freshness/status. Ingestion validates bearer token, identity, strict schema, coordinates, bounds and transactional ordering. No writes to trackers/air_traffic_objects.
- readFieldOperation authenticates the two supplied UIDs and reuses resolveAccountAccess for ACTIVE DSC_PLUS/workspaceSync on every call. Rules deny all direct client reads/writes. Polling preserves last good areas on transient/device errors, shows STALE after 15 seconds, clears private state on access denial/logout/context change and ignores late callbacks. Public tracker remains LIVE/MANUAL and RID/aircraft stay DEMO; explicit DEMO mode unchanged.
- Tests added: tests/test_field_operations.py (22/22 via Python stdlib unittest; also pytest-compatible); DSC functions/test/field-operation-service.test.js and extended field-nodes-ui.test.js; test/security/field-operations.rules.test.js. DSC regression run: 133/133 (MF1/2, Workspace, operational place, Mission Workspace, access resolver). Actual Firestore emulator: 4/4, including projection storage, all-client direct read/list/write/delete denial, public tracker read unchanged and immediate membership revocation. Emulator used demo-dsc-field-operations only; Java ran after approved sandbox escalation. Syntax and scoped whitespace checks passed.
- Bundled Python lacks requests and pytest; new tests run without installing dependencies. Temporary storage tests use real mission/layer readers with the unrelated zone-download import stubbed. Full Flask/hardware startup, actual HTTP/Secret Manager deployment and physical <=10-second delivery have not been validated. No emulator/mocked test is hardware proof.
- Browser: refreshed existing isolated localhost harness to load the new source. Verified real public heartbeat, zero fabricated LIVE areas when private service is unavailable, and unchanged explicit DEMO polygon/RID/aircraft scene with real Leaflet. Auth/Workspace are simulated; preview cannot validate real private cloud synchronization.
- Documentation: new .kiro/specs/maker-faire-demo-cut/mf-demo-3.md; frontend/help/docs/developer/field-operation-sync.md and link from mission-storage.md; external DSC_PLUS_MF_DEMO_3_IMPLEMENTATION_20260920.md and updated Demo Cut plan. Manual includes exact configuration placement, deployment/package prerequisites, rollback and 17-step physical acceptance checklist.
- Original release/physical-validation prerequisites were subsequently superseded by the PO's passed physical acceptance listed above. Do not change the accepted sender/token/ingestion/read contract for MF4. No additional claim is made here about unreported restart or both-account tests. Current 3D implementation is described in the MF4 record above; live RF transport, animation, recording/replay and Mission V3 remain deferred.

### MF-DEMO-2 — Field Operations 2D implemented (2026-09-20)

- MF-DEMO-1 and MF-DEMO-2 are PRODUCT OWNER ACCEPTED per the MF-DEMO-3 implementation request. The implementation details below record the completed MF-DEMO-2 slice.
- Existing uncommitted MF-DEMO-1 changes were preserved. Current branches remain main. No commit, push, deploy, device installation or MkDocs.
- Visible scene: original real public tracker coordinates and MANUAL provenance; one fixed Vescovio DEMO Polygon, one DEMO RID and one DEMO aircraft. No new Internet traffic integration. The approved card appearance is unchanged.
- Extended the same scene with per-object origin/source, operation metadata, normalized geometry and optional explicit altitude unit/reference. Mixed scene keeps tracker LIVE; synthetic additions do not relabel it DEMO. Unknown provenance is not inferred as live.
- New field-map.js owns a single Leaflet feature group and one compact bottom-left control with counts, source legend, Ricentra and Chiudi. Source/controller lifecycle remains in field-node-card.js. Initial open fits once; repeated opens, heartbeat/freshness updates and geometry updates respect the camera. Recenter includes only owned layers.
- Renderer handles GeoJSON Polygon rings/holes and normalized Circle {center:[lon,lat],radiusM}; tested adapter accepts Mini Tracker Point Feature with leafletType=Circle/radius. Stable IDs reconcile updates/deletions without duplicating layers; invalid objects are skipped with visible notice. Popups use text nodes.
- MF-DEMO-3 readiness: YES. It can supply normalized real saved areas into the same store/renderer; bypass the explicit withDemoOperation source composition. No map UI architecture change is needed.
- DSC files changed relative to MF-DEMO-1: public/index.html; public/js/field-operations/{scene-store,fixture-source,field-node-card}.js; functions/test/field-nodes-ui.test.js. New: public/js/field-operations/field-map.js; public/css/field-operations.css; functions/test/field-operations-map.test.js; functions/test/helpers/field-map-fakes.js.
- Mini Tracker: this handoff and .kiro/specs/maker-faire-demo-cut/mf-demo-2.md only. No runtime changes; public transmissions and offline receiver/UI functions remain untouched.
- Checks: node --test test/field-operations-map.test.js test/field-nodes-ui.test.js test/pilot-workspace-refactor.test.js test/operational-place-ui.test.js test/mission-workspace-ui.test.js from DSC/functions passed 109/109 (13 added in this slice; 30 focused Field Operations/Nodes tests total). Syntax checks passed for all seven changed/new JS files; scoped diff whitespace passed.
- Browser: existing local isolated harness, real public heartbeat and real Leaflet; mixed LIVE/DEMO objects, Polygon, RID popup/altitude, desktop dark/light, pan, Ricentra, close leaving base map intact. Automated tests cover repeated update, source switch, logout/context and Circle. Harness still uses a simulated access/Workspace wrapper; full authenticated DSC integration and hardware are not claimed as tested by this slice.
- Development report: C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_MF_DEMO_2_IMPLEMENTATION_20260920.md. Plan and MF-DEMO-1 acceptance status updated there, outside product docs.
- Next: Product Owner review of MF-DEMO-2, then MF-DEMO-3 real Mini Tracker saved-area transport. No sender/token/private scene, real area sync, live target transport, 3D, animation, recorder/replay, Mission V3, Flight Plan, messaging or provisioning implemented.

### MF-DEMO-1 — Field Nodes card implemented (2026-09-20)

- Status: PRODUCT OWNER ACCEPTED (confirmed by MF-DEMO-2 request). Changes uncommitted on current main branches. No deployment, push, device installation or MkDocs.
- Owner: Codex for this explicitly authorized DSC UI slice; Mini Tracker runtime ownership and hardware remain unchanged.
- Real-position addendum supersedes the earlier fixture-first assumption. Default is a read-only subscription to existing public `trackers/dsc-node02`; map to display serial MTRK26-0001. A focused public read verified its ID and published position 42.33163 / 12.60444. Preserve actual published coordinates; explicit DEMO fallback uses PO site 42.331704 / 12.604630.
- MANUAL is PO-confirmed deployment configuration, not GPS or a source field emitted by the legacy heartbeat. Physical tracker may be at home. Never move RF observations to the configured site; future scene adapters must distinguish LOCAL/RX, NETWORK/INTERNET and DEMO.
- Existing heartbeat lastSeen drives ONLINE <= 90 s, STALE <= 180 s, OFFLINE thereafter. Missing/invalid/future time is not ONLINE. Receiver capabilities show SUPPORTATO / NON NOTO; reception health is not claimed. Explicit DEMO has labelled simulated receivers and deterministic states.
- UI exposure uses existing authenticated ACTIVE DSC_PLUS / workspaceSync context plus the two supplied UIDs. This is a public-data/demo UI gate, not private-scene server authorization.
- CTA reuses window.map and owns one removable marker. Mode/context/logout invalidate callbacks and clean up marker/listener/timer. Workspace close retains updates only while its marker is in use. Existing public tracker layers and Mission Planner state remain unchanged.
- DSC existing files changed: public/index.html, public/js/pilot-workspace.js. New: public/css/field-nodes.css; public/js/field-operations/{scene-store,fixture-source,presence-source,field-node-card}.js; functions/test/field-nodes-ui.test.js.
- Mini Tracker files changed: this handoff and new .kiro/specs/maker-faire-demo-cut/{requirements,design,tasks}.md only. No runtime changes. Future private sender must remain isolated from offline map, local missions, receivers and local awareness.
- Validation: from DSC/functions, `node --test test/field-nodes-ui.test.js test/pilot-workspace-refactor.test.js test/operational-place-ui.test.js test/mission-workspace-ui.test.js` passed 96/96 (17 new). `node --check` passed for four new modules, modified Workspace and new test file; scoped diff whitespace passed. Test log: temporary mf-demo-1-tests.log.
- Browser validation: isolated local harness with real Leaflet and public Firestore heartbeat; dark/light card, DEMO ONLINE/STALE/OFFLINE, focus on Vescovio, repeated CTA (one marker), simulated logout removes marker. Auth context and Workspace wrapper were test doubles; full real-account Workspace integration, mobile viewport, PWA install/cache and physical offline interruption were not browser-validated.
- Development delivery report: C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_MF_DEMO_1_IMPLEMENTATION_20260920.md. Revised Demo Cut plan records this addendum. No development docs under DSC/docs.
- Next: PO visual review, then MF-DEMO-2. No operational area/target rendering, private sender/token/latest-scene storage, network ADS-B adapter, RF transport, 3D, recording/replay or Mission V3 implemented in this slice.

### Maker Faire Demo Cut — Product Owner priority update (2026-09-20)

- Status: revised plan and targeted operational-area inspection complete; no application implementation, commit, push or deployment in this task.
- Priority authority: Raffaello's Demo Cut decisions supersede conflicting P0 delivery recommendations. P0 inventory remains the baseline; do not repeat broad discovery.
- Plan: `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_MINI_TRACKER_MAKER_FAIRE_DEMO_CUT_20260920.md`.
- Existing accounts supplied by Product Owner: DPT operator UID `n09Od1INlVfntbQXB7i16j4DfVt2`; associated pilot UID `8NwMdjaObwNiFVsf1uGgk4cAmYt2`. No new users or organization provisioning. UIDs are not assumed to be canonical operator IDs.
- New order: Pilot Workspace card with explicit DEMO fixture and working map-focus CTA; Field Operations 2D; saved Mini Tracker area projection; dedicated `public/mission3d/`; shared scene/motion; record/replay; live receivers; polish/rehearsal. Minimum grant/device-token checks precede real private data, not the first synthetic UI.
- Public/free tracker presence and awareness flows MUST remain intact. This supersedes P0's suggestion to suppress public RID forwarding. Private areas/team/richer status use distinct payload/storage; replay never enters public ingestion.
- Field Operation is primary. Mission V3, sidecar linking and pinned Flight Plan are not prerequisites. One server-owned two-account grant and static device token are sufficient for the demo; tracker does not authenticate DSC users.
- Targeted area findings: selected mission pointer plus per-layer JSON under `/home/pi/tracker-mini/missions`; existing current/layers CRUD APIs. Drawn polygons/rectangles are GeoJSON; circles are Point + feature properties `leafletType: Circle` and `radius`. Edits can serialize FeatureCollections. Show/Hide is browser-local, not persisted/publication state. No dedicated drawn-area altitude field. Exclude imported DSC regulatory zones from operational areas.
- Proposed projection: read saved current mission + user-drawn layers every scene cycle; full bounded area arrays with stable IDs; distinguish empty scene from transient read failure; optional clearly labelled demo volume height only. No Mission V3 schema changes or bidirectional sync.
- 10 October is now real Vescovio field trial plus video and scene-data recording; record/export/replay must be rehearsed beforehand. Keep real capture provenance and distinguish playback from live/synthetic additions. Meshtastic and event-live overlays remain optional to the critical path.
- Files changed: this handoff and the external Demo Cut plan only. Checks: targeted source/manual inspection, document structure/link validation and scoped diff whitespace. No tests rerun because application code is unchanged; P0 test results are not new implementation validation.
- Next implementation: MF-DEMO-1. Create the bounded Feature Spec at implementation start; preserve existing viewers/local workflows. No Production deployment or commit/push authorized by this task.

### P0 — DSC+ / Mini Tracker Maker Faire integration discovery (2026-09-20)

- Owner for this discovery: Codex, at Raffaello's explicit cross-repository request. Existing implementation ownership is unchanged.
- Status: discovery complete; implementation awaits Product Owner review. No application code, hardware configuration, deployment, commit or push was performed.
- P0 report now verified at `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_MINI_TRACKER_MAKER_FAIRE_DISCOVERY_20260920.md` after Raffaello's copy. Raffaello explicitly authorized writes to that development directory. Historical recommendations below are superseded where the Demo Cut entry conflicts.
- Inspected baselines: Mini Tracker `f80d078` on `main`; DSC `4a6a01a`. Both scoped workspaces were initially clean.
- User-confirmed account context: DSC+ has exactly two existing accounts, the Drone Pilots Team operator account and its associated pilot Raffaello Di Martino. Reuse these accounts; persisted UID/operator/role/feature mappings were not queried.
- Findings: local status/GPS/ADS-B/RID/Meshtastic/proximity services are reusable. Existing heartbeat/RID upload writes to DSC public collections and cannot carry the private operational scene. The current proximity API returns pairs, not a full target catalog. Local ADS-B read timestamps and altitude provenance need focused correction before trustworthy live export.
- Proposed decisions, not implemented: outbound authenticated private snapshot push; reuse DSC account/operator authorization; Mission sidecar binding; isolated deterministic DEMO session through the same ingestion/read path. Raffaello explicitly permits a dedicated DSC+ 3D viewer: recommend `public/mission3d/`, reusing selected Preview3D path/terrain code and Airspace3D target-rendering parts while retaining both existing viewers.
- First proposed slice: private binding/contract skeleton, followed promptly by Mission/Flight Plan/terrain. No implementation started.
- Files changed by P0: this handoff and the external discovery report only. Product manuals and generated help remain unchanged.
- Checks: DSC targeted tests from its functions directory passed 126/126 (mission geometry, Flight Plan contract, M3 read gate, Mission Workspace UI, traffic aggregator). Initial root-directory invocation failed cwd-relative UI paths, resolved by using the package directory. Mini Tracker targeted pytest could not start via `.venv` (access denied); bundled Python lacks pytest. No dependencies installed.
- Delivery checks passed: 13 numbered report sections, valid JSON example, 17 existing local evidence links, balanced Markdown fences and scoped Git diff whitespace checks. DSC workspace remains unchanged; only this handoff changed in Mini Tracker.
- Limitations: no Firebase runtime/account inspection, emulator security proof, browser/WebGL/terrain visual validation, physical receiver validation or NAT end-to-end test.
- Remaining work: Product Owner review, bounded Feature Spec, approved test environment and existing-account mapping, private transport/security tests, Mission 3D by 27 September, physical integration by 1 October, replay by 4 October, freeze 9 October, video 10 October. Raffaello remains responsible for device package installation and physical validation.

```text
Task: Multilingual Documentation Policy Update
Feature: Documentation workflow and Italian manual support
Owner: Codex
Working branch: main
Starting commit: Pending due local repository safe-directory restriction
Latest commit: Pending
Push status: Pending
Status: Documentation policy updated
Started: 2026-08-06
Last updated: 2026-08-06

Observed issue:
  - DOCUMENTATION.md required all documentation to be written only in English.
  - Raffaello wants Kiro to produce the Mini Tracker manual in Italian as part of a multilingual documentation workflow.
  - The intended workflow may use mkdocs-static-i18n.

Files modified:
  DOCUMENTATION.md
  AI_HANDOFF.md

Implementation:
  - Changed the documentation language policy from English-only to English canonical plus explicitly requested Italian localized documentation.
  - Added mkdocs-static-i18n guidance using the suffix structure, for example index.it.md next to index.md.
  - Documented that English remains the default/root canonical language unless Raffaello approves otherwise.
  - Documented that Italian pages should preserve verified technical meaning, warnings, UI labels and operational procedure order.
  - Confirmed that Kiro and Codex still must not run MkDocs; Raffaello runs MkDocs manually.

Tests/checks:
  - Passed: reviewed DOCUMENTATION.md language and mkdocs-static-i18n policy changes.
  - Passed: reviewed AI_HANDOFF.md documentation workflow guidance.
  - Passed: git diff --check -- DOCUMENTATION.md AI_HANDOFF.md.
  - Not run: MkDocs build, by explicit Raffaello instruction.

Known limitations:
  - mkdocs-static-i18n package installation and generated-site verification remain Raffaello-managed/manual steps unless the workflow is explicitly changed.
```

```text
Task: Map Trail Display Preferences
Feature: Browser-local map visualization controls
Owner: Codex
Working branch: main
Starting commit: Pending due local repository safe-directory restriction
Latest commit: Pending
Push status: Pending
Status: Local implementation complete, physical Mini Tracker validation pending
Started: 2026-08-06
Last updated: 2026-08-06

Observed issue:
  - Movement trail visibility and history duration needed operator-facing map controls.
  - Trail settings should be local to the browser/tablet display rather than Mini Tracker backend configuration.

Files modified:
  frontend/index.html
  frontend/css/drawer.css
  frontend/js/dashboard.js
  frontend/js/traffic/track-history.js
  frontend/js/air/air-layer.js
  frontend/js/drones/drone-layer.js
  frontend/js/meshtastic/meshtastic-layer.js
  frontend/help/docs/user/maps.md
  frontend/help/docs/user/traffic-monitoring.md
  frontend/help/docs/developer/frontend.md
  AI_HANDOFF.md

Implementation:
  - Added a Map Trails card inside the Maps drawer panel.
  - Added browser-local toggles and duration selectors for ADS-B, Remote ID drone and Meshtastic operator trails.
  - Added a Clear Trails action that removes currently displayed trails without disabling future rendering.
  - Stored preferences in localStorage using mapTrails.* keys.
  - Updated the shared track-history helper and each traffic layer so trail settings apply immediately without backend changes.
  - Post-deploy correction: moved the Map Trails card from the preceding Network drawer group into the actual Maps drawer panel after Raffaello reported it was not visible in Maps.

Tests/checks:
  - Passed: node --check frontend/js/traffic/track-history.js.
  - Passed: node --check frontend/js/air/air-layer.js.
  - Passed: node --check frontend/js/drones/drone-layer.js.
  - Passed: node --check frontend/js/meshtastic/meshtastic-layer.js.
  - Passed: node --check frontend/js/dashboard.js.
  - Passed: Node smoke test with mocked localStorage and Leaflet polyline/container behavior for trail enable, disable, duration persistence and clear.
  - Passed: git diff --check -- .
  - Not run: MkDocs build, by explicit Raffaello instruction.

Known limitations:
  - Browser visual behavior must be validated by Raffaello after installing the package through System Update on the Raspberry Pi Mini Tracker.
  - Trail preferences are browser-local and do not synchronize across tablets or browsers.
```

```text
Task: Documentation Build Responsibility Clarification
Feature: Documentation workflow boundary
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
  - Clarified that Kiro and Codex must not compile documentation with MkDocs during normal development or validation.
  - Documented that Raffaello manually runs MkDocs and regenerates documentation output when needed.
  - Updated documentation workflow guidance and current architectural decisions.

Tests/checks:
  - Passed: git diff --check -- AI_HANDOFF.md.
  - Not run: MkDocs build, by explicit Raffaello instruction.

Known limitations:
  - Documentation output regeneration remains a manual Raffaello step.
```

```text
Task: Traffic, Drone and Operator Trail Fade
Feature: Movement history visualization
Owner: Codex
Working branch: main
Starting commit: Pending due local repository safe-directory restriction
Latest commit: Pending
Push status: Pending
Status: Local implementation complete, physical Mini Tracker validation pending
Started: 2026-08-06
Last updated: 2026-08-06

Observed issue:
  - ADS-B aircraft trails could remain impressed on the Dashboard map after aircraft and helicopters were no longer present.
  - Remote ID drones and Meshtastic operators did not have category-specific movement trails to support search-pattern awareness.

Files created:
  frontend/js/traffic/track-history.js

Files modified:
  frontend/index.html
  frontend/js/air/air-layer.js
  frontend/js/drones/drone-layer.js
  frontend/js/meshtastic/meshtastic-layer.js
  frontend/help/docs/user/traffic-monitoring.md
  frontend/help/docs/user/teams.md
  frontend/help/docs/developer/frontend.md
  frontend/help/docs/hardware/ads-b.md
  frontend/help/docs/hardware/remote-id.md
  frontend/help/docs/hardware/meshtastic.md
  AI_HANDOFF.md

Implementation:
  - Added a shared frontend track-history helper that stores timestamped movement points, samples by movement distance, renders faded Leaflet trail segments and removes expired segments.
  - Replaced the old ADS-B trail implementation with the shared helper, fixing stale segment cleanup and keeping aircraft trails short.
  - Added blue dashed Remote ID drone trails with a longer retention window for search-pass and perimeter movement awareness.
  - Added green dotted Meshtastic operator trails with the longest retention window for slow team movement and search coverage awareness.
  - Trails are cleared when their source layer is stopped and otherwise fade/remove independently from marker lifecycle.

Tests/checks:
  - Passed: node --check frontend/js/traffic/track-history.js.
  - Passed: node --check frontend/js/air/air-layer.js.
  - Passed: node --check frontend/js/drones/drone-layer.js.
  - Passed: node --check frontend/js/meshtastic/meshtastic-layer.js.
  - Passed: Node smoke test with mocked Leaflet polyline/container behavior for update and clear cleanup.

Known limitations:
  - Browser visual behavior must be validated by Raffaello after installing the package through System Update on the Raspberry Pi Mini Tracker.
  - Local checks do not validate real ADS-B, Remote ID or Meshtastic timing on hardware.
```

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

---

## 2026-09-30 — Real DS110 RID in private Field Operations LIVE (local implementation)

Work completed: traced DS110 in-memory `remoteid_aircraft` and `/api/remoteid/aircraft`, the separate public DSC RID bridge, the private field snapshot, and the DSC LIVE scene consumers. Added a location-only `position_observed_at`, a bounded private RID target projection (32 targets, 10-second position age), and the corresponding private DSC validation/composition. The same LIVE target reaches Geoawareness (RID-to-area and aircraft-to-RID without self-pairs), 2D, Google 3D and recording/replay in local tests. No area-containment filter, ADS-B integration, public RID-path change, or Mini Tracker UI change was made.

Mini Tracker files modified: `backend/services/ds110.py`, `backend/services/field_scene.py`, `tests/test_remoteid_stale.py`, `tests/test_field_operations.py`, `frontend/help/docs/developer/api.md`, `frontend/help/docs/developer/field-operation-sync.md`, `frontend/help/docs/hardware/remote-id.md`, and this handoff. Feature design: `.kiro/specs/real-rid-field-operations/requirements-design.md`. DSC changes are in the separate `C:\Users\raffa\DroneSkyCheck` workspace (private function, operational source, Geoawareness selection, target altitude validation/presentation, focused tests); its pre-existing local edits were preserved.

Decisions: stable ID `rid:remoteid:<serial>` or `rid:dji:<serial>`; `last_seen` remains the generic local track age, while only valid location data refreshes the private position age. A valid OpenDroneID geographic altitude is `WGS84_ELLIPSOID`; the separate `height` is withheld because its reference type is not decoded. LIVE Google 3D retains its truthful unresolved-altitude ground marker fallback. Area-only private snapshots remain accepted. Private target expiry is enforced on the DSC client even when cloud last-good data is retained.

Checks executed: Mini Tracker `unittest discover -s tests -p test_field_operations.py -q` (26 passed); four `test_remoteid_stale.py` functions executed with an inline monkeypatch shim because the available bundled Python lacks pytest (4 passed); DSC focused Node suites covering private service, LIVE UI/map, Geoawareness, 3D, scene motion, recorder and replay (398 passed); Git diff checks for both workspaces. Earlier local `.venv` pytest execution was denied by the environment. These are software tests only.

Known limitations and integration work: no physical DS110 run, cloud deployment, authenticated production read, actual Google 3D rendering or RF stop/restart test was performed here. The supplied physical observation of Dronetag serial `1596A34EE1D16FD` predates these code changes. Deploy updated Mini Tracker backend plus DSC function/frontend through the approved process, then confirm the real target and loss/recovery on hardware. The public DSC traffic layer and private Field Operations layer may render the same RID separately if both are visible; public-path deduplication needs separate Product Owner review. No commit, push or deploy was performed.
