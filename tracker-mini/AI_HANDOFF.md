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

Kiro may:

* inspect the entire repository;
* create and update Feature Specs;
* modify backend and frontend code;
* add and update tests;
* update documentation;
* create focused development branches;
* connect to the physical Mini Tracker through the LAN;
* deploy approved development branches to the staging installation;
* execute hardware validation;
* prepare commits for review.

Raffaello remains responsible for:

* approving requirements and design decisions;
* approving potentially destructive system operations;
* validating operational behavior;
* approving merges into the stable branch;
* deciding when a feature is ready for operational use.

Codex is not currently working on the Mini Tracker repository.

No file ownership split between Kiro and Codex is required during this phase.

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

Status: Planned

The initial task is to inspect:

* repository architecture;
* backend services;
* frontend structure;
* application startup;
* hardware integrations;
* network configuration;
* traffic ingestion;
* Meshtastic integration;
* DSC communication;
* persistence;
* logging;
* tests;
* physical Mini Tracker deployment.

No new production feature should be implemented before the onboarding inspection and steering documentation have been reviewed.

Expected outputs:

* `.kiro/steering/product.md`;
* `.kiro/steering/tech.md`;
* `.kiro/steering/structure.md`;
* `.kiro/steering/workflow.md`;
* `.kiro/steering/hardware.md`;
* `.kiro/steering/documentation.md`;
* repository and deployment comparison;
* recommended staging workflow;
* recommended rollback workflow;
* first Feature Spec scope.

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

Status: Planned
Owner: Kiro
Specification: Not created
Implementation branch: Not created

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
Implementation branch: Not created

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
Implementation branch: Not created

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

All source code changes must be made in the development clone of the repository.

Do not use the physical Mini Tracker as the primary code-editing environment.

For each major feature:

1. create or update its Feature Spec;
2. review requirements;
3. review technical design;
4. review implementation tasks;
5. create a dedicated Git branch;
6. implement focused tasks;
7. run development-machine tests;
8. create a commit;
9. deploy the committed branch to the staging installation;
10. run physical hardware tests;
11. record the results in this file;
12. merge only after approval.

Recommended branch names:

```text
feature/traffic-proximity-awareness
feature/meshtastic-operational-network
feature/dsc-operational-area-sync
fix/<short-description>
```

Do not make unrelated refactoring changes inside feature branches.

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

* stable branch;
* stable commit;
* staging branch;
* staging commit;
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

* Kiro is the primary Mini Tracker development agent for this phase.
* Major features must use separate Feature Specs.
* Features will be developed sequentially.
* Development changes are made in the Git clone.
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

Repository remote: To be verified
Development branch: To be verified
Development commit: To be verified
Physical deployment path: To be verified
Physical branch: To be verified
Physical commit: To be verified
Python version: To be verified
Operating system: To be verified
Service manager: To be verified
Staging environment: To be defined

Kiro must replace these placeholders after the onboarding inspection.

---

## Test Status

Automated test framework: To be verified
Backend tests: To be verified
Frontend tests: To be verified
Hardware mocks: To be verified
Physical integration tests: Not started
Traffic simulation tools: To be verified
Meshtastic test support: To be verified
DSC integration test support: To be verified

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

---

## Active Work Record

No active implementation task has been started.

When work begins, add an entry using this format:

```text
Task:
Feature:
Owner:
Branch:
Specification:
Status:
Started:
Last updated:

Files modified:
Services affected:
Hardware affected:
Shared contracts affected:

Tests completed:
Physical tests completed:
Known issues:
Rollback reference:
Next action:
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
