# Mini Tracker Agent Instructions

## Project Context

Mini Tracker is an operational Raspberry Pi-based system developed as part of the Drone Sky Check ecosystem.

It integrates multiple traffic, positioning and communication sources and provides a local operational interface suitable for portable and field use.

Before making changes, inspect the existing implementation and preserve the current project architecture, conventions and operational workflows.

## Required Reading

Before modifying the project:

1. Inspect the relevant source code.
2. Read the corresponding existing documentation under `frontend/help/docs/`.
3. Read `DOCUMENTATION.md` before creating or modifying documentation.
4. Read `AI_HANDOFF.md` to identify active work, task ownership and integration constraints.
5. Inspect configuration, service and deployment files related to the affected subsystem.

The current implementation is the primary source of technical truth.

Do not invent features, interfaces, device capabilities or operational workflows that cannot be verified in the repository.

## Development Rules

* Preserve the existing architecture unless a change is explicitly required.
* Prefer small, focused changes over broad refactoring.
* Do not modify unrelated files.
* Reuse existing services, helpers, models and frontend components where possible.
* Follow existing naming, logging, error-handling and API conventions.
* Maintain backward compatibility unless an approved specification explicitly requires a breaking change.
* Do not add dependencies without explaining why existing dependencies are insufficient.
* Do not replace working implementations solely because another implementation appears cleaner.
* Do not leave placeholders, incomplete stubs or simulated production behavior unless explicitly requested.

## Hardware Safety

Mini Tracker interacts with real Raspberry Pi hardware and external devices.

Do not change any of the following without first inspecting their current usage and documenting the reason:

* GPIO assignments;
* serial ports;
* serial baud rates;
* I2C addresses;
* network interfaces;
* system services;
* startup and shutdown procedures;
* hardware initialization order;
* power-management behavior.

Development-machine tests do not replace validation on the Mini Tracker hardware.

Code must handle unavailable hardware gracefully when running in development environments.

## Backend and Frontend

When changing a backend response or data model:

* inspect all frontend consumers;
* preserve existing fields where possible;
* document new or changed fields;
* update tests;
* verify failure and offline behavior.

When changing frontend behavior:

* preserve existing operational workflows;
* verify desktop and Mini Tracker display layouts;
* avoid unnecessary UI redesign;
* do not hide errors that are operationally relevant.

## Networking and Offline Operation

Mini Tracker may operate with intermittent or unavailable Internet connectivity.

New functionality must explicitly consider:

* offline behavior;
* retries and timeouts;
* stale data;
* duplicate messages;
* delayed messages;
* local persistence where appropriate;
* recovery after process or device restart.

Never assume continuous network connectivity.

## Testing and Verification

For every implementation task:

1. Identify the affected execution paths.
2. Add or update relevant tests.
3. Run the available tests and static checks.
4. Report exactly which checks were executed.
5. Report checks that could not be executed and explain why.
6. Verify that unrelated functionality was not intentionally changed.

Tests that mock hardware must not be presented as proof that the hardware integration works.

## Documentation

All documentation work must follow `DOCUMENTATION.md`.

Never edit:

```text
frontend/help/site/
```

Modify only documentation sources under:

```text
frontend/help/docs/
```

Documentation updates must reflect verified implementation changes.

## Feature Development

Complex features must be planned through a Kiro Feature Spec before implementation.

A feature specification should include:

* requirements and acceptance criteria;
* technical design;
* affected components;
* failure modes;
* compatibility considerations;
* testing strategy;
* implementation tasks.

Do not begin implementation until the specification has been reviewed when the task explicitly requires a review gate.

## Collaboration

Kiro is currently the primary and only Mini Tracker development agent.

A dedicated feature branch is not required. Work is performed directly on the current checked-out branch.

Commits must remain small and focused.

Pushed history must not be rewritten.

All Git operations must remain limited to `tracker-mini`.

Before modifying files:

* inspect `AI_HANDOFF.md`;
* verify no unrelated local modifications exist.

After completing meaningful work, update `AI_HANDOFF.md` with:

* work completed;
* files modified;
* decisions made;
* tests executed;
* known limitations;
* remaining integration work.

## Security

* Never commit credentials, tokens, private keys or passwords.
* Do not print secrets in logs.
* Do not weaken authentication or authorization checks.
* Treat all external input, network data and imported files as untrusted.
* Preserve existing privacy boundaries between public, operator and device data.

## Workspace Scope

These instructions apply exclusively to the Mini Tracker workspace located at:

```text
C:\sviluppo\Droni\drone_detector_antsdr_dronescout\tracker-mini
```

This directory is part of a larger Git repository that also contains other tracker systems and unrelated components.

The current Kiro workspace root is `tracker-mini`.

Kiro must:

* inspect, search, modify, test and document only files contained inside the `tracker-mini` directory;
* treat `tracker-mini` as the complete application workspace for the current development phase;
* create all Kiro steering files and Feature Specs inside `tracker-mini/.kiro/`;
* restrict Git diffs, staging and commits to files under `tracker-mini`;
* report any dependency on files outside `tracker-mini` without opening or modifying those files.

Kiro must not:

* inspect parent directories;
* inspect or modify sibling tracker implementations;
* search the complete monorepository;
* modify files outside `tracker-mini`;
* create Kiro files in the parent repository root;
* run repository-wide refactoring, formatting, testing or cleanup commands;
* stage unrelated changes from other projects;
* use `git add -A`, `git commit -a`, repository-wide search-and-replace or other commands that may include files outside the Mini Tracker workspace.

Git metadata may indicate that `tracker-mini` belongs to a larger repository. This does not extend the permitted development scope.

When running Git commands, restrict them to the current workspace whenever applicable, for example:

```bash
git status --short -- .
git diff -- .
git add -- .
```

Before every commit, verify that all staged files belong to `tracker-mini`.

If a required dependency appears to exist outside the allowed workspace, stop that part of the task and report:

* the required file or component;
* why it appears necessary;
* the expected integration;
* the minimum access that would be required.

Do not inspect the external component until Raffaello explicitly approves it.
