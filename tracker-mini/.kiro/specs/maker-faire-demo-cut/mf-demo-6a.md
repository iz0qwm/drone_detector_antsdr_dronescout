# MF-DEMO-6A — recording format and recorder core

## Authority

Implement the approved MF-DEMO-6 Design Lock revision 2 and the Product Owner MF6-A request of 23 September 2026. No additional review gate. DSC source work is explicitly requested; Mini Tracker runtime is excluded. Existing local spec/handoff changes belong to prior tasks and must be preserved.

## Scope and acceptance

Pure V1 format/privacy projection, shared structural awareness validator, existing logical freshness policy materialized by the parent, immutable rescue scenario version, bounded detached full-frame recorder. One authoritative commit after composition/awareness/freshness; same-time terminal/restart commits retained. Hidden developer lifecycle only, no recording/import/export/replay UI or offline package.

Schema uses allowlisted fields and bounded values; no SDK/auth/account data. Clocks are injected monotonic versus UTC, metadata uses local build identity/UUID and recorded profile union. Limits match the Design Lock. Errors stop only the recorder, preserve the valid prefix and never interrupt consumers. Ordinary source close retains memory; authorization/context loss clears it.

## Components and compatibility

DSC public/js/field-operations: new awareness-schema, replay-format, scene-recorder; narrow scene-store, scene-motion, operational-source and field-node-card integration. Google validator extraction may update mission3d.js/html and its test loader only. Script order in public/index.html is updated. No core/profile/geometry, route, model/altitude/calibration, transport/backend or Mini Tracker runtime change.

## Failure/offline model

No recorder network or persistence. Capture only actual parent commits, not missed RF observations or RAF. Bound counts/bytes before append; failed first frame produces no recording. Duration check runs on parent logical ticks, also inside capture/stop. No claim of crash/OOM process recovery; MEMORY_PRESSURE supports explicit host stop. New start requires explicit discard of a retained recording.

## Verification tasks

- [x] Implement pure schema/validator and metadata/equality/byte checks with privacy and adversarial tests.
- [x] Implement recorder lifecycle/order/detachment/limits/failure tests.
- [x] Centralize parent final commits and logical freshness, test close/revocation/failing subscriber/3D independence.
- [x] Preserve scenario version through all controls and run a real-producer 60-second capture without renderers.
- [x] Run requested existing regressions, syntax/scope checks, and measured recorder performance.
- [x] Write external implementation report and update this spec and development handoff with exact evidence/limitations.

## Status

**MF-DEMO-6A — RECORDER CORE IMPLEMENTED.** Ready for MF6-B JSON export/import + validation worker.

56 focused tests (33 format, 23 recorder) and all 343 requested existing regression tests passed: 399 total, zero failures/cancellations/skips. Syntax checks passed for 13 JavaScript files; scoped whitespace and unchanged-component hash checks passed. Real-producer non-looping run: 62 frames, 60000 ms, 334415 compact UTF-8 bytes, terminal COMPLETE/SUSPENDED with area WARNING/INSIDE/CURRENT/0 m. Two terminal commits retain differing observation timestamps at the same tMs, as required by semantic comparison.

Measured Node v24.11.1 / Windows x64 / AMD Ryzen 5 8645HS: capture p50 0.3883 ms, p95 0.7939 ms, max 3.3525 ms; largest frame 5411 bytes; no timed call >=50 ms. Ordinary 601-frame ten-minute size projection 3236861 bytes. Browser Long Tasks, browser memory, fresh visual acceptance and hardware testing were not performed; no such results are claimed. MEMORY_PRESSURE is an explicit stop signal, not an automatic detector. No user UI, import/export, replay or offline implementation.

Report: `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_MF_DEMO_6A_RECORDER_IMPLEMENTATION_20260923.md`. It lists all 15 DSC files, integration, policies, exact tests/commands and performance methodology. No deploy, commit, push, dependency additions, Mini Tracker runtime changes or MkDocs.
