# MF-DEMO-5C-5 — Native Google target models

## Authority and scope

2026-09-22: PO visually accepts 5C-4 and authorizes per-asset native Google model evaluation. Stability, readability, then visual quality. Keep existing marker fallback and unchanged normalized scene, sampler, awareness, 2D, camera and altitudes. No GLB edits/downloads/optimization, Cesium, Three.js, rotor animation, deploy, commit, push, MkDocs or Mini Tracker runtime changes.

## Design and acceptance

- Inspect actual assets and native Google API; calibrate orientation/scale without geographic coordinate corrections.
- Presentation-only class mapping and per-target lifecycle, at most one model per ID. Preserve current interpolation and fallback labels/selection. UNKNOWN stays marker-only.
- Native API has no documented model load-success event. Keep fallback visible conservatively; never infer success from time or map steady state. Synchronous failures and documented/standard errors must remain local and disable retries for the session.
- Evaluate helicopter alone, drone alone, marker baseline and both together. Preserve a per-asset rejection option if orientation/origin/readability/performance fails. Avoid repairing problematic assets.
- Observe complete 60 s story, preferred ten loops for accepted helicopter, camera interaction, controls, selection, labels, pause/resume/restart/loop and close/reopen. Actual browser evidence must be distinct from mocks and no invented FPS.
- Focused tests: mapping, orientation/wrap, stable resources and cleanup, failures/disabled/unsupported targets, awareness and camera independence. Required 13-suite regression and scope/hash checks.

## Tasks

1. Inventory and native API verification.
2. Bounded native presentation prototype with safe fallback, per-asset calibration and decision.
3. Lifecycle/fallback tests and browser gates; retain only approved product configuration.
4. Final report, spec and handoff with exact results and limitations.

## Implemented presentation

Six DSC files: `public/mission3d/mission3d-model.js`, `mission3d-renderer.js`, `mission3d.js`, `mission3d.html`, `mission3d.css`, `functions/test/mission3d.test.js`. Prior local changes preserved. No binary changes.

Verified assets: drone 490912 bytes / 8348 triangles / optional KHR_materials_specular; helicopter 3676360 bytes / 313140 triangles / no extensions; airplane_low 578792 bytes / 31068 triangles (inventory only, not enabled). No animation channels in these assets. Native Google rendered drone materials acceptably without conversion. Official API sources: [models](https://developers.google.com/maps/documentation/javascript/3d/models), [Model3DElement](https://developers.google.com/maps/documentation/javascript/reference/3d-map-draw#Model3DElement), [orientation](https://developers.google.com/maps/documentation/javascript/reference/coordinates#Orientation3D).

Original browser calibration (scale superseded by refinement below): helicopter heading correction +90, tilt 0, roll 0, scale 3; drone heading correction 0, tilt 90, roll 0, scale 4. Verified 0/90/180/270/359/1 visually in native Google, using a local isolated calibration page with a north-up camera and an independent direction line. Oblique view verified upright drone; top view verified rotorcraft rotor plane. Origins were usable without coordinate correction. These are constant presentation scales (~103 m helicopter length, ~50 m drone width), not true aircraft dimensions. At route overview distance the marker is more readable; models improve close inspection.

Default `Solo simboli`, session-only selector supports Drone / Elicottero / both. Each active class has a cached bounded same-origin HEAD availability check (10 s timeout); the native component loads the original GLB. HEAD success is not treated as native decode/render success. A marker and its selection/optional labels always remain, alongside an optional native model; no undocumented success event, timed marker hiding or blinking. No additional clock, altitude conversion, camera movement or domain mutation. At most two concurrent enhancements, one per target record. Position/orientation update only when pose changes; scale/source assigned on construction. Failures disable the affected class for the viewer session, without per-frame retries or resetting the scene. Error listeners are defensive; permanent marker visibility is the fallback for asynchronous native errors without an event. Pending work is invalidated on OFF/removal/dispose; no late resurrection. Reopening starts a fresh viewer session with markers only.

Tests: 81 focused (60 existing + 21 new), 363 across the required 13 suites, 0 failed/0 skipped. Includes mapping/unsupported classes, cardinals and wrap, exact interpolated pose/altitude, fixed scale, one instance per target, 10 simulated loop/restart cycles, pause/resume, disappear/re-entry/class changes, OFF/ON, missing/timeout/constructor/update/event failures, retained error notice, late work, close/reopen, <=2 bound, unchanged awareness and camera. Browser acceptance is not inferred from these tests.

Browser stress completed: ten full 60 s cycles with both calibrated native assets, review run IDs [1,8] through [1,17]. All ten terminal checkpoints were NORMAL/DIVERGING for both references; observations retained two models/one map. Orbit/zoom/Ricentra, selection, compact/restore, labels, pause/resume and close/reopen after stress were verified. Paused model poses matched across observations and matched marker positions. No new captured renderer errors during stress. A source-less MutationObserver exception captured before the first model load is recorded separately in the report. No FPS, native GPU memory or first-load duration claim. Reopened viewer starts Solo simboli, models off, on the current scenario run. Final separate configuration comparisons are recorded in the implementation report.

Final comparisons: drone-only full run [1,19], one model; marker-only full run [1,20], zero models. Both completed NORMAL/DIVERGING for both references with the unchanged final distances. Preliminary helicopter-only run [1,3] completed before final calibration; its definitive calibrated performance gate is the ten-cycle combined sequence. Decisions: drone ACCEPT, helicopter ACCEPT, both as optional native enhancements with retained marker fallback; airplane_low NOT EVALUATED/disabled. Default recommendation: Solo simboli for reliable overview, optional Drone ed elicottero for closer presentation. PO subsequently accepted model animation/fluidity; scale refinement is recorded below. MF-DEMO-6 normalized scene/recorder architecture is unaffected.

## Product Owner visual scale refinement — 22 September 2026

The Product Owner accepted model fluidity and animation, then requested a larger helicopter presentation symbol within this same slice.

Previous helicopter scale: 3. Final helicopter scale: 6. Drone scale: 4 unchanged. Scale 5 was evaluated first; 6 was retained because fuselage/tail are easier to distinguish while remaining clear of the operational volume and measured line. No value above 6 was evaluated. This is a presentation preference, not physical sizing. At the widest initial framing or a small viewport, the retained marker still dominates fine mesh details; unambiguous rotorcraft silhouette at that distance is not claimed as fully verified and remains subject to PO visual review. Close views show the helicopter shape more clearly. No camera or fallback changes were made to compensate.

Actual Google browser validation at 1280x900: scale 6 run [1,22] completed all 60 seconds with pause/resume, ending NORMAL/DIVERGING for both relations. Paused position/orientation attributes were identical across observations. Restart [1,23] returned to the route start and continued; two model elements remained throughout sampled observations, with scales 4 and 6. Tested Ricentra/initial framing, one-step moderate zoom out, closer views, orbit both ways, compact/restored Traffic Awareness and labels OFF. Map and controls remained responsive; no rendering stop or new renderer errors observed. Existing source-less MutationObserver exception at 03:27:42 UTC was already present before the scale-6 run and remains unattributed. Qualitative observations only, no FPS claim.

Focused regression rerun: node --test test/mission3d.test.js — 81 passed, 0 failed, 0 skipped. Model JavaScript syntax check passed. The earlier 363-test run above belongs to the original slice; it was not repeated for this constant-only refinement. Before/after comparison confirms one production change (ROTORCRAFT scale 3 -> 6) and one existing test expectation (3 -> 6). Loading, lifecycle, heading/tilt/roll, positions, altitude, route, awareness, camera logic, 2D, scene-motion and GLBs unchanged. Documentation updated in this report, existing Feature Spec and handoff. No deploy, commit, push, new slice or Mini Tracker runtime changes.
