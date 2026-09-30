# MF-DEMO-5C-4B — Rescue helicopter operational story

## Authority and acceptance

Product Owner accepts the current models/animation, including helicopter scale 6 and drone scale 4. Implement a deterministic 60-second DEMO arrival story without changing the geoawareness core, real areas, transport or model calibration. No review gate is requested before implementation.

## Design

Keep both target IDs, helicopter classification and altitude metadata. Revise only the synthetic routes: UAS patrol stays inside the canonical 200 m DEMO circle; at 30 s it leaves the search pattern toward an explicit synthetic waiting point southwest of the circle, stopping at 38 s. Keep 35 m AGL: this is a suspended operation, not a simulated touchdown. Helicopter approaches from east (3600 m from center) to the center by 60 s. Native horizontal core independently evaluates UAS and area; WARNING must begin before entry under the unchanged 500 m threshold. No guaranteed timeline is claimed for arbitrary LIVE footprints.

Add a bounded presentation-only `motion.scenario` record with fixed scenario ID, DEMO origin, UAS state and narrative phase. The same producer supplies normal and terminal snapshots. Both views display the same Italian narrative using a shared text helper, including an explicit operator decision/no automatic command statement. Keep the narrative visible above collapsible panel bodies. No new awareness states, priorities, thresholds or engine calculations. No rescue role inferred for LIVE targets.

## Scope and compatibility

Existing scene-motion module, compact narrative presentation in field-map and mission3d page, minimal styles, existing tests and bounded report/handoff. Loading, fallback, interpolation/run reset, controls, label defaults, SVG/GLB assets, model scales and free camera preserved. The shared motion record crosses the existing bridge unchanged; no transport schema or operational persistence. Offline behavior remains local deterministic motion; Google online availability/fallback is unchanged.

## Verification tasks

- Implement synthetic route and story; verify safe waiting point and patrol geometry with the real core.
- Derive a 5-second oracle from real geometry/thresholds; test separate UAS/area results, entry at 0 m, no forced core values.
- Test pause/resume, restart, loop terminal checkpoint and no end-to-start interpolation; identical 2D/3D story and bounded model lifecycle.
- Run the focused and existing 13-suite regression, syntax/scope checks.
- Run full uninterrupted 2D and actual Google 3D stories; test compact/normal, zoom/orbit, pause/resume, restart and loop. Report browser limits separately from automated evidence.
- Update bounded report and handoff. No deploy, commit, push, MkDocs or Mini Tracker runtime changes.

## Implemented result

Nine DSC files changed: scenario/motion source, 2D narrative row, Google page narrative row, two stylesheets and three existing test suites. No core/profile/geometry/controller/bridge/renderer/model/GLB changes. `motion.scenario` is emitted with terminal and normal samples; shared text remains outside collapsed bodies. Google scroll is restricted to scene-body so the operator narrative stays visible while reaching controls.

Measured canonical progression (area/UAS): 0 s NORMAL/NORMAL; 5 s MONITOR/NORMAL; 10–20 s MONITOR/MONITOR; 25–45 s CAUTION/CAUTION; 50 s WARNING/CAUTION; 55–60 s WARNING/WARNING. Area distance at 55/60 is 0 m, INSIDE. Trends UNKNOWN at 0/5, APPROACHING thereafter. Distances and independent oracle are recorded in the addendum report. UAS states: ACTIVE until 30 s, RETURNING 30–38 s, SUSPENDED from 38 s. Patrol independently checked within 200 m; waiting point ~382 m southwest, altitude stays 35 m AGL.

Automated verification: 182 focused tests / 368 across all 13 required regression suites, all passed, no skips. Five new tests plus updated scenario expectations. Six JS syntax checks and nine-file whitespace checks passed. Earlier fly-through timeline is intentionally superseded. No recorder implementation or new dependencies.

## Browser result and delivery

Actual 2D run [1,2] and native Google run [1,3] each completed 60 s uninterrupted and matched all 13 core checkpoints. Interaction run [1,4], loop into [1,5], compact/normal, drag/390 px in 2D, orbit/zoom/Ricentra, pause/resume and settled native pose equality verified. Two native models, scales 4/6, labels OFF. Full oracle, limitations and source-less pre-model MutationObserver exception are recorded in `DSC_PLUS_GEOAWARENESS_5C4B_RESCUE_STORY_20260922.md` under the authorized DSC+ reports directory. No rendering stop observed. PO visual review pending; no hardware/LIVE or FPS claims. No deploy/commit/push.

## PO refinement — DEMO helicopter ground approach, 22 September 2026

The PO visually accepted the rescue story and requested a visual descent inside the rescue area. This supersedes the fixed 600 m helicopter presentation at arrival, without altering the source altitude or the horizontal scenario.

Only the designated DEMO rotorcraft in RESCUE_ARRIVAL_DEMO receives the native Google presentation override. The adapter reads the existing CURRENT area relation (INSIDE/BOUNDARY and distance 0); it does not measure geometry or generate alerts. From scenario 55 s to 59 s the visual relative-to-ground altitude decreases linearly from 600 to 35 m. The final second settles at 35 before COMPLETE freezes interpolation. Existing interpolation smooths the presentation-only field, including model and marker together; pause holds it and run reset restores 600 m. The original source remains 2400 ft MSL; source target samples and awareness snapshot are not mutated. 35 m is the same above-ground presentation height as the drone, not identical absolute elevation over differing terrain, and not a touchdown.

Three DSC files changed for this refinement: public/mission3d/mission3d-model.js, public/js/field-operations/scene-motion.js (one conditional presentation-field interpolation), functions/test/mission3d.test.js. No source route, UAS movement, core, profile, geometry, renderer, camera, model scale/asset, 2D behavior or Mini Tracker runtime changes. The 2D samples do not carry this adapter-local field. Existing labels/details now report the actual visual height separately from source MSL.

Validation: 115 focused tests (85 mission3d + 30 scene-motion), 370 across all 13 regression suites, all passed, zero failures/skips. Two new tests cover gated descent, bounded heights, source/core independence, no descent outside or on historical measurement/non-DEMO scenario, smooth marker/model pose, pause, final 35 m, restart and stable identities/scales/camera. Two production JS syntax checks and three-file whitespace/scope checks passed. SHA-256 comparison confirms exactly the three listed changes against the immediate pre-refinement DSC baseline.

Actual Google browser run [1,6] completed 60 s: observed helicopter at 600 m at 33 s, approximately 183.6 m during interpolated descent at 58 s after INSIDE, and both native target elements at altitude 35 at COMPLETE. Two native models, scales 4/6, orbit/zoom responsive; no captured warnings/errors in this run. No hardware or actual landing validation. Existing scenario and historical test/browser records above remain as chronological evidence, with this refinement as current arrival-height behavior. No deploy/commit/push.
