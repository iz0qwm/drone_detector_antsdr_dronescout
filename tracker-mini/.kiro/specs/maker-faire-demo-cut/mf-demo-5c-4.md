# MF-DEMO-5C-4 — Maker Faire helicopter scenario

## Authority

2026-09-22: Product Owner accepted 5C-3 visually and authorized this scenario slice. Domain profile/core/geometry, transport, rendering semantics and engine remain locked. No GLB, deploy, commit, push or Mini Tracker runtime changes.

## Requirements and design

- Keep canonical `demo:aircraft-01`; display `Elicottero 01`, explicit ROTORCRAFT/DEMO classification. Preserve `demo:rid-01` identity, altitude and existing bounded patrol.
- One 60 s logical sampler drives 2D, Google 3D and awareness. Helicopter present throughout, including final checkpoint. A continuous east-to-west flyby north of the operation establishes closest approach and subsequent divergence without discontinuous turns. Heading follows the actual path. This is a compressed demonstration timeline, not an aircraft performance model.
- Explicit fixture uses a 200 m circle. The route anchors on existing area bounds/center; real geometry remains untouched and no exact state timeline is promised for arbitrary real footprints. Frozen route within a run remains compatible with incoming area updates.
- Intended semantic acceptance: both references NORMAL -> MONITOR -> CAUTION -> MONITOR -> NORMAL, area enters Monitor before/equal UAS, approaching after warm-up, unknown near closest approach, diverging after sufficient history. Actual computed checkpoint distances and tolerances will be locked in tests after geometry inspection, independently of classifier helpers.
- Preserve labels OFF, camera freedom, existing controls, final checkpoint and new-run interpolation reset. Existing renderer supports explicit rotorcraft without changes.

## Tasks and validation

1. Focused source/fixture changes; inspect dependent tests for obsolete generic identity/presence assumptions.
2. Deterministic route/heading/cardinal tests and real-core/controller checkpoint oracle, independent relation IDs/history/provenance, pause/resume/restart/loop/60000 ms.
3. Required 13-suite regression, syntax/scope checks.
4. Full uninterrupted 60 s browser observation in review environment, then interaction/lifecycle tests in real Leaflet and Google; distinguish GPU evidence from mocks.
5. Implementation report and handoff. No MkDocs build.

## Implementation and verified results

Status: MF-DEMO-5C-4 — HELICOPTER GEOAWARENESS SCENARIO IMPLEMENTED. Ready for Product Owner review; the scenario is technically ready for 5C-5 model evaluation after acceptance.

Seven DSC files changed: `public/js/field-operations/scene-motion.js`, `fixture-source.js`, `public/mission3d/mission3d-model.js`, and `functions/test/{scene-motion,mission3d,geoawareness-controller,field-operations-map}.test.js`. Three application files and four test files. Existing unrelated local changes preserved.

The only presentation adaptation includes validated `motion.bounds` in initial/explicit-recenter Google camera framing. Without it, the previous area-only frame excluded the far helicopter. It does not follow targets or change the camera on updates. No awareness renderer semantics changed. The native Google marker representation and Leaflet rotorcraft SVG are reused.

The real sampler/controller/core oracle locks the following independently expected values to 0.05 m tolerance. UAS and area relation IDs, reference types, histories and provenance remain separate. No distance, state or trend is assigned by scenario code.

| Seconds | UAS m | Area m | UAS state | Area state | Both trends |
|---:|---:|---:|---|---|---|
| 0 | 3766.13 | 3305.76 | NORMAL | NORMAL | UNKNOWN |
| 5 | 3221.39 | 2822.01 | NORMAL | MONITOR | UNKNOWN |
| 10 | 2616.10 | 2359.07 | MONITOR | MONITOR | APPROACHING |
| 15 | 2097.27 | 1930.56 | MONITOR | MONITOR | APPROACHING |
| 20 | 1673.09 | 1561.79 | MONITOR | MONITOR | APPROACHING |
| 25 | 1339.43 | 1297.56 | CAUTION | CAUTION | APPROACHING |
| 30 | 1270.58 | 1198.43 | CAUTION | CAUTION | APPROACHING |
| 35 | 1336.92 | 1297.56 | CAUTION | CAUTION | UNKNOWN |
| 40 | 1633.16 | 1561.79 | CAUTION | CAUTION | UNKNOWN |
| 45 | 2014.39 | 1930.56 | MONITOR | MONITOR | DIVERGING |
| 50 | 2441.10 | 2359.07 | MONITOR | MONITOR | DIVERGING |
| 55 | 2880.00 | 2822.01 | MONITOR | MONITOR | DIVERGING |
| 60 | 3347.24 | 3305.76 | NORMAL | NORMAL | DIVERGING |

Focused: 28 scene-motion tests (16 previous + 12 new) and 60 mission3d tests (58 previous + 2 new), 88 total passed. Required 13-suite regression: 342 passed, 0 failed, 0 skipped. Seven JavaScript syntax checks and whitespace/conflict-marker checks passed. SHA-256 baseline: exactly seven changes among 169 DSC files; 97 Mini Tracker runtime files unchanged. Locked domain/controller/bridge/renderer implementations unchanged.

Browser: production modules in the existing explicit DEMO review harness on localhost:8766. Full uninterrupted 60 s runs in Leaflet and actual Google 3D produced the same 13 checkpoints. 2D camera count stayed at two throughout; Google center/heading/range remained identical throughout its untouched run. Verified actual terrain, helicopter SVG in 2D, native Google markers, labels OFF and ON/OFF toggles, UAS/area selection, orange CAUTION ground relation, compact panels, 2D panel drag (left 10 -> 610), map pan/zoom, Google orbit/zoom, pause with unchanged marker positions/revision over the observation interval, resume, restart, two loop boundaries and close/reopen on the current run. New-loop checkpoint returned both relations NORMAL/UNKNOWN. One map and stable marker/awareness element counts; browser captured no warnings/errors. Unit tests additionally assert final 60000 ms callback before next run and no end-to-start interpolation over three loops, revision-only ground updates and cardinal/turn headings.

Limitations: browser evidence is the explicit local DEMO fixture, not authenticated private LIVE areas or hardware. LIVE provenance/geometry preservation and a large LIVE footprint honestly producing WARNING/INSIDE are unit-tested. No exact canonical timeline is promised for arbitrary real areas. Smooth visual observation is not an FPS/GPU benchmark; parent harness timings measure 2D update work only. The 60 s flyby is time-compressed (~6.44 km); it is not an aircraft speed/performance model. Google retains native markers, presentation altitudes and its existing target heading representation. No GLB, Cesium, deploy, commit, push, MkDocs build, dependency or Mini Tracker runtime change.

Full report: `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_GEOAWARENESS_5C4_HELICOPTER_SCENARIO_20260922.md`.
