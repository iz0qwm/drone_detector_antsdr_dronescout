# MF-DEMO-5C-3 — Google 3D Traffic Awareness

## Authority and scope

2026-09-22: Product Owner authorized implementation after accepting 5C-1 and 5C-2. Google is the only Field Operations 3D engine. This slice changes the mission3d presentation and its tests; no parent/domain, Mini Tracker runtime, scenario, deployment or dependency changes.

## Requirements and acceptance

Consume the validated immutable scene.awareness snapshot in core order. Preserve every semantic pair field without calculation. Show Italian state/trend, horizontal distance and measurement kind, separate endpoint freshness/provenance, safe reasons, and zero-distance footprint semantics. Selected relationship uses exact measured endpoints at ground level; an independent outline preserves canonical footprint and DEMO 120 m volume. Selection, labels (default OFF) and compact mode are view-local. Awareness never moves the camera or runs on animation frames. All four target classes remain selectable with native markers. Closing removes only iframe-owned state; reopening receives latest parent state.

## Design

- mission3d-model.js: presentation-only labels and revision/selection state, borrowing verified 2D terminology. No shared mapping currently exists, so avoid refactoring accepted 2D.
- mission3d.js/html/css: one integrated card, bounded relationship selector and detail text; current receiver validation remains authoritative. Accessible labels toggle and compact/normal control.
- mission3d-renderer.js: dedicated ground polyline/area outline, bounded ownership, subtle selected heading emphasis, native interactive markers; label preference independent of motion. Updates keyed by awareness context/revision/selection.
- Failure: existing Google error notice, parent continues. Missing snapshot means unavailable, never safe. Existing rejection of malformed, conflicting, old or retired-run snapshots stays intact.
- Compatibility: current bridge, provenance, motion interpolation, explicit orbit/zoom/recenter and configuration stay unchanged. No empty polygon innerPaths.

## Verification plan and tasks

1. Implement view model, overlay and Google adapters.
2. Test snapshot parity, states/trends, freshness/provenance, exact ground endpoints, zero outline, stable ownership, revision gates, labels/compact, camera and disposal.
3. Run all 13 required regression suites and syntax/scope checks.
4. Try real Google in available browser; separately record GPU evidence and fake-renderer coverage. External Edge preferred if accessible.
5. Update this specification, AI_HANDOFF and requested implementation report with exact results/limitations. No MkDocs build.

## Implementation results — 2026-09-22

Status: MF-DEMO-5C-3 — GOOGLE 3D TRAFFIC AWARENESS IMPLEMENTED. Ready for Product Owner visual review; not an automatic approval of the next slice.

Six DSC files changed: public/mission3d/mission3d-model.js, mission3d-renderer.js, mission3d.js, mission3d.html, mission3d.css; functions/test/mission3d.test.js. No bridge/parent/domain edits. View-model keeps the exact core pair; bounded selector uses first five, or first four plus a surviving explicit selection. Run/revision gate prevents frame-driven awareness reconstruction. Panel uses safe text nodes, exact 2D translations/colors/reason mapping and independent endpoint provenance/freshness. Selection is local and resets with a new context. Native interactive markers remain clickable when labels are OFF. Generic native markers support all four classes, which are explicitly identified in panel/optional labels; custom models and helicopter scenario are deferred.

Ground line uses targetPoint/referencePoint for UAS and targetPoint/nearestPoint for areas. Zero INSIDE/BOUNDARY omits line; transparent independent outline preserves footprint/120 m extrusion and polygon holes without empty innerPaths. Selected target heading arrow is modestly thickened/colored, no geographic radius. Fixed measurement geometry remains independent of interpolated target positions. Compact summary keeps updating; object list collapses independently, card scrolls within viewport. Native Google rejects label="": OFF removes the label attribute. This was discovered in the real browser and added to the fake API contract.

## Verification results

- Focused: node --test test/mission3d.test.js: 58 passed, 0 failed, 0 skipped (28 previous + 30 new; two previous checks now explicitly enable labels).
- Mission3D plus geoawareness-controller: 66 passed, 0 failed, 0 skipped.
- Required 13-suite regression: 328 passed, 0 failed, 0 skipped, including the 58 focused tests. Four JavaScript syntax checks and six-file whitespace/conflict-marker check passed.
- SHA-256 comparison: 169 DSC files checked, exactly the six expected files changed; locked domain/controller/motion/parent files unchanged. 97 Mini Tracker runtime baseline files unchanged. Existing unrelated modifications preserved.
- Browser: actual Google Maps JS weekly/native renderer, terrain and volume rendered in Codex IAB, using local explicit DEMO harness at localhost:8766 with actual production FieldMap, bridge, mission3d, motion and awareness core/controller. Node fakes were not used in this browser path. External Edge was not exposed by available tools. Actual authenticated DSC+ page on port 8766 showed Non connesso; prior server 8765 was unreachable. Private/live authenticated path not claimed.
- Verified actual Google UAS line clamp-to-ground exact measured path, area nearest-point path and independent outline; NORMAL/MONITOR/CAUTION/WARNING, approaching/diverging, UNKNOWN missing observation time, STALE UAS last-valid distance 1.1 km with traffic fresh/reference stale and gray #87909a, zero area outline without ground line, explicit rotorcraft fixture, labels absent/ON/removed again, object selection/details, camera orbit/zoom, compact/normal, pause/resume/restart/loop, close/2D/reopen, dark/light and 390 px card without horizontal overflow.
- Camera DOM attributes remained heading 85, range 1575 and center 42.33163,12.604439999999954 across about 40 seconds of original scenario updates. Paused marker positions matched across later reads. Loop advanced runs [1,14] -> [1,15] -> [1,16]; reopening received current run/time rather than restarting. Representative live count: one map, one ground line, three polygons (footprint/volume/emphasis), two interactive targets. Expired relationships removed line/outline.
- Performance: no sustained rendering stop after label fix during multi-minute interaction/loops. Parent harness displayed 2D update p95 about 2.5 ms at final run; this is NOT a 3D renderer timing/GPU benchmark. Browser/background frame gaps varied up to about 1 s; no FPS or dense-load guarantee. Tests verify awareness nodes unchanged across 120 interpolation frames and 10 accelerated source loops remain bounded.

## Limitations and next work

Actual private Mini Tracker area/account path, external Edge, physical touch/pen, dense load and hardware were not revalidated. PROVISIONAL and mixed DEMO/LIVE provenance are covered by unit tests, not real private-source browser data. Current inherited renderer only displays DEMO target positions; this slice does not add live traffic transport. Generic native pins are intentional; no GLB or scenario/classification invention. Browser visual acceptance remains Product Owner-owned. Ready for review before 5C-4 Maker Faire Helicopter Scenario; 5C-5 optional models remains a later independent slice. No deploy, commit, push, MkDocs build, dependency, credential change or Mini Tracker runtime change.

Full report: C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_GEOAWARENESS_5C3_IMPLEMENTATION_20260922.md.

