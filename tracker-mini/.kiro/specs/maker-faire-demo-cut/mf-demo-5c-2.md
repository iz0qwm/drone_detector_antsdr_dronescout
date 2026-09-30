# MF-DEMO-5C-2 — Traffic Awareness 2D

Implementation authorized by the Product Owner on 2026-09-22. Authority: the 5C-2 implementation request and [5C-1 implementation report](<C:/PROVA/DRONI/CORSI-PROGETTI/DSC+/DSC_PLUS_GEOAWARENESS_5C1_IMPLEMENTATION_20260921.md>).

## Requirements and acceptance

Consume only `scene.awareness` for domain state, distance, trend, freshness, ranking and provenance. Add a compact, accessible section inside existing Field Operations; maximum five keyed rows. Retain all states, including subdued NORMAL, in core order. Preserve explicit selection while its relation survives; if it leaves the first five, show the first four plus that relation, still in core order. Report omitted count. Default to the first displayable relation.

Render one selected measured relationship using target/reference endpoints, or target/nearest-area endpoints. Zero-distance INSIDE/BOUNDARY uses an independent area outline instead of a zero-length line. Never use interpolated marker coordinates for distance or line endpoints. Use text and pattern as well as severity colour; explain historical/provisional/unknown results and mixed provenance. Helicopter icon only for explicit ROTORCRAFT; retain existing drone/aircraft icons and production scenario.

## Design and affected components

Keep presentation helpers and the owned awareness group within `field-map.js`; extend `field-operations.css` for themes, focus and bounded responsive layout. Reuse existing area rendering adapter only for outline drawing, not domain geometry calculation. Update on awareness revision/context or user selection; no awareness work in RAF. Existing map controls, camera, source ownership, motion and 3D remain unchanged.

## Failure modes and compatibility

Missing/empty/expired snapshots show explicit availability text, never a safety assertion. Missing endpoints draw no invented line. Missing area geometry cannot produce an invented outline. Reset selection and layers on run/operation/mode changes, close and authorization-driven teardown. Stable IDs prevent duplicate rows/layers; all external text uses textContent.

## Verification tasks

- Cover every state/trend/distance kind, references including hole/interior, explicit provenance, five-row limit and revision-only updates.
- Cover selection persistence/removal/new run, stable owned layers, no mutation, icons/headings, keyboard semantics and safe text.
- Run the 13 requested regression suites and syntax/whitespace checks.
- Review real Leaflet in the local isolated browser harness, including dark/light, narrow layout, all requested states, animation and lifecycle.
- Record automated evidence separately from visual browser evidence in the requested implementation report; update handoff. No deploy/commit/push, Mini Tracker runtime or 3D UI changes.

## Implemented — 22 September 2026

Status: MF-DEMO-5C-2 — TRAFFIC AWARENESS 2D IMPLEMENTED. Completed in DSC field-map.js and field-operations.css, with focused tests/helper updates. No core/controller/scenario/3D changes. Exact presentation policy above retained.

Verification: 45 focused tests passed (12 existing + 33 added), 0 failed, 0 skipped; full 13-suite regression 283 passed, 0 failed, 0 skipped (includes focused tests). Includes direct rendering of an immutable snapshot from the actual core. Three JavaScript syntax checks and four-file whitespace checks passed. SHA-256 baseline: only four changed DSC files among 141; 97 checked Mini Tracker runtime files unchanged.

Actual local Leaflet browser review completed separately: NORMAL/MONITOR/CAUTION/WARNING, approaching/diverging, UAS/area line, zero area outline, stale/unknown, helicopter fixture, keyboard selection, dark/light, desktop and 390 px viewport, existing deterministic animation with actual controller, pause/resume/restart/loop, close/reopen. No authenticated live-session, new Edge, hardware or 3D acceptance claimed. One isolated MutationObserver error without source URL remained in cumulative browser logs; no recurrence or observed render stoppage in subsequent review. Dense maximum-core performance is still the documented 5C-1 limitation.

Report: `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_GEOAWARENESS_5C2_IMPLEMENTATION_20260922.md`. 5C-3 can consume unchanged scene.awareness without recalculation. Deferred: 3D awareness/models, helicopter scenario, real transports, record/replay, legacy migration. No deploy, commit, push or MkDocs build.

## Product Owner UI polish — authorized 22 September 2026

Scope: presentation-only target label toggle, movable panel header and normal/compact modes. Default target labels OFF; setting remains in the current view instance across updates/close/reopen, without browser storage. Tracker/area labels and all icons/popups/awareness stay unchanged. Compact mode preserves a draggable FIELD OPERATIONS header, restore and close buttons while updates continue.

Implementation plan: reuse field-map.js and CSS. Scope tooltip visibility to owned RID/AIRCRAFT tooltips. Use pointer capture only on an explicit header handle; suspend map dragging only while that pointer is active and restore its previous enabled state on every termination path. Clamp panel to the map on drag/resize/content size changes. At map widths <=700 px use anchored bottom-left placement without dragging. Reset placement and collapse on close; dispose resize/pointer listeners. No core, controller, scenario, bridge, 3D, storage or runtime changes.

Verification plan: label default/toggle/classes/new arrivals/update persistence; handle-only drag, pointer cancellation, all bounds, resize/narrow placement and cleanup; collapse without teardown and continued animation/revisions. Run focused map tests plus requested eight regression suites and related Field Operations regressions. Actual browser review separately covers labels, icon/popup retention, drag corners/map isolation, compact animation, restore, themes and narrow layout.

## UI polish implementation and verification — 22 September 2026

**FIELD OPERATIONS 2D UI POLISH — IMPLEMENTED**

### Delivered behaviour

- Target labels default OFF for each new view instance. The checkbox `Etichette target` toggles owned RID/UAS and AIRCRAFT labels, including explicit ROTORCRAFT, FIXED_WING and UNKNOWN. Icons, popups, halo, measured line and awareness rows remain. Tracker and area labels are unaffected. Leaflet repositions tooltips after showing them so their previously hidden zero dimensions cannot leave them overlapping icons.
- The label choice survives source/awareness updates and close/reopen of that view instance. No localStorage/sessionStorage, coordinates, scene or operational data are persisted.
- The FIELD OPERATIONS handle accepts a primary pointer/mouse drag; ordinary controls never start a drag. The control is positioned inside the map with a 10 px inset and clamped against actual map dimensions. Pointer capture handles leaving the header; pointerup/cancel/lost capture/window blur/close restore the prior map-dragging state. ResizeObserver and Leaflet resize re-clamp after container or panel size changes.
- At map width <=700 px, dragging is disabled and previous manual placement is discarded; deterministic bottom-left placement returns. Panel width/height are constrained to the map. Desktop positioning and compact state reset on close/reopen.
- Normal/compact are explicit modes. Compact hides only the body; the draggable FIELD OPERATIONS title, Ripristina and Chiudi stay visible. Expanded state is exposed through aria-expanded. The same renderer, animation and awareness continue; restoring does not create another controller, group or panel. No manual resize feature or compact severity badge was added.

### Files

DSC changes: `public/js/field-operations/field-map.js`, `public/css/field-operations.css`, `functions/test/field-operations-map.test.js`, `functions/test/helpers/field-map-fakes.js`, plus one necessary test selector update in `functions/test/field-nodes-ui.test.js` because close is now also in the header and the body is wrapped. Documentation: this Feature Spec and `AI_HANDOFF.md`. The accepted 5C-2 report remains a historical record; this addendum records the later UX refinement.

### Automated verification

From `C:\Users\raffa\DroneSkyCheck\functions`:

```text
node --test test/field-operations-map.test.js
```

60 passed, 0 failed, 0 skipped (45 previous + 15 polish tests). Coverage: label defaults/all classes/toggle/new targets and update persistence, retained icon/popup/overlays, source preservation, allowed handle only, primary/wrong/cancelled pointers, original map-dragging state, x/y movement, four bounds, resize, narrow anchoring, close during drag, handler cleanup/reopen, compact visibility, ongoing animation/revisions, restoration/clamping and compact close.

Full related regression (includes the focused tests):

```text
node --test test/geoawareness-profile.test.js test/geoawareness-geometry.test.js test/geoawareness-core.test.js test/geoawareness-controller.test.js test/field-operations-map.test.js test/field-nodes-ui.test.js test/scene-motion.test.js test/mission3d.test.js test/field-operation-service.test.js test/pilot-workspace-refactor.test.js test/operational-place-ui.test.js test/mission-workspace-ui.test.js test/access-resolver.test.js
```

298 passed, 0 failed, 0 skipped. All eight requested suites plus five relevant existing suites. Four changed JavaScript syntax checks passed; five DSC files checked without trailing whitespace. Baseline comparison: five expected DSC changes among 141 inspected files; core/profile/geometry/controller/motion/scenario/3D unchanged. 97 inspected Mini Tracker runtime files unchanged. Scoped documentation diff check passed.

### Actual browser evidence

Existing isolated review path: `http://localhost:8766/__traffic-awareness-review`, real Leaflet in the integrated browser. Desktop 1280 px and actual iframe viewport 390 px; temporary viewport override reset after review.

Observed label OFF at startup with aircraft/drone icons, line, halo and rows retained; ON restores labels above icons; OFF still permits aircraft popup. Rotorcraft fixture label also toggled. Dragged to all four bounds: normal panel 320×672 px in 1280×772 map; left positions clamped to 10/950 px, top to 10/90 px. Camera counter remained 2 across corner drags, then increased only after deliberate map pan/zoom. Actual map interaction resumed after drag.

Collapsed to header bar; animation in the unchanged production route continued, snapshot revision advanced from 1 to 29 and measured area relation appeared while compact. Restore displayed current rows. Light/dark reviewed; changing from dragged desktop to 390 px returned the bar to bottom-left, then restored a readable panel with no horizontal overflow. Closed from compact header: zero panels/lines/contours; reopen normal and clean. Mouse dragging was exercised in browser; physical touchscreen/pen was not tested. No authenticated/hardware/3D acceptance claim.

A single MutationObserver exception without source URL was present in cumulative browser tooling logs during this review; the product changes do not instantiate a MutationObserver. No observed panel/map failure or repeated error series. Its origin is not established; do not equate the Node suite with browser acceptance or claim a clean browser console.

### Limits / 5C-3

Narrow mode is deliberately anchored. No drag keyboard placement, manual resize or preference persistence. Existing long awareness lists still scroll inside the normal panel. Pointer/touch path uses browser pointer capture, but no physical touch-device validation was performed. Impact on 5C-3: NONE. No scene.awareness contract/domain change, Google 3D modification, new dependencies, deployment, commit, push, Mini Tracker runtime changes or MkDocs build.
