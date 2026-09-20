# MF-DEMO-2 — Field Operations 2D

MF-DEMO-1 is PRODUCT OWNER ACCEPTED. This slice is authorized for implementation; no additional review gate. Preserve the accepted card design.

## Requirements and acceptance
- Existing DSC map only: one owned Field Operations group, one compact Leaflet control.
- Default scene: real public dsc-node02 tracker at published coordinates, MANUAL; fixed Vescovio DEMO Polygon, RID and aircraft. No new Internet traffic integration in this slice.
- Each object carries origin (LIVE/DEMO) and optional source (INTERNET/LOCAL_RX/DEMO). A mixed scene keeps the original tracker origin; no global DEMO relabelling. Unknown provenance is never inferred to be LIVE.
- Stable IDs and full-array reconciliation; remove deleted objects, deduplicate updates, render meaningful popup details using text nodes.
- Recenter frames only owned visible layers, once at opening or by explicit button. Heartbeat/timer updates never move the camera.
- Close, source switch, logout and context change remove owned layers/control and release retained source/timer when Workspace is closed. Other DSC layers/controls and planner remain untouched.
- Polygon uses GeoJSON [longitude, latitude] rings (including holes). Circle uses normalized {type:"Circle", center:[longitude,latitude], radiusM}; add a tested adapter for Mini Tracker's GeoJSON Point with leafletType=Circle/radius. Invalid geometry is skipped with a visible notice, never silently rendered at 0/0.
- No 3D placeholder button. Preserve offline Mini Tracker functions by making no runtime changes.

## Design
Extend the current scene and fixture source. Add tracker.origin; operation is {id,name}; areas use {id,name,geometry,origin,source}; targets use {id,name,type,position,origin,source,altitude?}. Altitude is optional {value,unit,reference}; no conversions or invented datums.

field-map.js owns tracker, Polygon/Circle, RID/aircraft, compact counts/source legend, recenter and close. It consumes scene arrays without fixture knowledge. The existing card/controller continues owning authentication, source selection, freshness and listener/timer lifetime; it delegates map ownership entirely to field-map. The selected public-presence source is explicitly composed with DEMO objects in fixture-source, before entering the existing store. MF-DEMO-3 can instead provide real normalized areas directly without changing this renderer.

Fixtures use fixed Vescovio coordinates, never relocate actual RF or follow a real tracker to another site. LIVE tracker position/time remain unchanged. The renderer has no timers, subscriptions, network calls or publications. Reconcile changed IDs only; retain unchanged geometry.

## Verification
Focused tests: one tracker/area/RID/aircraft; Polygon and Circle adapter; LIVE/INTERNET/DEMO/LOCAL_RX labels; mixed provenance; independent bounds; no duplicate layers/controls; update/deletion/invalid geometry; close/logout/context/source/heartbeat lifecycle; no automatic camera movement; unrelated map/planner preserved; no fixture network publication. Run existing Field Nodes/Workspace/map tests and syntax/whitespace checks. Browser review is separate, using the existing isolated local harness and real Leaflet.

Excluded: MF-DEMO-3 sender/token/private storage, live area sync, network/RF target transport, interpolation, 3D, replay/recording, Mission V3/Flight Plan, commands, provisioning, Mini Tracker installation, MkDocs, commit/push/deploy.

## Completion
Implemented, ready for Product Owner visual review. 109 tests passed, including 30 focused Field Nodes/map tests; changed/new JS syntax and scoped diff whitespace passed. Browser checked mixed real public heartbeat + DEMO content, real Leaflet Polygon/RID/aircraft, desktop dark/light, pan, recenter and close preserving the base map in the isolated harness. Authentication/Workspace wrapper remained simulated. No deployment or physical-device validation.

MF-DEMO-3 can replace scene.areas with real normalized Polygon/Circle objects without changing field-map architecture. Source-side normalization/private transport remain deliberately unimplemented.
