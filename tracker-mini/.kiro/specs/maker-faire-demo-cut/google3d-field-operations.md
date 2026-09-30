# Field Operations Google 3D — approved renderer migration

Date: 2026-09-21. Owner approval: supplied Product Owner Decision, Google 3D for Field Operations only.

Latest validation: PO deployed to the website and reports the integrated Google 3D view works perfectly in external Edge, with fluid motion and no blocking. This is positive real-browser evidence reported by PO. Duration, exact loop count, browser version/GPU and individual control checklist were not supplied; the preferred 10 x 60-second stress is not yet recorded as passed. The agent's internal-browser WebGL2 unavailability is a separate environment limitation. No renderer changes were needed for this update.

## Requirements and acceptance

Google Maps 3D replaces the renderer under DSC `public/mission3d/` in the existing private 2D → iframe flow. No renderer selector or fallback. Preserve normalized tracker/operation/areas/targets/team/motion samples and the existing MF5 source. No Mini Tracker transport changes. Preview3D and Airspace3D remain unchanged.

Tracker is ground-clamped with truthful LIVE/MANUAL provenance. Real operational footprints retain LIVE/MINI_TRACKER labels. Synthetic volumes are DEMO 120 m AGL. RID uses DEMO 35 m AGL; aircraft source 2400 ft MSL stays metadata while visual placement is explicitly DEMO +600 m above ground. Preserve polygon holes; omit innerPaths entirely without holes, including Circle adaptation and extrusion. Never mutate canonical geometry.

## Design and affected components

Promote native Map3DElement/Polygon3DElement/Marker3DElement/Polyline3DElement adapter logic from the isolated Google spike into mission3d. Keep the existing model validation, sidebar and safe text handling. Use one map per iframe, keyed static objects and stable target marker/heading arrow pairs. Shared createTargetInterpolator remains presentation-only; source controls continue through the versioned same-origin bridge. Camera changes only on initial placement or explicit user interaction.

Replace mission3d-config.js with a bounded asynchronous loader reading same-origin, Git-ignored google-maps-config.json (browser-restricted Maps JavaScript API key; optional mapId). No credentials or scene in URLs between application pages or persistent browser storage. Google loader necessarily receives its browser API key; never include it in diagnostics. Missing configuration/network/render failures show an explicit unavailable state with 2D remaining usable. Closing/logout during loading must prevent late map creation; closing cancels animation and listeners and removes private scene/UI.

## Verification

Update mission3d tests from the obsolete engine to Google fakes for geometry, provenance, stable identity, interpolation/heading, removal/reentry, camera independence, teardown and async loader races. Preserve bridge/security tests; run MF1–MF5 regressions and spike tests. Mocks prove contracts, not GPU stability. Attempt real Chrome/Edge with real authenticated area and preferred 10 complete 60-second loops. Record unavailable prerequisites honestly; PO reports the spike's initial visual run as fluid and correct.

## Tasks

- Implement focused model, renderer, loader, lifecycle and HTML/CSS migration.
- Add contract/regression tests and verify protected files against baseline hashes.
- Validate in browser where configuration and browser access permit.
- Update Demo Cut and AI_HANDOFF with results and remaining visual checks.

## Compatibility and deferred work

No recorder/replay implementation: future MF6 can consume the same normalized samples independently of renderer. Airspace3D is a future Google migration candidate only. Preview3D stays Cesium until a product reason exists. No common framework, RF transport, POI sync, Mission V3, Flight Plan, provisioning or messaging work.

Historical Cesium MF5B is abandoned for Maker Faire with unresolved RangeError Invalid array length / updateFrustums / createPotentiallyVisibleSet. Preserve reports; do not resume investigation.
