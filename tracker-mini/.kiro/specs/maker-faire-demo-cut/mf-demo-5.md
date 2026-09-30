# MF-DEMO-5 — Animated targets

## Authoritative update — 21 September 2026

Latest PO result after website deployment: integrated Google 3D works perfectly in external Edge, is fluid and has no blocking. Positive product browser test reported by PO; exact duration/loop count and full control checklist unspecified, so do not claim the preferred ten-cycle stress passed. The earlier agent-side WebGL2 limitation described below applies to the internal browser.

MF5A 2D is Product Owner accepted. MF5B Cesium animation is abandoned for Maker Faire; its original defect remains unresolved and must not receive more debugging time. The Product Owner has approved Google Maps 3D for Field Operations only. See [approved migration spec](google3d-field-operations.md). Google now consumes the same normalized scene and unchanged shared motion/interpolation source. Preview3D and Airspace3D remain unchanged.

194 DSC regression/contract tests passed, including the Google renderer, native polygons/holes/Circle/volume, stable target identity/interpolation/heading, entry/removal, source controls, camera independence, private handoff and close/logout. Actual in-app product loading reaches Google, then reports WebGL2 unavailable before drawing. After PO login, authentic handoff of the two retained STALE Vescovio areas and LIVE/MANUAL tracker was verified. The PO's initial isolated spike passed visually; integrated Google visual/stress validation still requires functioning WebGL. Do not treat accelerated fake loops or successful data handoff as GPU proof.

The original specification and investigation below are retained as history. Their Cesium renderer, terrain fallback and old revalidation instructions are superseded by the approved Google spec.

## Original MF5 specification (historical)

MF1/2 accepted; MF3 physical acceptance passed; MF4 visual acceptance passed in the real DSC application, as reported by the Product Owner. Preserve the accepted 2D/3D presentation and all private/public transport.

## Requirements and design
- One deterministic 60-second Vescovio scenario, client-side only. RID starts at 3 s, crosses a bounded route anchored to the current first valid area (or tracker), turns around 35 s and returns by 55 s. Aircraft enters at 15 s, crosses higher and exits at 45 s. Route anchor is frozen for a run; real edits remain visible without restarting motion.
- A small scene-motion source controller publishes normalized logical targets at 1 Hz into the existing scene store. Preserve the latest real tracker, operation, areas and team on every composition. Only the two existing DEMO target IDs are replaced. No alternate scene store, cloud writes or frame-rate source samples.
- Logical motion metadata includes elapsed time, state, sample time and run identity. MF6 can subscribe to the existing store and capture elapsedMs + scene before interpolation. Use monotonic elapsed timing, explicit READY/PLAYING/PAUSED/COMPLETE and Start/Pause/Resume/Restart; inexpensive optional Loop.
- Shared bounded target interpolation helper used by both renderers. Stable Leaflet marker updates and Cesium entity position/orientation updates, no viewer/volume recreation or implicit camera movement. A newly opened 3D view takes the current logical state and acquires the next interpolation segment within one source interval.
- Pause freezes visual movement, resume continues elapsed time, restart/loop invalidates old interpolation. Fresh samples interpolate for at most one interval; no extrapolation; stale after 3 s and expired after 10 s without samples. Explicit paused/completed demo remains frozen. No source-observed timestamp is synthesized by the interpolation helper.
- Tracker/area provenance remains LIVE where provided; volume remains DEMO 120 m AGL, RID DEMO 35 m AGL and aircraft source 2400 ft MSL with separate visual terrain +600 m. Orientation follows route; angular interpolation takes the shortest turn.
- Minimal motion controls in 2D and 3D, with exact-origin/source/version-checked actions forwarded to the parent source. 3D close leaves 2D motion running; Field Operations close/logout/context/source changes/denial stop motion and clear timers/private state. Reopen starts cleanly.

## Failure and compatibility
Malformed coordinates rejected; unavailable area uses the configured tracker anchor, never alters real area geometry. Terrain failures retain MF4 explicit fallback. Out-of-order asynchronous terrain work cannot overwrite current motion. Background throttling resumes from elapsed scenario time without a backlog. Incoming heartbeat/private area updates cannot reset synthetic targets. Existing source contract/endpoints, receivers, Mini Tracker runtime and existing viewers stay untouched.

## Tasks / verification
- [x] Source/controller and shared interpolation; logical cadence and recorder seam.
- [x] Stable 2D marker updates, controls, cleanup.
- [x] Stable 3D animation/orientation, controls/handoff, terrain and camera preservation implemented; actual complete WebGL validation pending below.
- [x] Focused tests: deterministic timestamps/coordinates/headings, pause/resume/restart/loop, freshness/expiration, stable IDs/entities/layers, mixed provenance, real area edits, lifecycle and regressions. 174 tests passed, 11 syntax checks passed.
- [ ] Complete actual browser/WebGL review: 2D motion/pause/resume/popup/restart/loop/zoom verified. Early 3D showed moving targets and TERRAIN OK with orbit, then render/context failure prevented the full cycle. PO reports the same class of WebGL failure with the accepted static view. Recovery/restart and revalidation required; do not claim the fault resolved or MF5 ready for acceptance yet.
- [x] Demo Cut, handoff and delivery report with pending WebGL gate and PO 18-step review.

Deferred: trails/follow camera, recorder/export/import/replay, real RF transport, POI, Mission V3, Flight Plans, messaging, provisioning. No deploy/commit/push/install/MkDocs.

## 21 September crash investigation
PO confirms 2D works correctly; 3D crashes after the first drone movement. Root cause remains unresolved. Fixed a separate render-error lifecycle race: pending terrain completion cannot restart a failed renderer, queued animation is cancelled and subsequent updates/camera actions are rejected. Added regression; complete MF5 suite now 175/175 passed, changed JS syntax and whitespace checks passed. Actual Cesium 1.114 CPU checks found finite target transforms/straight guides over 5,551 target frames, with synthetic ground; not WebGL validation.

An independent minimal canvas probe confirmed both WebGL 1 and 2 unavailable with GL_RENDERER = Disabled, before loading Cesium. Local bounded error diagnostics remain ready for the next actual rendering run. Full visual gate remains unchecked until the original RangeError is reproduced, corrected and validated with functioning WebGL. See AI_HANDOFF.md for exact evidence and remaining work.
