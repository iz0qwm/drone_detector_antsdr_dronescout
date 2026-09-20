# MF-DEMO-4 — Field Operations 3D

MF-DEMO-1/2 are PRODUCT OWNER ACCEPTED. MF-DEMO-3 is PRODUCT OWNER PHYSICAL ACCEPTANCE PASSED, as reported by the Product Owner: real Polygon create/update/delete, Circle, private synchronization, local offline continuity, retained remote stale geometry and latest-state recovery. Point/POI exclusion (Punto di decollo) is intentional and remains post-MF4.

## Requirements and acceptance
- Dedicated public/mission3d viewer, opened by Vista 3D in the accepted 2D panel. Same normalized in-memory scene; no private data in URL/storage or independent cloud/Firestore reads.
- Same-origin iframe with a small versioned message envelope; both sides validate origin/source, late messages are invalidated, close/logout/context/source changes clear private state and destroy Cesium. Returning to 2D preserves its layers/camera/source ownership.
- Ground-positioned tracker, Polygon/holes and Circle footprints, preserved IDs/name/color/source. Only the adapter discretizes circles. Semi-transparent volume with visible roof/upper perimeter is explicitly VOLUME DEMO, 120 m AGL; no operational altitude or airspace claim.
- DEMO RID source 35 m AGL is positioned over sampled terrain. DEMO aircraft source 2400 ft MSL is retained in details; display height is an explicitly approximate/demo placement, no certified separation or MSL/ellipsoid equivalence. Unknown/live altitude semantics must never use the DEMO conversion silently.
- Oblique initial framing includes all valid objects; explicit Ricentra repeats framing. Incoming scene changes never recenter an already framed camera. Target entities are updated in-place by stable ID, with no motion/interpolation/trails/replay.
- Real terrain is the primary acceptance requirement. Visible TERRAIN OK only after provider and actual finite terrain samples succeed; visible TERRAIN FALLBACK on provider/sampling failure, with a working ellipsoid scene. Test the actual browser/WebGL.
- Clean, readable overlay in DSC dark/light themes. Compact permanent labels; source and altitude details remain available via selection. No existing 2D redesign.

## Technical design
Reuse Preview3D's Cesium 1.114 initialization, World Terrain asset 1/public client configuration, sampling and Cartesian ground-position concepts; reuse Airspace3D's target icon assets/orientation and stable entity-map concepts. Existing viewer files are unchanged. No shared Cesium framework/refactor and no new dependency versions.

Add a parent bridge module owning modal/focus/listeners/iframe. Existing controller owns subscriptions and pushes current scene plus display-only notice/theme. Add a guarded CTA callback to field-map; preserve all accepted map ownership.

Inside mission3d, a pure adapter validates/bounds geometry and computes Circle rings, DEMO altitude semantics and framing inputs. A Cesium renderer samples terrain with bounded waits and cached positions, creates footprint/roof/walls/outlines, and reconciles entity groups. Main iframe script owns UI, message validation, readiness, asynchronous generation guards and disposal. No credentials/private geometry in logs. Parent close removes the iframe even if child teardown fails; pending asynchronous work cannot repopulate a disposed viewer.

Failure modes: unavailable Cesium/WebGL visible error; terrain unavailable explicit fallback; malformed shape skipped/reported; missing scene waits rather than auto-generating a LIVE fixture; invalid source messages ignored; removed/private-denied areas disappear; stale data remains labelled with parent status. Slow terrain work coalesces updates and does not let old geometry resurrect.

## Tasks and verification
- [ ] Parent handoff/CTA and lifecycle
- [ ] Dedicated viewer/adapter and terrain/volume/targets
- [ ] Focused tests: secure handoff, Polygon/Circle/holes, immutable radius, ground placement, synthetic height, altitude labels/provenance, framing/recenter, stable updates/deletion, cleanup/invalidation and malformed data
- [ ] Existing MF1/2/3 regression checks; confirm Preview3D/Airspace3D and private transport unchanged
- [ ] Actual desktop WebGL: terrain, tracker, polygon, Circle, volume, RID/aircraft; orbit/zoom/recenter, dark/light, close/reopen/2D return
- [ ] Demo Cut, AI_HANDOFF and delivery report with actual terrain status and 18-step PO review

No Mini Tracker runtime/transport modification, deploy, commit/push, physical installation or MkDocs. No target animation, recorder/replay, live RF, generic POI, Mission V3, Flight Plan or messaging.
