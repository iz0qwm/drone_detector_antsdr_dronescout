# MF-DEMO-3 — Real saved operational areas

MF-DEMO-1 and MF-DEMO-2 are PRODUCT OWNER ACCEPTED. Implementation is explicitly authorized. Preserve their UI and renderer ownership.

## Acceptance and scope
Read the selected local mission and saved user Polygon/Rectangle/Circle layers in-process. Flatten small edited FeatureCollections with stable serial:mission:layer:index IDs. Exclude imports, DSC regulatory zones and POIs. Preserve longitude/latitude, radius, name and optional hex color; no altitude. Full replacement arrays support create, rename, geometry edit, deletion and mission switch. No Mission V3 records, RF targets, 3D, replay, provisioning or changes to public feeds.

## Design and failure modes
- New services/field_scene.py uses existing mission/layer readers; selection is checked before/after collection. Missing selected mission/layer storage, malformed JSON, invalid operational geometry and exceeded bounds produce an ERROR snapshot, never an empty successful scene. Successful no-selection or zero operational areas publishes an empty projection.
- New services/field_sender.py is disabled without explicit private configuration outside the installation tree. Existing configured node identity must be dsc-node02, associated with MTRK26-0001. A collector every 5 seconds and separate single HTTP worker share one replaceable latest snapshot. Requests have bounded connect/read timeouts, redirects disabled, bounded backoff (5–30 seconds), no history, disk queue or receiver/UI dependency. Restart recollects disk state.
- One HTTPS ingestion function validates a static Secret Manager token, node/serial, strict allowlists, size <=128 KiB, <=100 areas, <=20 rings/2000 vertices per polygon and <=100 km radius. Sample time must be recent and transactionally newer than stored state; delayed/duplicate data cannot overwrite newer state.
- Private fieldOperationsLatest/MTRK26-0001 only. Direct client access denied. Geometry projection is stored as JSON text because Firestore does not accept nested coordinate arrays. Hashing avoids rewriting unchanged geometry; freshness metadata still updates. ERROR snapshots preserve last good projection/time. Missing/old/error state is explicit.
- Authenticated read callable reuses resolveAccountAccess(requiredFeature=workspaceSync), checks ACTIVE DSC_PLUS and the two Product Owner UIDs on every read. No token or other mission fields returned/stored. Read path decodes canonical geometry.
- Existing controller adds a private read every 5 seconds, at most one in flight, with generation guards. Last good areas survive temporary read failure; permission denial clears private state. Public presence and DEMO targets are independently composed with real operation/areas; explicit DEMO mode still uses the existing fixture. Scene operationalStatus carries component freshness; panel notice reports stale/error after 15 seconds. Context/logout/source/close release resources.
- The existing map needs only optional hex color support. Canonical scene geometry remains suitable for the later 3D adapter.

## Verification strategy
Use isolated Python unittest tests compatible with pytest: storage/normalization/exclusion, lifecycle replacement, error versus empty, blocked upload versus independent collection, latest-only retry and startup config. Node tests cover transport token/identity/bounds/order/private storage/access, frontend composition/error/deletion, approved renderer and MF1/2 regressions. Run syntax/whitespace checks. No hardware or deployed authorization claims from mocks. Document exact configuration, release prerequisites and the 17-step physical acceptance flow; no deploy, commit/push, device installation or MkDocs.

## Tasks
- [x] Collector and isolated sender/startup
- [x] Private ingestion, latest storage and authorized reader
- [x] Existing scene consumer integration
- [x] Focused and regression tests
- [x] Delivery report, manual/configuration and handoff

## Implementation evidence
22 Mini Tracker unittest checks passed (also pytest-compatible); 133 DSC Node checks passed, including existing MF1/2/access-resolver tests; 4 real Firestore emulator checks passed with demo-dsc-field-operations. The emulator verified actual geometry persistence, private get/list/write denial for all client classes, public tracker read and membership revocation. Java required an approved sandbox escalation; all emulator state stayed local. No physical/deployed validation.

System Update inspection found that verification imports app. The new sender starts only under the normal script entry point, not during that import check; a focused startup test covers both paths. Existing service startup behavior is unchanged. Bundled development Python lacks requests/pytest, so tests use stdlib unittest and stub only the unrelated zone-download import for actual temporary mission/layer storage tests. Production already depends on requests.
