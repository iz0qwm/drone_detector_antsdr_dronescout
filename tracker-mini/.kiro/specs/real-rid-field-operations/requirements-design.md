# Real Remote ID in private Field Operations LIVE

## Requirements and acceptance

- Publish only real DS110 Remote ID detections with a stable serial and valid position through the existing private field snapshot. A target outside saved mission areas remains eligible.
- Keep the public DSC traffic bridge, local Remote ID API, DEMO and REPLAY behavior intact.
- Use a timestamp captured when a valid Location message is received. Basic ID, System and HTTP reads must not refresh the position timestamp.
- Bound target count, age, identifiers and payload size. An absent or expired target must leave the next complete private snapshot; loss of the private link must not make an old position fresh.
- Preserve geometric altitude as WGS84 ellipsoid altitude when valid. A zero height is not interpreted as ground level; the current decoder does not extract the encoded height reference.
- The same private target ID and observed time must pass through the DSC LIVE scene, Geoawareness, 2D, Google 3D, recorder and validated replay.

## Design and components

1. `backend/services/ds110.py`: retain public `last_seen` behavior and add `position_observed_at` only at valid Location or DJI position update seams. Keep receiver configuration and startup unchanged.
2. `backend/services/field_scene.py`: read a detached DS110 state snapshot, normalize eligible RID objects into the existing target shape, and add bounded `targets` to a complete private snapshot. No area-containment test. The collector remains usable without hardware.
3. `functions/fieldOperations/field-operation-service.js`: validate optional `targets` on the existing authenticated endpoint and include them in the private JSON projection. Reject invalid, synthetic or oversized data.
4. `public/js/field-operations/operational-source.js`: merge private RID targets into the authoritative LIVE scene, expire them using position observation time, and retain future non-RID LIVE sources. No public traffic query is added.
5. Feed the received RID into the existing Geoawareness target selection for operational-area relations and UAS reference selection for aircraft, omitting self-pairs. Reuse the existing horizontal calculations and thresholds. Leaflet, Google 3D and recorder consume the same scene target. Extend the existing altitude reference allowlist to preserve WGS84 ellipsoid provenance in details and replay. Google 3D uses its unresolved-altitude ground marker fallback.

## Failure and compatibility

- Old area-only snapshots remain valid and yield zero private targets.
- A malformed target rejects the entire new snapshot; the cloud retains the last good projection with its original observation times. Client expiry still removes old traffic from the LIVE scene.
- Private connectivity and hardware absence do not block the local map, public RID path or sender startup. The sender keeps its one-latest bounded retry behavior.
- Existing area schema, IDs, thresholds, 3D camera, recording limits and mode controls are unchanged.

## Verification tasks

- Unit tests: valid/invalid Location timestamps, generic packet non-refresh, stable ID, altitude sentinel/zero, outside-area inclusion, bounds, expiry and recovery.
- Contract tests: cloud auth/schema/projection, LIVE composition, Geoawareness, 2D and 3D object identity, recorder/replay.
- Run focused Mini Tracker and DSC tests, syntax checks and diff checks. Physical DS110 reception, deployed cloud flow and RF loss/recovery require a separate device test after deployment.
