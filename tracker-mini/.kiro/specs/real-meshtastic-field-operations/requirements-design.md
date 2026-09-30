# Real Meshtastic Team operators in DSC+ Field Operations

## Requirements and acceptance

- Reuse the existing Mission Teams association: exact, case-sensitive `shortName` match; retain configured `longName` for display. No local Teams workflow or radio setting changes.
- Publish only resolved `operators[]` with a valid position observation through the existing private Field Snapshot. Never publish `external_nodes[]`, unassociated nodes, or gateway data.
- Use `scene.team`, stable `mesh:<nodeId>` identity, a separate operator count and person markers in 2D and Google 3D. The global label switch controls the name in both views.
- A new position packet updates the location time; text packets and NodeDB polling do not. After ten minutes mark stale, after 30 minutes remove. Missing or invalid coordinates do not create markers.
- Preserve RID, ADS-B, DEMO, air-traffic Geoawareness and area behavior. Team members remain outside `scene.targets` and may be outside saved polygons.
- The LIVE recorder captures team state; V1 replay scene schema 1.1 permits bounded team members while prior 1.0 recordings remain loadable and retain recorded freshness.

## Design and affected components

`meshtastic_service.py` stores position-specific coordinates and observation time in its existing node cache when `POSITION_APP` arrives. `teams.py` passes these fields through its existing resolved operator status. `field_scene.py` projects only that status into a bounded private team DTO. DSC's existing write validator and authenticated reader validate and transport it. The LIVE scene composer, 2D map and Google 3D bridge consume that same array. The recorder allowlist and replay validator accept team in schema 1.1; old team-empty recordings stay 1.0.

## Failure and compatibility behavior

Unresolved, duplicate-association, gateway, external and positionless nodes are omitted. The stable node ID is used only after the current Teams match. Radio loss does not renew `position_observed_at`; stale members remain visible until retention ends. On process restart there is no persisted position timestamp and a new location packet is required. Altitude remains `UNKNOWN` datum and is detail only. Existing transport fields remain accepted, and no new dependency or command channel is introduced.

## Verification tasks

1. Test configured/unconfigured and removed operator projection, ambiguity, position omission, stale expiry and unchanged RID/ADS-B.
2. Test packet callback against text and repeated NodeDB polling without hardware.
3. Test authenticated cloud validation, 2D count/person marker/labels, 3D marker and private handoff.
4. Test replay 1.1 team round trip and original 1.0 fixture import.
5. On the physical Mini Tracker and deployed DSC+, verify a real moving node, radio loss, restart, external node exclusion and Teams removal. This last gate requires hardware and release access.
