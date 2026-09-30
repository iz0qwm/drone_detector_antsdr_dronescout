# Private Field Operations scene synchronization

MF-DEMO-3 projects saved Mini Tracker mission areas into DSC+ Field Operations. Mini Tracker remains the authority; remote views cannot edit or command it. MF-DEMO-1/2 UI is Product Owner accepted. The implementation requires backend deployment and device configuration before physical validation.

## Source and normalized contract

`backend/services/field_scene.py` reads `get_current_mission()` and `list_layers(mission_id)` in-process. The existing `/home/pi/tracker-mini/missions` storage is unchanged. Only saved layers are read; editing gestures and browser-local Show/Hide are not publication controls.

User layers have no `properties.source`. Imported DSC zones and other layers/features with a source are excluded. Points without Circle metadata and line geometries are excluded. Unsupported/malformed operational geometry rejects the complete cycle and reports an error, preserving the remote last good projection. Selected mission metadata, layer names and optional six-digit hex colors are preserved. Descriptions, credentials and operational area altitudes are not exported. Configured Team operators with valid Meshtastic position observations are exported separately in `team`; DS110 Remote ID and local readsb ADS-B remain the only traffic targets in `targets`.

```javascript
{
  operation: { id: "mission_001", name: "Field mission" },
  areas: [{
    id: "MTRK26-0001:mission_001:layer_1:0",
    name: "Work area", origin: "LIVE", source: "MINI_TRACKER",
    style: { color: "#12ab34" },
    geometry: {
      type: "Polygon",
      coordinates: [[[12,42], [13,42], [13,43], [12,42]]]
    }
  }]
}
```

Rectangle becomes Polygon. A Circle saved as Feature Point with `properties.leafletType = "Circle"` and `properties.radius` becomes:

```javascript
{ type: "Circle", center: [12,42], radiusM: 125 }
```

Coordinates are longitude/latitude; radius is metres. FeatureCollection children use stable zero-based indices within the saved layer. IDs survive rename/geometry changes while the saved child order remains stable. No altitude is invented. The same canonical projection can feed the future 3D adapter without a transport change.

Bounds: 128 KiB UTF-8 JSON, 100 areas, 100 children per collection, 20 polygon rings, 2000 polygon vertices, 100000 metre maximum circle radius; IDs use letters/digits/underscore/hyphen up to 80 characters, names up to 200 characters. Invalid names/IDs or out-of-range/non-finite coordinates also fail the cycle. A first error has no valid geometry; it is never reported as a successful empty mission.

## Sender, timing and failure handling

`backend/services/field_sender.py` is opt-in. The normal `backend/app.py` script startup starts two daemon threads. Importing `app` for System Update verification does not start this private sender. Deployments using a different entry point such as a WSGI importer must be verified separately; no WSGI launcher is supplied here.

- Collector: every 5 seconds, saved disk state only. Rechecks selected mission before and after reading.
- Uploader: awakened by a new snapshot, exactly one request at a time, one replaceable latest value. No persistent queue or history.
- Requests: existing `requests` library, HTTPS, connect/read timeout 2/3 seconds, redirects disabled, no response-body download. No token/geometry in error logs.
- Failed upload: retry after 5, 10, 20, then at most 30 seconds. Next attempt uses the latest collected state. No replay of an offline backlog.
- Receiver threads, local map/planner, local dashboard and public heartbeat/RID senders do not call or wait for this sender. Network failure only affects remote freshness.
- Restart: reread saved mission state. Correct device time/NTP is required; server rejects snapshots older than 60 seconds or more than 10 seconds in the future, and transactions reject timestamps older than or equal to the last accepted sample.
- Successful empty/no selection clears remote areas. Read/normalization ERROR carries no replacement geometry; cloud retains the previous valid operation/areas and last-good time. Deleted areas disappear on the next valid complete snapshot.

DSC reads every 5 seconds while the approved card or operational map is active. The healthy-path target is saved change visible within about 10 seconds (collection + network + reader); it is not a measured hardware guarantee. After interruption, bounded retry backoff can add up to 30 seconds before upload recovery. Geometry is retained when its last-good sample ages beyond 15 seconds and the panel shows STALE. Tracker heartbeat freshness remains separately controlled by the existing public source.

## Private DSC boundary

`ingestFieldOperation` and `readFieldOperation` run in `europe-west8`. The physical `dsc-node02` identity maps to display serial `MTRK26-0001`; the sender refuses a different configured identity rather than renaming it.

The write endpoint validates its Secret Manager bearer token, identity, schema allowlists, time, size and coordinates. It only writes `fieldOperationsLatest/MTRK26-0001`. Firestore stores the bounded projection as `projectionJson` (nested coordinate arrays cannot be stored directly), with content hash, sample/receive time, last-good time and status. Unchanged geometry is not rewritten; freshness metadata still writes each accepted cycle. No history and no device credential in the document.

Firestore rules deny all direct client read/list/write access, including for approved viewers. The authenticated read callable checks ACTIVE DSC_PLUS and `workspaceSync` with the existing account access resolver, plus one of the two Product Owner supplied UIDs. It returns the decoded canonical projection. Invalid/expired memberships and unrelated users are rejected.

The DSC LIVE scene consumes this projection. Tracker presence remains public LIVE/MANUAL. The private projection supplies real DS110 Remote ID and local readsb ADS-B targets; synthetic RID and aircraft belong only to explicit DEMO mode. Last-good private areas remain during transient read errors; access denial clears them. Logout, context/source change and map/card close stop their listeners, and late callbacks cannot restore an old context. Private geometry is not stored in browser persistent storage.

## Real Remote ID targets in the private LIVE scene

The collector reads the existing in-memory DS110 aircraft state and sends at most 32 valid `RemoteID` or `DJI DroneID` positions in the same authenticated snapshot as the areas. This does not depend on the selected mission or whether a saved area contains the aircraft. Each private target has a stable `rid:remoteid:<serial>` or `rid:dji:<serial>` ID, `type: RID`, `origin: LIVE`, `source: LOCAL_RX`, receiver provenance, position and the receiver's time of the last valid location message. Basic ID and other non-location messages do not refresh that position time. Coordinates with no usable location, missing observation time, positions older than 10 seconds, and positions more than 2 seconds in the future are omitted. The receiver may continue to retain an older object for its local map after its private position has expired.

The DSC write endpoint validates the bounded target list and keeps it in the private last-good projection. The LIVE composer checks position age again and removes an expired RID on its one-second refresh, including during sender/network outages. A later valid location message restores the same ID. Area status and target freshness are separate: an old last-good area can remain marked STALE while an old RID disappears. A valid RID outside all saved areas remains a LIVE target; Geoawareness uses the existing horizontal area and UAS relationship calculations when suitable references exist. A lone RID with no area or other target has no relationship to calculate.

The OpenDroneID geographic altitude, when valid, is exported as metres with `reference: WGS84_ELLIPSOID`. The separate decoded `height` value is not exported as an AGL altitude because the current decoder does not preserve its height reference type; a zero height is not a known ground level. DSC 2D details and replay retain the WGS84 reference. The 3D view keeps its existing ground marker fallback when no terrain conversion is available; it must not display the WGS84 value as height above terrain. Recordings preserve target ID, provenance and observation time. The public RID ingestion and normal DSC map remain on their existing path, so an operator viewing public traffic and the private Field Operations overlay together may see two representations of one detection until a coordinated deduplication design is implemented.

## Real local ADS-B aircraft in the private LIVE scene

The collector also reads `/run/readsb/aircraft.json` directly. It accepts only current `adsb_icao` records with a six-digit hexadecimal ICAO, valid position, and readsb `now` and `seen_pos` fields. The canonical ID is `adsb:<lowercase-icao>`; the trimmed callsign is a label, with ICAO as fallback. The source is `ADSBRx`, distinct from network `ADSBNet`. Re-reading an unchanged file does not change `observedAt`: it is `(readsb now - aircraft seen_pos)` in UTC milliseconds. Positions older than 10 seconds or more than two seconds in the future are omitted. Missing, malformed or oversized readsb files yield no private ADS-B targets without stopping the area and RID sender.

Valid `alt_geom` has precedence and is converted from feet to metres with `reference: WGS84_ELLIPSOID`. Otherwise a numeric `alt_baro` becomes metres with `reference: BARO`. Missing/invalid values and the string `ground` are not converted to zero altitude; numeric zero remains valid. Ground speed `gs` becomes `speedMps` from knots, and `track` becomes heading. Category `A7` is the only explicit rotorcraft classification; other aircraft remain generic. The private acquisition retains the local API's non-rotorcraft 1000-metre altitude filter and the shared 32-target/128-KiB bounds. No fixed 20 km Field Operations acquisition radius exists in the current implementation: the local map API uses viewport bounds and the separate proximity engine uses a 10 km relationship radius. The private collector does not use polygon containment or network ADS-B.

RID and ADS-B share one `targets` array, authenticated upload, LIVE source lifecycle, 2D/3D renderer, Geoawareness core and recorder/replay. The LIVE composer removes old positions on its regular refresh even if Firestore retains the last good projection; recovery with the same ICAO reuses the same ID. Google 3D keeps the truthful unresolved-altitude ground marker fallback for LIVE aircraft, retaining the source altitude in details. The normal DSC aircraft map reads its separate `traffic_live` collection; its writer is not present in this Mini Tracker workspace. Both map layers may show one real aircraft if enabled together.

## Real configured Meshtastic Team operators

The existing Mission Teams configuration stores both a display long name and an exact, case-sensitive Meshtastic short name. `get_team_status()` matches heard nodes by **short name**, excludes the gateway from `operators[]`, and places unmatched heard nodes in `external_nodes[]`. Field Operations reads only `operators[]`; it never exports the gateway or `external_nodes[]`. If two heard nodes match one configured operator ID, the private projection omits that ambiguous association.

The Meshtastic packet callback records `position_observed_at` only for a valid `POSITION_APP` location. It uses the decoded position timestamp when present, otherwise the packet receive time; repeated NodeDB polling and text/telemetry packets do not renew it. The private collector requires this timestamp and valid coordinates, rejects future observations beyond two seconds, retains a position for up to 30 minutes, and limits the list to 32 operators within the existing 128-KiB snapshot. A configured person without a valid position stays in local Teams but has no geographic Field Operations marker. Process restart requires a newly received position packet before geographic publication resumes.

Each exported member uses `id: mesh:!<eight lowercase hex digits>` from the heard node ID, configured long name as display `name`, `shortName`, `role: TEAM_OPERATOR`, `source: MESHTASTIC`, position and UTC millisecond `observedAt`. Optional altitude is metres with `reference: UNKNOWN`, because the current radio decoding does not establish its datum. `scene.team` remains separate from `scene.targets`, even outside an operational polygon; people do not enter air-traffic counts or Geoawareness. DSC marks them fresh for ten minutes, stale until 30 minutes, then removes them. The same private scene feeds Leaflet 2D, Google 3D ground person markers and the existing recorder. Replay V1 retains old 1.0 recordings and uses scene schema 1.1 for recordings containing team members, with recorded freshness preserved.

## Configuration and release prerequisites

The existing private sender configuration and cloud secret are reused. This ADS-B change has not been deployed, committed, pushed or physically installed by this task.

| Setting | Placement / purpose |
| --- | --- |
| `FIELD_OPERATIONS_DEVICE_TOKEN` | Firebase/Google Secret Manager secret, bound only to `ingestFieldOperation`. Use a cryptographically random token with at least 32 random bytes encoded as URL-safe ASCII; no spaces. |
| `enabled`, `token` | Keys in `/etc/tracker-mini/field-operations.json`. `enabled` must be boolean true. `token` must match the cloud secret; accepted length 32–256 ASCII characters. Keep this file outside Git, update ZIPs, the web root and shared reports. |
| `FIELD_OPERATIONS_CONFIG` | Optional process environment override for the private configuration file path; default is the path above. No environment changes are required for the default. |
| `dsc.node_id` | Existing Mini Tracker setting (or hostname fallback). Must already be `dsc-node02`; do not change device identity for this integration. |

1. Coordinate an authorized release of the local DSC functions, rules and frontend. Current working files are uncommitted; a pull alone cannot obtain them. Separately authorize any future commit/push/deploy.
2. The DSC maintainer creates the named Secret Manager secret and binds it through the included function declaration, deploys both new functions, the rules and the frontend. Use the approved DSC environment; the sender currently targets `https://europe-west8-droneskycheck-d0136.cloudfunctions.net/ingestFieldOperation`. Changing environment requires an explicit configuration/code review. Never put the secret in frontend configuration.
3. Verify each supplied DSC account has active DSC_PLUS and `workspaceSync` using existing account administration. No new users or automatic grants are created.
4. Prepare the normal complete Mini Tracker backend update ZIP with `backend/` at archive root (including `app.py`, all `routes/`, all `services/` and supporting backend modules). Do not create a three-file partial ZIP: System Update imports the staged backend. Include neither private configuration nor missions, `.git`, tests or virtual environments. Existing dependencies are unchanged.
5. On the physical device use System → System Update → select ZIP → Upload & Verify. Continue with Install Update only after the existing checks succeed. The external installer that applies the request is not present in this repository; use the established installation procedure and verify the installed version afterward.
6. Create the private configuration file on the device through its administrator, readable only by root and the tracker service identity (for example root-owned, service group-readable, mode 0640). Check the actual `tracker-mini.service` user/group before setting ownership. Enable only once the cloud endpoints/rules/frontend are deployed. Restart using the existing Tracker restart workflow, then verify a `[FIELD] Private operational-area sender started` log. Never paste the token into shared logs/screenshots.
7. Verify device clock and normal script startup, and sign in to DSC+ with one approved account. Open Pilot Workspace → Field Nodes → Presenza reale → Apri vista operativa. The localhost review page uses simulated authentication and cannot validate cloud access or physical synchronization.

To stop only this private integration, set `enabled` false and restart the tracker through the established workflow. Public flows remain unchanged; existing remote geometry ages to STALE. Keep the existing System Update backup for application rollback.

## Physical acceptance checklist (Product Owner)

1. Open Mini Tracker Mission Planning.
2. Select or create a field mission.
3. Draw a Polygon.
4. Save it.
5. Open DSC+ Field Operations in Presenza reale mode with an approved account.
6. Within the <=10 second healthy-path target, see the same area marked LIVE and the selected mission name.
7. Rename/edit the polygon and save.
8. Confirm name/geometry update without duplicate layers or camera movement.
9. Delete the polygon and save through the normal local workflow.
10. Confirm the remote area disappears.
11. Create/save a Circle.
12. Confirm the same centre and radius remotely; also try a Rectangle and mission switch.
13. Disconnect Mini Tracker Internet while retaining access to its local interface.
14. Confirm local maps, planning and receivers continue normally.
15. After 15 seconds without valid updates, remote areas show STALE and remain visible (public heartbeat has its independent timing).
16. Restore Internet.
17. Confirm only the latest saved scene returns; allow the documented retry backoff after an outage. Record save/visible times and any errors.

Also verify both approved accounts, unrelated/expired account denial, logout cleanup, and restart recovery. These are physical/deployed acceptance steps, not results of local mocks or the emulator.

The private LIVE scene includes received DS110 RID positions, local readsb ADS-B aircraft and positioned, configured Meshtastic Team operators for the existing 2D, 3D and recorder/replay paths. Bidirectional edits and Mission V3 integration are outside this slice. Physical Meshtastic ingestion into the private scene, deployed cloud authorization and rendering on the actual Mini Tracker display still require device and deployment checks.
