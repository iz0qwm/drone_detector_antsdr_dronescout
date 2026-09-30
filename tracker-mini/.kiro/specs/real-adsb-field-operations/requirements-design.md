# Real local ADS-B in private Field Operations LIVE

## Requirements and acceptance

- Add current local readsb aircraft to the same bounded private snapshot and `scene.targets` used by real RID. Do not add a network provider or change the public traffic path.
- Preserve RID and area transport. ICAO is the stable identity; callsign is only a label. Operational polygons do not filter aircraft.
- Derive position time from readsb file `now - seen_pos`, never from HTTP/collector read time. The same file reread later must produce the same observation time and expire naturally.
- Preserve `alt_geom` as WGS84 ellipsoid altitude, otherwise numeric `alt_baro` as barometric altitude. Missing or `ground` altitude stays unknown; numeric zero remains valid. Never infer AGL.
- Reuse Field Operations 2D, 3D, Geoawareness and recorder/replay consumers. Preserve the real RID flow and DEMO isolation.

## As-built seam and design

The local source is `/run/readsb/aircraft.json`. `backend/services/air_local.py` reads it for the local map API, but its `updatedAt` is generated at API read time and it merges altitudes, so that DTO is not suitable for the private scene. The existing private collector reads only DS110 RID state. Local readsb data reaches the normal DSC traffic path separately; that public path is not an input to Field Operations.

1. Add a small readsb file adapter in `backend/services/field_adsb.py`. Keep it independent of network ADS-B and hardware control. Missing/unavailable readsb gives an empty aircraft input while areas and RID continue.
2. Normalize readsb aircraft in `backend/services/field_scene.py` beside RID. Require a canonical six-hex-digit ICAO and valid lat/lon, `now` and `seen_pos`, and recent position. Use `adsb:<icao>` and explicit `ADSBRx` provenance. Limit the combined private target list to 32 and the existing 128 KiB snapshot.
3. Use the existing local aircraft altitude policy (non-rotorcraft over 1000 m excluded) without treating missing altitude as zero. Retain source category `A7` as the only explicit rotorcraft classification; other aircraft remain UNKNOWN. No polygon containment or new 20 km filter is introduced. The local map API remains untouched.
4. Extend the existing private DSC validator and LIVE composer to accept both normalized RID and ADS-B targets. `speedMps` is an optional addition to the existing target shape so 2D/3D details and strict replay can preserve source motion. Do not create another transport or recorder adapter.
5. Reuse the existing Google 3D unresolved-altitude ground marker fallback for LIVE aircraft. WGS84 and barometric altitude remain source details, not terrain offsets. Geoawareness uses its current core and selection rules.

## Failure, compatibility and tests

- Area-only and RID-only snapshots remain valid. Invalid aircraft are skipped by the local adapter; a missing file does not break area synchronization. An invalid private target submitted to DSC rejects the complete snapshot.
- Private aircraft expire from the LIVE scene on observation age, even when the cloud retains last-good data. An aircraft returning with the same ICAO reuses its ID.
- Test file rereads, freshness/recovery, ICAO/callsign, altitude precedence/absence/zero, category, speed/track, bounds, RID coexistence, private validation, 2D/3D, Geoawareness and recorder/replay. No development test proves physical readsb reception or online Google rendering.
- No service, GPIO, port, serial, startup or power-management configuration change. No Meshtastic schema or behavior change; `scene.team` remains separate from `scene.targets` for future work.
