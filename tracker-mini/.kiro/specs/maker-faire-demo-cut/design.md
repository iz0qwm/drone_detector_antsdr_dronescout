# MF-DEMO-1 design

Small classic browser scripts under DSC `public/js/field-operations/`, consistent with current assets: scene-store (normalization, freshness and subscriptions), fixture-source, presence-source (read-only single public document), field-node-card (UI/controller/owned marker). No dependencies.

Scene: `{origin, tracker:{id, serial, name, position:{lat,lon,source}, lastSeen, receivers}, operation:null, areas:[], targets:[], team:[], sampledAt}`. `sampledAt` is read time; only lastSeen determines connection state. Initial/missing/invalid heartbeat is OFFLINE. Thresholds: ONLINE <= 90 seconds (60-second emitter with margin), STALE <= 180 seconds, OFFLINE thereafter. Future timestamps beyond small clock tolerance are invalid.

Public presence is the default. MANUAL is supplied deployment configuration, not a field inferred from the legacy heartbeat. Capabilities render SUPPORTED / UNKNOWN, not ACTIVE. Read errors remain visible and retain only aged prior state; absence is not silently converted to live or demo. Explicit DEMO selector supplies controlled ONLINE/STALE/OFFLINE samples; switching modes invalidates old callbacks and removes the previous marker.

Gate uses authenticated context, active DSC_PLUS membership, workspaceSync and the supplied UID allowlist. This is UI exposure of already-public/synthetic data, not authorization for future private scenes. Context fingerprint includes UID/current operator; lifecycle events re-evaluate eligibility. Version guards invalidate asynchronous snapshots. Workspace open starts one timer/listener; close stops them unless map marker is in use. Logout/context change clears everything.

CTA closes Workspace with normal map-interaction restoration, focuses window.map on the scene position and creates/updates a dedicated marker. One marker; no map recreation, layer clearing or planner calls. Marker includes origin, status and manual-position label, and an explicit remove action.

Browser review must be distinguished from unit tests. No real user credentials are required by automated tests; test contexts are injected only in test harnesses.
