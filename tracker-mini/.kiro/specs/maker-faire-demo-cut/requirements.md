# MF-DEMO-1 requirements

Authority: Product Owner Demo Cut and real-position addendum, 20 September 2026. Implementation is authorized; no further review gate precedes this bounded slice.

- Existing authenticated, active DSC+ Workspace context and the two supplied accounts only. No new users or server provisioning.
- Field Nodes card: Mini Tracker Vescovio / MTRK26-0001, ONLINE / STALE / OFFLINE, heartbeat age, MANUAL position source, ADS-B / Remote ID / Meshtastic indicators and working map CTA.
- Prefer existing public presence. Read-only inspection verified `trackers/dsc-node02`, position 42.33163 / 12.60444, on 20 September. Preserve these published coordinates; the PO scenario fallback is 42.331704 / 12.604630. No coordinate matching or guessed GPS fix.
- Real heartbeat freshness must not be refreshed by reading or UI rendering. Capability flags indicate support only, never receiver health or observed RF. DEMO mode must be explicit and deterministic, with persistent labels on card and marker.
- Reuse the existing Leaflet map; add one owned marker only. Preserve other layers/planner state. Clean up subscriptions, timers and marker on logout/context changes; ignore late callbacks.
- Opening/closing Workspace and repeated CTA clicks must not leak resources. A focused marker may remain after Workspace closes until explicitly removed or context ends.
- No new Mini Tracker runtime work. Future sender must be isolated from local/offline operation. Internet loss ages the remote scene without blocking local receivers/UI. Manual site location does not relocate RF targets; later sources must distinguish LOCAL/RX, NETWORK/INTERNET and DEMO.

Excluded: MF-DEMO-2+ areas/traffic rendering, private sender/token/storage, live RF transport, Mission V3, 3D, recorder/replay, provisioning, device installation, MkDocs, production deploy, commit/push.
