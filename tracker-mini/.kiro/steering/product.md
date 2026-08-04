# Mini Tracker — Product Context

## What It Is

Mini Tracker is a **Portable Air Awareness Node** — a Raspberry Pi-based field device that integrates multiple traffic, positioning and communication sources into a single operational picture served through a local web Dashboard.

It is part of the **Drone Sky Check** ecosystem.

## Target Users

- Search and Rescue (SAR) teams
- Civil Protection
- Emergency response teams
- Drone operators
- Public safety organizations
- Technical field teams

## Core Capabilities

- Offline-first portable deployment
- Local ADS-B aircraft reception (readsb)
- Remote ID drone detection (DS110 receiver via MAVLink)
- OGN / FLARM glider and light aircraft awareness
- Meshtastic team position and messaging
- GPS node positioning
- Offline MBTiles map tiles
- Mission planning with GeoJSON layers
- Team coordination (operators matched to Meshtastic nodes)
- DSC cloud heartbeat and drone traffic ingest
- LCD local status display (20x4 I2C)
- OTA update workflow

## Design Principles

1. **Offline First** — core functions work without Internet
2. **Situational Awareness** — every feature must improve the operator's understanding
3. **Operational Simplicity** — hide complexity, surface critical information
4. **Modularity** — hardware and protocols evolve independently
5. **Reliability** — fewer robust features over many experimental ones

## Planned Features (not yet implemented)

1. Traffic Proximity Awareness
2. Meshtastic Operational Network
3. DSC Operational Area Synchronization

## References

- `frontend/help/docs/product-overview.md`
- `frontend/help/docs/product-vision.md`
- `AI_HANDOFF.md`
