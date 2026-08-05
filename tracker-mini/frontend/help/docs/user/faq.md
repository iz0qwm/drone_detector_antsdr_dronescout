# FAQ

### Part of the Mini Tracker User Guide

---

## Purpose

This document answers common operator questions about Mini Tracker field use.

---

## Can Mini Tracker Work Without Internet?

Yes. Mini Tracker is designed to support offline field use when local resources are available.

Offline operation can include:

- Installed offline maps
- Local ADS-B reception
- Remote ID reception
- Meshtastic team awareness
- Local mission objects

Internet is required for online map sources, network ADS-B, OGN / FLARM network data, map downloads and Drone Sky Check imports.

---

## Is Traffic Proximity Awareness A Certified Collision-Avoidance System?

No. Traffic Proximity Awareness is informational only.

It helps the operator understand horizontal proximity between detected drones and aircraft, but it is not certified and must not be used as the sole basis for separation decisions.

---

## Why Is An Aircraft Or Drone Not Shown?

A traffic object may be missing for several reasons:

- The source is disabled.
- The receiver is not connected or not active.
- The object is outside receiver range.
- The object is not transmitting compatible data.
- Network connectivity is unavailable for network sources.
- The object is outside the current map view.
- The track became stale and was removed.

The absence of a marker is not proof that the area is clear.

---

## Why Is A Meshtastic Operator Online But Not On The Map?

An operator can be associated with a Meshtastic node without having current position data.

The Teams panel can show node identity and telemetry, while the map marker requires valid latitude and longitude from the operator node.

---

## Why Are External Meshtastic Nodes Shown?

External nodes are Meshtastic nodes visible to the gateway but not matched to configured mission operators.

They may be unrelated field nodes, stale NodeDB entries or mission nodes that have not yet been configured with matching short names.

---

## Do Message Entries Persist After Restart?

No. Recent notification and message entries are stored in backend memory and are not persistent across backend restarts.

---

## Should I Use Online Or Offline Maps?

Use **Automatic** for normal operation. Mini Tracker can use online maps when Internet is available and offline maps when local coverage is installed.

For planned field operations, prepare offline map coverage before deployment.

---

## Can I Change Receiver Settings During An Operation?

It is possible, but it should be done carefully.

Changing receiver settings or service state can interrupt live data until the hardware and backend service are available again.

---

## Related Documentation

- `user/dashboard.md`
- `user/settings.md`
- `user/maps.md`
- `user/traffic-monitoring.md`
- `user/teams.md`
