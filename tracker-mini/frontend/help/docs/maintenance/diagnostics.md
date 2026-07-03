# Diagnostics

### Part of the Mini Tracker Maintenance Documentation

---

## Purpose

This document describes the diagnostic information currently available in Mini Tracker.

Mini Tracker does not currently include a dedicated diagnostics subsystem. Diagnostic information is presented through existing Dashboard panels and status indicators.

---

## Current Status

A dedicated diagnostics module is planned for future development.

The current implementation does not provide a separate diagnostics console, diagnostic report generator or guided diagnostic workflow.

---

## Dashboard Diagnostic Information

The Dashboard already exposes several status areas that support maintenance and troubleshooting.

| Area | Current Information |
|----------|---------------------|
| System status | Hostname, CPU usage, RAM usage and disk usage. |
| Service indicators | Network, ADS-B receiver, ADS-B network, Remote ID, OGN, Meshtastic and DSC state. |
| Hardware status | Wi-Fi Client adapter, DS110 receiver, DS110 heartbeat, Meshtastic gateway, Meshtastic link, ADS-B receiver and ADS-B decoder state. |
| GPS status | GPS availability, fix state, fix mode, satellite count, HDOP and position. |
| Logs | In-memory backend log entries shown in the Dashboard log viewer. |

These areas are distributed across the Dashboard rather than grouped into a single diagnostics page.

---

## System Status

System status is shown in the Dashboard System panel.

The backend reports basic host information using local operating system and resource data. The Dashboard refreshes this information periodically.

---

## Hardware Status

Hardware status is shown inside the System Settings area.

The Dashboard combines hardware status data with GPS status data. This allows maintainers to verify whether expected receivers, gateways and decoders are detected or active.

---

## Service Indicators

The top status bar and System panel show service indicators for operational services.

These indicators reflect current implementation checks such as Internet availability, local ADS-B decoder freshness, DS110 worker state, Meshtastic worker state and Internet-dependent network services.

---

## GPS Status

GPS status is available through the hardware status display.

The current status includes receiver availability, fix state, fix mode, satellite count, HDOP and current position when available.

---

## Logs

The Dashboard includes a System Logs modal.

Logs can help maintainers observe backend events during map downloads, traffic processing, updates, mission actions and service activity. Logs are memory-backed and are not persistent across backend restarts.

---

## Future Direction

A dedicated diagnostics module is planned for future development.

No future diagnostic behavior is defined in the current implementation. Future documentation should be added only when the feature exists.

---

## Related Documentation

- `maintenance/logs.md`
- `maintenance/update.md`
- `user/dashboard.md`
- `hardware/overview.md`
- `developer/api.md`
