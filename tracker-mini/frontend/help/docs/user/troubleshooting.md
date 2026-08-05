# Troubleshooting

### Part of the Mini Tracker User Guide

---

## Purpose

This document provides operator-level troubleshooting procedures for common Mini Tracker field issues.

It focuses on checks that can be performed from the Dashboard and visible system state. It does not replace hardware maintenance procedures or physical validation on the Mini Tracker device.

---

## Troubleshooting Approach

Start from the Dashboard and move from general checks to source-specific checks.

Recommended sequence:

1. Confirm the Dashboard is reachable from the operator device.
2. Check the top status indicators.
3. Open the System panel and review service state.
4. Open Hardware Status and review connected devices.
5. Check whether the affected source is enabled.
6. Review logs when the Dashboard reports an error or a source remains unavailable.

Avoid changing multiple settings at once. Change one condition, wait for the Dashboard to refresh, then verify the result.

---

## Dashboard Not Reachable

If the Dashboard cannot be opened:

- Confirm the operator device is connected to the Mini Tracker Access Point or expected LAN.
- Check that the Mini Tracker node is powered.
- Try reconnecting to the Mini Tracker network.
- If using User LAN or Wi-Fi Client mode, confirm the expected IP address and network path.
- If the device was recently restarted, wait for services to start before retrying.

If the Dashboard remains unreachable, treat the condition as a system availability issue and use the approved maintenance procedure for the deployed device.

---

## Map Not Available

If the map is blank or does not show the expected area:

- Confirm the selected map source.
- Use **Automatic** map source for normal operation.
- Use **Offline Maps** when Internet connectivity is unavailable.
- Confirm that an offline map for the operating area is installed and active.
- Use **Online Topo** only when Internet connectivity is available.

Traffic and mission layers depend on the base map for geographic context, but changing the base map source does not change the underlying traffic or mission data.

---

## Traffic Not Visible

If aircraft, drones or other traffic are not visible:

- Confirm the required traffic source is enabled in the System panel.
- Check Hardware Status for the relevant receiver or service.
- Confirm the map is centered on the operating area.
- Check whether altitude filtering is hiding higher ADS-B aircraft.
- Wait for the normal refresh interval.
- Review logs if the source remains unavailable.

The absence of visible traffic is not proof that the area is clear. It may indicate missing transmissions, receiver range, antenna placement, disabled services, stale data or unavailable network sources.

---

## Remote ID Not Visible

If Remote ID drones are not visible:

- Confirm **Remote ID (DS110)** is enabled.
- Check Hardware Status for DS110 receiver connection and heartbeat.
- Confirm the drone is transmitting compatible Remote ID data in range.
- Wait for the map refresh interval.
- Remember that stale Remote ID markers fade and then disappear after the retention period.

Development-machine tests do not prove DS110 hardware reception on the physical Mini Tracker.

---

## ADS-B Not Visible

If ADS-B traffic is missing:

- Confirm **ADS-B Local (RTL-SDR)** is enabled when using the local receiver.
- Check Hardware Status for ADS-B receiver and decoder state.
- Confirm antenna placement and receiver connection.
- Confirm **ADS-B Network** only when Internet connectivity is available.
- Enable broader altitude visibility if relevant aircraft may be above the default filter.

Local ADS-B can operate offline when the receiver and decoder are active. Network ADS-B requires Internet connectivity.

---

## Meshtastic Operators Not Visible

If mission operators are missing:

- Confirm **Meshtastic** is enabled.
- Check Hardware Status for Meshtastic gateway connection and link activity.
- Open **Teams** and verify that the expected node appears.
- Confirm the configured operator short name matches the Meshtastic node short name.
- Confirm the operator node has position data if a map marker is expected.

Operators can appear in the Teams panel even when they do not have current position data for map display.

---

## Messages Not Delivered

If Meshtastic messages are not delivered:

- Confirm the target operator is online in the Teams panel.
- Confirm the operator has an associated Meshtastic node identifier.
- Check Meshtastic gateway and link status.
- Use short messages to reduce radio airtime.
- Review the Messages section for sent, received or failed entries.

Meshtastic delivery depends on radio conditions, gateway state, node availability and network configuration.

---

## GPS Has No Fix

If GPS is connected but has no fix:

- Confirm Hardware Status shows the GPS receiver.
- Check whether the GPS fix indicator is available.
- Move the antenna or Mini Tracker node to improve sky visibility.
- Allow time for the receiver to acquire satellites.
- Use the manual DSC position workflow when GPS is unavailable and the operation requires a known tracker position.

GPS receiver availability and GPS fix state are separate conditions.

---

## System Update Issues

If a system update cannot proceed:

- Confirm the package was uploaded successfully.
- Run the available package checks before requesting installation.
- Do not install updates during active operations unless service interruption is acceptable.
- Review logs if validation or installation preparation fails.

Update and restore procedures are covered in the maintenance documentation.

---

## Related Documentation

- `user/dashboard.md`
- `user/settings.md`
- `user/maps.md`
- `user/traffic-monitoring.md`
- `user/teams.md`
- `maintenance/diagnostics.md`
- `maintenance/logs.md`
