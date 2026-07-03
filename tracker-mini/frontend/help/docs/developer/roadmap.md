# Roadmap

### Part of the Mini Tracker Developer Documentation

---

## Purpose

This document tracks planned features and architectural improvements for Mini Tracker.

Unlike the other developer documents, this page describes future work rather than the current implementation.

Features should be removed from this document once implemented and documented in the appropriate User, Hardware or Developer documentation.

---

## Mission Planning

### Mission management

- Export complete missions.
- Import complete missions.
- Improve mission object management.
- Allow reordering mission objects.            NOT NECESSARY
- Improve mission object visibility controls.  DONE

### Mission objects

- Support editable labels and tooltips.
- Display distances on lines and paths.
- Display area measurements for polygons.   DONE
- Display radius information for circles.   DONE
- Improve marker styling and customization.

### External mission data

- Import mission areas from DSC when Internet connectivity is available.
- Mark imported DSC areas as read-only.
- Prevent accidental modification of synchronized operational areas.

---

## Situational Awareness

### Collision awareness

- Drone-to-drone proximity alerts.
- Drone-to-aircraft proximity alerts.
- Drone-to-helicopter proximity alerts.
- Operator safety alerts based on mission position.
- Configurable visual and acoustic warning thresholds.

---

## Teams and Meshtastic

### Team management

- Hide or remove unwanted external Meshtastic nodes.
- Improve operator management.
- Better gateway status information.

### Messaging

- Send messages from Mini Tracker to field operators.
- Support private Meshtastic messages.
- Support broadcast messages.

### Automatic notifications

Automatically notify operators about important events, including:

- Drone detected nearby.
- Aircraft approaching.
- Helicopter approaching.
- Mission updates.
- Safety warnings.

---

## Backup and Maintenance

- Create application backups from the Dashboard.
- Restore backups from the Dashboard.
- Export complete system configuration.
- Simplify migration to a new Raspberry Pi.

---

## Future Ideas

The following ideas are under evaluation and are not yet scheduled.

- Search pattern planning.
- Mission templates.
- Offline elevation profiles.
- Additional traffic sources.
- Extended SAR tools.
- Multi-tracker synchronization.

## Documentation

- Automatic documentation validation.
- Keep Help documentation synchronized with implementation.
- Expand developer documentation for every new subsystem.