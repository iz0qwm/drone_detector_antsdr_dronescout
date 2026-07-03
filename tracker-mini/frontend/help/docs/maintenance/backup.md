# Backup

### Part of the Mini Tracker Maintenance Documentation

---

## Purpose

This document describes the backup capabilities currently available in Mini Tracker.

It distinguishes operator-managed backup workflows from the internal backups used by the update subsystem.

---

## Current Status

Mini Tracker does not currently provide an automatic application backup feature from the Dashboard.

There is no Dashboard workflow for creating, downloading, scheduling or managing general application backups. Manual application backup and restore workflows are planned for a future release.

---

## Update Backups

The current implementation creates internal backups as part of the software update process.

During the update pre-install check, the backend update service creates a backup of the live backend and frontend directories before an installation request is created. These backups are stored under the updater working directory and are intended to support update safety.

They are not presented as a general user-managed backup system.

---

## Backup Scope

The implemented update backup includes:

| Area | Current Behavior |
|----------|------------------|
| Backend application files | Archived by the update subsystem. |
| Frontend application files | Archived by the update subsystem. |
| User-managed backup scheduling | Not implemented. |
| Dashboard backup download | Not implemented. |
| Full system image backup | Not implemented. |

The update subsystem currently keeps only a small number of internal backups and removes older backup directories during cleanup.

---

## Future Direction

General application backup is planned for a future release.

Future backup documentation should be added only when the implementation exists. Until then, maintainers should treat the current backup capability as part of the update process, not as a standalone maintenance feature.

---

## Related Documentation

- `maintenance/restore.md`
- `maintenance/update.md`
- `maintenance/logs.md`
- `developer/backend.md`
- `developer/api.md`
