# Restore

### Part of the Mini Tracker Maintenance Documentation

---

## Purpose

This document describes the restore capabilities currently available in Mini Tracker.

It distinguishes update rollback behavior from a general system restore feature.

---

## Current Status

Manual application restore is not currently available through the Dashboard.

Mini Tracker does not provide a Dashboard workflow for selecting a user-managed backup and restoring the application or system state. A general restore feature is planned for a future release.

---

## Update Rollback

The update subsystem includes restore functions for internally generated update backups.

These restore functions are used in the context of software updates. They can restore backend and frontend application files from backups created by the updater before installation.

This is not the same as a general system restore feature. It is part of the update safety workflow and is tied to update-generated backups.

---

## Restore Scope

| Restore Area | Current Behavior |
|----------|------------------|
| Backend application files | Supported by update backup restore functions. |
| Frontend application files | Supported by update backup restore functions. |
| Dashboard manual restore workflow | Not implemented. |
| Full operating system restore | Not implemented. |
| Configuration restore workflow | Not implemented as a standalone feature. |

---

## Operational Notes

The Dashboard System Update panel shows installed package information and rollback status when this information is present in the updater metadata.

Operators should not treat this as a complete restore console. It reflects update state rather than a full maintenance restore workflow.

---

## Related Documentation

- `maintenance/backup.md`
- `maintenance/update.md`
- `maintenance/diagnostics.md`
- `developer/backend.md`
- `developer/api.md`
