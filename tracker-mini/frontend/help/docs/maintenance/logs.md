# Logs

### Part of the Mini Tracker Maintenance Documentation

---

## Purpose

This document describes the current Mini Tracker logging system.

It explains how backend logs are stored, displayed and cleared in the current implementation.

---

## Logging Model

Mini Tracker currently uses an in-memory backend logging service.

Backend services call the shared logger to create entries with:

| Field | Description |
|----------|-------------|
| `time` | Short local time used for display. |
| `timestamp` | ISO timestamp. |
| `component` | Source component name. |
| `level` | Log level, such as `INFO` or `ERROR`. |
| `message` | Log message text. |

The logger keeps up to 2000 entries in memory.

---

## Dashboard Log Viewer

The Dashboard includes a **Logs** button that opens the System Logs modal.

When the modal is open, the Dashboard loads logs from the backend and refreshes them periodically. Entries are displayed newest first in the format:

```text
[time] [level] [component] message
```

The log viewer also includes a copy action that copies the visible log text from the Dashboard.

---

## Log Clearing

The System Logs modal includes a **Clear Logs** action.

Clearing logs removes the current in-memory log entries from the backend logger. It does not delete files because the current logger does not write persistent log files.

---

## Console Output

Each backend log entry is also printed to standard output.

This allows service managers, terminals or external runtime environments to capture backend output independently from the Dashboard log viewer.

---

## Persistence

Logs are not persistent across backend restarts.

Because the current logger stores entries in memory, restarting the backend process clears the log history available through the Dashboard.

---

## Current Limitations

The current logging system has the following limitations:

- Logs are stored in memory only.
- Logs are limited to the most recent 2000 entries.
- Logs are lost when the backend restarts.
- The Dashboard log viewer does not provide filtering or search.
- The current implementation does not provide log export as a backend file.

The logging system is expected to evolve over time.

---

## Related Documentation

- `maintenance/diagnostics.md`
- `maintenance/update.md`
- `developer/backend.md`
- `developer/api.md`
