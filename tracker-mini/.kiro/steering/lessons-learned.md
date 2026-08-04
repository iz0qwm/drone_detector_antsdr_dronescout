---
inclusion: manual
---

# Mini Tracker — Lessons Learned

Project-specific patterns, preferences, and lessons learned over time.

## Shell Command Execution on Windows

- The terminal output in this workspace has heavy character-echo noise (each character of the command is repeated as it's typed). Commands work but output parsing requires ignoring the echo prefix.
- `plink` requires host key acceptance before `-batch` mode works. Use OpenSSH `ssh` with `-o StrictHostKeyChecking=accept-new` instead.
- The Raspberry Pi does not accept public key auth from this Windows machine — password-based SSH is required.

## Documentation vs Implementation Drift

- `ISTRUZIONI.TXT` is a historical scratch file and is NOT authoritative. It references project structures (db/, schema.sql, mission_export.py) that never existed or were removed. Always trust the actual file tree.
- `mkdocs.yml` uses `language: it` which contradicts `DOCUMENTATION.md`'s English-only policy — this is a known inconsistency awaiting a decision.

## Configuration Secrets in Repository

- `config/map_provider.json` contains a Thunderforest API key committed to Git.
- `backend/services/network_manager.py` has the hotspot password "tracker123" hardcoded.
- These are known; Raffaello has been asked whether to externalize them.

## Absolute Workspace Isolation

- This project (Mini Tracker / Drone Sky Check) is **completely separate** from all Octotelematics projects (GD201, HB3, BlobParser, OCTO sandbox, etc.) that may be open in the same Kiro session.
- Nothing from Mini Tracker must ever be written to, influence, or reference steering/lessons-learned files in other workspaces.
- Nothing from Octotelematics workspaces applies here — different product, different team, different architecture.
- All Mini Tracker steering and knowledge files go exclusively in `tracker-mini/.kiro/steering/`.
- If Kiro context includes Octotelematics steering rules, they must be ignored when working on Mini Tracker.

## People

- **Claudia** è l'operatrice che lavora su questo progetto con Kiro (utente della sessione).
- **Raffaello** è il proprietario del prodotto Mini Tracker. Approva requisiti, design, e operazioni distruttive sul device fisico.
- Quando il codice dice "chiedi a Raffaello" si intende per decisioni di prodotto o accesso privilegiato al device. Il lavoro quotidiano di sviluppo è con Claudia.
- **In autopilot mode**: non fermarsi tra milestone, non chiedere permesso per procedere, non aspettare review intermedi. Claudia ha il controllo e può interrompere quando vuole.

## Git Push Behavior in This Terminal

- The first `git push` (without explicit `origin main`) works but produces no visible success output due to terminal echo noise.
- A subsequent `git push origin main` reports "Everything up-to-date" because the first push already succeeded.
- To confirm push success, check `git log --oneline -1` and verify that `origin/main` points to the expected commit hash. Don't rely on push command output in this terminal.
- UPDATE: `git push` does show full progress output when there are actual objects to push. The silent case only occurs when the push has already happened.

## Traffic Data Field Differences

Critical for any feature working with multiple traffic sources:

- **Aircraft `updatedAt`** = millisecond epoch (number). **Drone `last_seen`** = ISO 8601 UTC string. Always convert before comparing.
- **Aircraft altitude** ≈ MSL (barometric or geometric, mixed depending on source). **Drone altitude** = WGS84 ellipsoid (ODID encoding). These are NOT comparable — difference can be 20-50m in Italy.
- **Aircraft speed** = m/s (converted from knots). **Drone speed** = m/s (from ODID). Units happen to match, but sources differ.
- **Aircraft identifier** = `icao` (hex string). **Drone identifier** = `serial` (manufacturer string).
- **Staleness**: Aircraft uses miss-counter + 60s grace in frontend. Drones are removed immediately when absent from API.

## Frontend Traffic Module Access Pattern

- Aircraft state lives in `markersByIcao` (Map) inside `air-layer.js` module scope — accessed via `window.AIR`
- Drone state lives in `DRONES.markers` (plain object keyed by serial) — directly accessible
- Both are module-scoped but exposed on `window` globals
- Refresh cadences: Aircraft 15s, Drones 5s, OGN 10s — any cross-source feature must handle different update rates

## ADSBNet Enable/Disable Architecture Gap

- ADSBNet has NO backend-side enable/disable setting. The enable preference lives **entirely** in browser `localStorage("adsbNetworkEnabled")`.
- The backend (`air_network.py`) always serves data if the API is called — it doesn't check any preference.
- This means any new backend feature that needs to know whether the user wants ADSBNet (like the proximity engine) must have its OWN configuration flag (e.g., `proximity.adsb_net_enabled`).
- Do not assume the backend can read the frontend localStorage preference.
- This pattern may apply to OGN as well (`localStorage("ognNetworkEnabled")` — same frontend-only toggle pattern).
- **UPDATE**: MT-TRAFFIC-01 introduces a unified `settings.traffic.adsb_net_enabled` backend setting with migration from localStorage. Once implemented, this replaces the localStorage-only pattern.

## Updater Deployment Scope

- The Mini Tracker updater deploys ONLY `backend/` and `frontend/`.
- `config/settings.json` is NEVER overwritten by updates — it stays on the device.
- Every new configuration key must have a code-defined default in backend Python.
- Missing config sections must be handled gracefully (merged with defaults, no crash).
- Tests, specs, pytest.ini, .kiro/ files are development-only and not deployed.
- Before finishing any feature: mentally verify that removing everything outside `backend/` + `frontend/` would not break the running application.

## Python Path for Tests

- Tests live at workspace root: `tests/`
- Backend code lives at: `backend/services/`, `backend/routes/`
- The `tests/conftest.py` must add `backend/` to `sys.path` so pytest can import `services.*` and `routes.*`
- Use `sys.path.insert(0, str(BACKEND_DIR))` in conftest.py
- The canonical test command is `python -m pytest tests -v` (from workspace root)
- Do NOT use `py -3` (Windows-specific); use `python` on dev, `python3` on RPi

## .gitignore for __pycache__

- Add `.gitignore` with `__pycache__/` and `*.pyc` BEFORE the first test run to avoid committing bytecode
- The first test run without .gitignore will create cached files that need `git rm --cached` to fix

## Onboarding Dependency

- Physical device inspection requires SSH access that cannot be automated without stored credentials.
- The onboarding task should be structured as two phases: local workspace (can complete independently) and physical device (requires interactive SSH session with credentials fornite da Raffaello).
