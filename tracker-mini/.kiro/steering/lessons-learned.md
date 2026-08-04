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
- **Raffaello** è il proprietario del prodotto Mini Tracker. Approva requisiti, design, merge e operazioni distruttive sul device fisico.
- Quando il codice dice "chiedi a Raffaello" si intende per decisioni di prodotto o accesso privilegiato al device. Il lavoro quotidiano di sviluppo è con Claudia.

## Git Push Behavior in This Terminal

- The first `git push` (without explicit `origin main`) works but produces no visible success output due to terminal echo noise.
- A subsequent `git push origin main` reports "Everything up-to-date" because the first push already succeeded.
- To confirm push success, check `git log --oneline -1` and verify that `origin/main` points to the expected commit hash. Don't rely on push command output in this terminal.

## Onboarding Dependency

- Physical device inspection requires SSH access that cannot be automated without stored credentials.
- The onboarding task should be structured as two phases: local workspace (can complete independently) and physical device (requires interactive SSH session with credentials fornite da Raffaello).
