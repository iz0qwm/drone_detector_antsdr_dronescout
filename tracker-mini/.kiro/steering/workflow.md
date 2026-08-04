# Mini Tracker — Development Workflow

## Branch Strategy

- Kiro works directly on the current checked-out branch (currently `main`).
- A separate feature branch is not required.
- Work is divided through Feature Specs and small, focused commits.
- Completed and tested work is pushed to the current remote branch.
- Raffaello retrieves completed work using a normal `git pull`.
- Pushed history must never be rewritten.

## Development Cycle

1. Verify current branch, commit, and that local is synchronized with remote
2. Read or create the relevant Feature Spec (`.kiro/specs/`)
3. Implement the work incrementally
4. Create small, focused and meaningful commits
5. Run all relevant available tests
6. Perform physical Mini Tracker validation when required
7. Update documentation when required
8. Update `AI_HANDOFF.md`
9. Push completed and tested commits to the current remote branch

## Git Rules (workspace boundary)

- Always restrict Git operations to `tracker-mini`:
  ```
  git status --short -- .
  git diff -- .
  git diff --cached -- .
  git add -- .
  ```
- Before every commit and push, verify all staged paths belong to `tracker-mini`
- Never use `git add -A` or `git commit -a`
- Never force push or rewrite pushed history
- Never rebase or amend already pushed commits
- Never delete remote history
- Never mix unrelated changes in one commit
- Never commit credentials, passwords, tokens or device-specific secrets

## Testing

- **No automated test framework exists** in the current repository
- Development-machine tests cannot validate hardware integration
- Hardware tests must be run on the physical Mini Tracker
- When adding a test framework, use `pytest` (standard Python choice)
- Mocked tests must be reported as mocked, not as proof of hardware function

## Physical Device Access

- SSH to `192.168.1.115` (user `pi`, hostname `dsc-node02`)
- Read-only during onboarding; changes require explicit approval
- Never store credentials in repo, steering, or logs
- One process at a time per exclusive hardware interface (serial, GPIO)

## Staging Workflow

The physical staging installation may be used for hardware testing without requiring a separate Git branch. A committed version from the current branch may be deployed to staging.

Recommended layout on the device:
```
/home/pi/tracker-mini          ← stable installation
/home/pi/tracker-mini-staging  ← development testing
```

Before testing staging:
1. Record stable service state and current commit
2. Verify rollback procedure
3. Stop only the required service
4. Start staging version
5. Test and collect logs
6. Stop staging, restore stable
7. Verify normal operation

## Rollback

Before every physical deployment, record:
- Current branch and stable commit
- Deployment commit
- Services stopped/started
- Configuration files affected
- Rollback commands

Rollback is complete when: stable service runs, hardware responds, Dashboard reachable, no new errors in logs.

## Code Style

- Follow existing patterns (module-level functions, not classes for services)
- Reuse `services/logger.py` `log()` function
- Handle missing hardware gracefully (try/except, report via log)
- Keep routes thin; logic in services
- Persist only via `save_settings()` or explicit file writes

## Offline Considerations

Every network-dependent feature must handle:
- Offline behavior
- Retries and timeouts
- Stale/duplicate/delayed data
- Local persistence where needed
- Recovery after restart

## Deployment Rule

The Mini Tracker updater deploys only:
```
backend/
frontend/
```

Therefore every file required for a feature to operate at runtime must be under `backend/` or `frontend/`. This includes:
- Backend services, routes, config defaults
- Frontend JS, CSS, HTML changes
- Documentation under `frontend/help/docs/`

Configuration defaults must be code-defined in backend Python (not in `config/settings.json`). The updater never overwrites the device's settings file. Missing config sections must be handled with code defaults so the app starts immediately after update.

Development-only files (tests, specs, pytest.ini, requirements-dev.txt, .kiro/) may reside outside these directories.

## Documentation Updates

- Follow `DOCUMENTATION.md` rules
- Edit only `frontend/help/docs/` sources
- Never edit `frontend/help/site/`
- Update `docs/index.md` status table when adding documents
- Write in English

## References

- `AGENTS.md` — complete development rules
- `AI_HANDOFF.md` — current status and ownership
- `DOCUMENTATION.md` — documentation guidelines
