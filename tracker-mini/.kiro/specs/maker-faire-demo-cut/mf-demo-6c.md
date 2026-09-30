# MF-DEMO-6C — Replay source and explicit operational modes

## Authority and scope

Implement the Product Owner's MF6-C request and MF6 revision-2 Design Lock. The newer product rule supersedes recording/import presentation in the earlier lock: LIVE records, DEMO operates the rescue scenario, REPLAY imports and plays validated files. Preserve the developer DEMO capture API. No Mini Tracker runtime, domain core/profile/geometry, offline package, deployment, commit or push changes.

## Requirements and design

- Parent owns exactly LIVE/DEMO/REPLAY, visible even in compact 2D/Google panels. Only the selected mode's actions can execute. Active and unsaved recording gates prevent silent data loss.
- Pure reusable driver consumes frozen validated snapshots, READY frame zero, monotonic 1x timing, ordered equal-time frames/static tail, bounded 100-frame/8-ms tasks, visible backlog pause, hidden-tab pause, restart generations and stale callback protection.
- Replay bypasses LIVE subscriptions, motion composition, freshness timers and domain evaluation. Recorded awareness, freshness, source provenance and narrative stay immutable.
- Shared renderers use playback presentation timing. Reset histories/revision guards on generation change, preserve camera and IDs, settle terminal pose before acknowledgment. Bridge v2 validates exact origin/window/channel/version/generation. Healthy consumers have a 2-second ACK deadline; unavailable or closed Google never blocks 2D.
- Explicit exit clears private runtime and views, restores only prior source choice and waits for explicit operational reopen. Context/auth loss clears imports/recordings as before.

## Failures and compatibility

Invalid import preserves the accepted session. Google loader/model failure retains the existing fallback and removes failed consumer obligations. No nested recordings or historical freshness evaluated against current UTC. No seek/speed controls. Preserve canonical MF6-A/B fixture, approved mixed LIVE-source DEMO overlays, rescue route/timing and scale 4/6 models. Existing Google LIVE altitude limitation remains explicit.

## Implementation and verification

- [x] Driver, shared presentation clock and version-2 bridge.
- [x] Parent ownership, lifecycle, mode-specific controls and persistent badges.
- [x] Focused clock/mode/isolation/renderer/ACK tests and all requested regressions.
- [x] Actual browser review, canonical full playback, performance and limitations.
- [x] Implementation report and handoff; list bounded cleanup candidates for later MF-DEMO-POLISH after C/D acceptance.

Status: **MF-DEMO-6C — REPLAY + MODE SEPARATION IMPLEMENTED**. Ready for broader MF6-D acceptance; PO acceptance pending.

Verification: 517 passed (174 focused MF6 tests including 36 new cases, plus 343 retained regressions), zero failed/cancelled/skipped. Syntax checks passed on 19 JavaScript files. Instrumentation proves zero source/motion/core ingestion/evaluation during replay. Canonical 62-frame playback reached terminal sequence 61 with acknowledgments from both 2D and native Google; scales 4/6 and the versioned 600-to-35 m descent were retained. Actual LIVE record/export/import/replay workflow used a new 187-frame synthetic-source recording. Pause/resume, restart/loop, compact badges, Google close/reopen and explicit exit/fresh LIVE reopen were reviewed.

Report: `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_MF_DEMO_6C_REPLAY_IMPLEMENTATION_20260923.md`. It contains the complete 24-file DSC inventory, exact test command, browser measurements and deferred polish candidates. Browser evidence uses a local synthetic-source host with production modules and actual Google, not authenticated deployment or Mini Tracker/RF hardware. Existing LIVE Google altitude limitations remain. No offline package, deploy, commit, push, dependency additions, Mini Tracker runtime edits or MkDocs build.
