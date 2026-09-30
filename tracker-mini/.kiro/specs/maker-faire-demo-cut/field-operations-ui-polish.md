# Field Operations — Final UI polish / rehearsal pass

Authority: Product Owner request of 23 September 2026. This bounded presentation/lifecycle addendum supersedes MF6-C's explicit Replay exit control. Existing feature architecture and MF6 recording/playback contracts remain locked.

## Acceptance criteria and design

- X closes Field Operations in LIVE, DEMO and REPLAY. No explicit Replay exit button in either renderer. The former body-level Chiudi duplicate is removed.
- Once loading/accepted, REPLAY locks LIVE/DEMO mode changes in the panel, Field Nodes selector and controller. EMPTY permits choosing another mode. X disposes the replay driver, cancels callbacks/timers, invalidates generation, clears imported session, scenes and interpolation, and closes Google. A new driver is created for a later Replay session; no source automatically reconnects before explicit reopen.
- Google panel X delegates to the same owner close lifecycle via the existing exact-origin/window/version bridge. Chiudi 3D and Escape retain their existing view-only return to 2D and do not stop Replay.
- Persistent mode identity, concise paused/completed replay status, recording duration visible when compact; no frame/byte counters in normal recorder UI. Keep source provenance and useful failures.
- Mode controls precede Traffic Awareness and source summaries; compact 2D also retains the selected awareness summary. Labels remain OFF. Preserve model options, scales, camera and fallback.
- Italian DEMO state labels and consistent buttons, segmented selector with pressed outline/underline, dark/light styling and responsive bounds. Before import, show only the mode selection and load control.

## Affected components and compatibility

DSC production changes are limited to Field Operations controls/map/controller/bridge, presentation helpers and Google view/CSS. Replay source changes only its visible terminal error sentence. No domain calculator, profile, geometry, recording/import format, limits, playback clock, route/timing, model assets/scales, transport, service or deployment changes. No new dependencies. Mini Tracker runtime remains untouched.

Failure handling remains the MF6 contract: invalid imports preserve the accepted recording, import cancellation is atomic, old callbacks cannot restore a closed scene, offline Google leaves 2D available, and access/context loss clears private runtime. Unsaved recorder gates and discard behavior remain intact.

## Verification tasks

- [x] Implement X-only replay close, mode guards and disposed-driver renewal.
- [x] Add lifecycle, mode-control, compact-header and Google bridge tests; update expectations for approved copy/layout changes.
- [x] Run MF6 and requested regression suites.
- [x] Complete final browser rehearsal and dark/light checks at 390/768/1280 pixels.
- [x] Record exact results, file inventory, UI cleanup review table and limitations in the requested DSC+ report; update handoff.

Status: **FIELD OPERATIONS UI POLISH — IMPLEMENTED**. Full requested regression: 530 passed, zero failed/cancelled/skipped/todo (517 retained + 13 new). Final copy follow-up: 37 affected tests passed. Syntax: 18 changed JavaScript files passed, final lifecycle/control edits checked again; scoped whitespace checks passed.

Actual Chromium rehearsal completed DEMO and all 62 canonical Replay frames; 2D and native Google acknowledged terminal sequence 61. Restart/loop advanced through another complete generation. Both 2D X and Google Field Operations X stopped Replay; imported runtime was cleared. The browser exposed an already-exported recorder buffer reappearing as unsaved after file-state clear; Replay close now releases that saved memory buffer and its dedicated regression plus browser retest pass. No disk file is deleted. Reopen permits LIVE/DEMO/EMPTY Replay without previous playback.

Native filechooser automation was unavailable in the integrated browser. A test-host-only canonical-file button exercised the unchanged actual import worker and playback path; native OS picker remains a manual check. Synthetic source/access callbacks are not evidence of authenticated production or hardware acceptance. Full inventory, UI keep/remove/rename table, browser observations and limitations: `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_FIELD_OPERATIONS_UI_POLISH_20260923.md`.

No review gate was requested for this bounded pass. No commit, push, deployment or MkDocs.
