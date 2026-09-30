# MF-DEMO-6B — local JSON export/import

## Authority and scope

Implement the approved MF-DEMO-6 revision 2 and the explicit MF6-B request, consuming MF6-A unchanged contracts. Add stopped-chunk local export, a single bounded worker validator, immutable in-memory accepted session and compact Field Operations record/save/load controls. No replay/source replacement, offline package, deployment, commit, push or Mini Tracker runtime work.

## Requirements and acceptance

- Export finalized chunks with bounded metadata, deterministic privacy-safe UTC/UUID filename, JSON MIME, bounded Object URL cleanup and retryable retained buffer.
- Import only selected local File, size-gate before read; byte lexical scan before full decode/parse, fatal UTF-8, duplicate/prototype/credential key rejection, depth/token/string/byte budgets.
- Reuse V1 projection and awareness validation; check all frames, metadata, versions, recorded profiles/scenario, same-time semantic duplicates and awareness run/revision progression. Unknown bounded optional fields are omitted.
- Worker deadline 15 s; request generations cancel/terminate obsolete work. Only fully validated sanitized/frozen objects reach an accepted session. Errors preserve LIVE/DEMO, current map, recorder and previous imported session.
- UI recording indicator survives compact mode, unsaved-buffer Save/Discard gate, safe text-only status, normal close retains buffer; authorization loss clears buffer/session/worker.
- Generate synthetic canonical fixture from actual MF6-A recorder: 62 frames/60000 ms and terminal COMPLETE/SUSPENDED/WARNING/INSIDE/0 m. Exact semantic round trip through real worker.

## Components and failure model

Extend recorder with stopped serialized export parts; new export/import coordinator, shared worker validator/worker and controls modules. Narrow parent/map/CSS/script integration. No calculation/profile/geometry/route/Google renderer changes. Offline will reuse the validator source rather than duplicate it. Browser download initiation is not disk-persistence proof. Process OOM is not recoverable by schema validation; temporary allocations and main-thread transfer/freezing must be measured separately.

## Verification tasks

- [x] Implement chunk export and bounded lexical/schema/worker pipeline.
- [x] Integrate owner lifecycle and compact recording controls, with atomic unsaved-buffer protection.
- [x] Add hostile/security/cancellation/round-trip/UI tests and generated synthetic canonical fixture.
- [x] Run all requested regression suites and scope/syntax checks.
- [x] Actual-browser export/import, malformed samples, compact layout and performance of canonical/large bounded inputs; document environment/limitations.
- [x] Complete implementation report and development handoff.

Status: **MF-DEMO-6B — JSON EXPORT / IMPORT IMPLEMENTED**. Ready for MF6-C Replay source + Play/Pause/Restart/Loop; no playback or offline package implemented.

481 tests passed, zero failures/cancellations/skips: 138 across four focused suites (33 format, 24 recorder, 67 import, 14 controls) plus 343 requested existing regressions. 82 tests added relative to MF6-A; all 399 prior tests retained. Syntax passed on 15 JavaScript files; scoped whitespace/scope checks passed.

Canonical fixture is generated from actual recorder chunks: 62 frames, 60000 ms, 334415 bytes, full semantic equality after real-worker import. Filename uses the injected historical UTC: DSC_Field_Operation_20330518_033320_UTC_12345678.json. Browser actual 60 s scenario run exported 63 frames/77240 ms/339518 bytes (READY prelude plus static tail); disk file verified and selected through native picker for successful import. Source stayed DEMO. Six hostile browser samples rejected atomically. Compact indicator and 390 px layout verified.

Final Chromium 153 / Windows x64 / Ryzen 5 8645HS measurement: canonical native-worker import 127.2 ms total (66.8 ms worker); 16504483-byte valid file 1659 ms total (1309.6 ms worker), zero observed >=50 ms main-thread tasks after sanitized batch transfer. Exact 32 MiB hostile token input rejected by actual Node worker in 444.2631 ms. No heap guarantee, deployed-account/Google visual acceptance or hardware evidence is claimed. Browser host uses isolated synthetic sources and real application/Leaflet/Worker modules; host stopped after verification.

Authoritative report: `C:\PROVA\DRONI\CORSI-PROGETTI\DSC+\DSC_PLUS_MF_DEMO_6B_EXPORT_IMPORT_IMPLEMENTATION_20260923.md`. It lists 12 new/7 modified DSC files, full tests/commands, browser evidence, performance methodology and limitations. No deploy/commit/push/dependencies/Mini Tracker runtime edits/MkDocs.
