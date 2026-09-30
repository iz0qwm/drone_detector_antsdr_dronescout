# MF-DEMO-5C-1 — implementation contract

The Product Owner explicitly approved implementation on 2026-09-21. The authoritative requirements, acceptance criteria, design, affected components, failure modes, compatibility and test strategy are the 25 sections of [the approved Design Lock](<C:/PROVA/DRONI/CORSI-PROGETTI/DSC+/DSC_PLUS_GEOAWARENESS_5C1_DESIGN_LOCK_20260921.md>).

Implementation tasks:

- Implement the immutable profile and pure spherical geometry modules.
- Implement the bounded observation cache, independent freshness, relation lifecycle and immutable snapshots.
- Adapt existing logical scenes and deterministic motion samples at the parent ownership seam.
- Exercise every section 20 matrix case and the independent 60-second golden oracle.
- Run the nine requested regression suites, measure recurring evaluation and geometry preparation separately, and report limitations.

Only DSC Field Operations implementation files, new tests, this development specification, the development handoff and the requested external implementation report are in scope. Mini Tracker runtime, existing renderers, transports and visual scenario remain unchanged. No deployment, commit or push is authorized. Contradictions block only the affected requirement and must be reported explicitly.

## Implementation outcome

Implemented on 2026-09-21: 68 focused tests and 182 required regressions passed, with no failures or skips. All 53 matrix identifiers are covered. Raffaello resolved the V03 scope conflict by explicitly authorizing the minimum 3D receiver adjustment; the receiver validates/rejects wire data and never computes awareness. Actual renderer and bridge implementations remain unchanged.

See [the implementation report](<C:/PROVA/DRONI/CORSI-PROGETTI/DSC+/DSC_PLUS_GEOAWARENESS_5C1_IMPLEMENTATION_20260921.md>) for the 20 requested delivery sections, exact commands and performance measurements. Normal demo workload met the measured local budget. Maximum-density preparation and some combined evaluation cycles exceed 50 ms in Node; representative Edge performance/visual acceptance remains unverified and is explicitly not inferred from unit tests.
