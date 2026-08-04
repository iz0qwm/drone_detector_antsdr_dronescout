# Mini Tracker — Documentation Steering

## Authoritative Reference

All documentation rules are defined in `DOCUMENTATION.md` at the workspace root. Read it before creating or modifying any documentation.

## Key Rules Summary

- **Language**: English only
- **Source**: `frontend/help/docs/` (MkDocs Markdown)
- **Config**: `frontend/help/mkdocs.yml`
- **Generated**: `frontend/help/site/` — never edit
- **Style references**: `docs/index.md`, `docs/product-overview.md`, `docs/product-vision.md`, `docs/hardware/overview.md`
- **Images**: `frontend/help/docs/images/` with naming convention `<section>_<document>_<description>.png`

## When Modifying Documentation

1. Read the existing document first
2. Extend rather than replace
3. Keep diffs small and focused
4. Update `docs/index.md` status table when adding new documents
5. Cross-reference existing docs rather than duplicating content
6. Verify claims against the source code (implementation is truth)
7. Use Mermaid diagrams where they improve understanding

## Document Sections

| Section | Audience | Style |
|---------|----------|-------|
| Product | Decision makers, operators | High-level, operational |
| User Guide | Field operators | Workflow-oriented, no implementation details |
| Hardware | Integrators, maintainers | Product perspective, not "Raspberry Pi project" |
| Maintenance | Maintainers | Procedures and diagnostics |
| Developer | Developers | Architecture, APIs, implementation details |

## Do NOT

- Write blog-style or tutorial-style content
- Use marketing language
- Describe Mini Tracker as a "Raspberry Pi project"
- Create duplicate documents
- Reference screenshots that don't exist
- Edit generated files under `frontend/help/site/`

## MkDocs Build

Documentation is built outside the running application:
```
cd frontend/help
mkdocs build
```

The Flask backend serves the generated site from `frontend/help/site/` under `/help/`.
