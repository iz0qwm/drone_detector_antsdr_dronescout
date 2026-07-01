# DOCUMENTATION.md

# Mini Tracker Documentation Guidelines

This document defines the documentation rules for the Mini Tracker project.

Always read this document before creating or modifying any documentation.

---

# Documentation Location

The documentation source is located here:

```
frontend/help/docs/
```

The MkDocs configuration is located here:

```
frontend/help/mkdocs.yml
```

The generated HTML documentation is located here:

```
frontend/help/site/
```

Never edit files inside **site/**.

Always modify files inside **docs/** and let MkDocs generate the HTML output.

---

# Documentation Structure

The documentation is organized as follows.

```
docs/

index.md

product-overview.md
product-vision.md
release-notes.md
glossary.md

user/
    installation.md
    first-start.md
    dashboard.md
    maps.md
    traffic-monitoring.md
    mission-planning.md
    teams.md
    settings.md
    troubleshooting.md
    faq.md

hardware/
    overview.md
    power.md
    networking.md
    gps.md
    ads-b.md
    remote-id.md
    meshtastic.md
    raspberry-pi.md

maintenance/
    backup.md
    restore.md
    update.md
    diagnostics.md
    logs.md

developer/
    architecture.md
    backend.md
    frontend.md
    services.md
    api.md
    mission-storage.md
    coding-guidelines.md
    roadmap.md
```

---

# Documentation Style

All documentation must follow the writing style already used in the existing documents.

Use the existing documentation as the reference before creating new files.

The primary style references are:

* docs/index.md
* docs/product-overview.md
* docs/product-vision.md
* docs/hardware/overview.md

Maintain the same writing style, structure, terminology and formatting.

---

# Writing Rules

Documentation must be written as official product documentation.

Write for:

* operators
* maintainers
* integrators
* developers

Do not write blog articles.

Do not write tutorials unless the document specifically requires them.

Avoid marketing language.

Avoid implementation details unless the document belongs to the Developer section.

Explain operational concepts before technical details.

Use clear section headings.

Use Markdown tables where appropriate.

Use Mermaid diagrams whenever they improve understanding.

---

# Hardware Documentation

Hardware documentation describes Mini Tracker as an integrated product.

Do not describe Mini Tracker as a Raspberry Pi project.

Avoid low-level electronics unless required for maintenance or integration.

Describe hardware from the perspective of:

* operational role
* installation
* integration
* maintenance
* troubleshooting

---

# User Documentation

User documentation describes operational workflows.

Describe what the operator sees and how the operator performs common tasks.

Avoid discussing backend implementation.

---

# Developer Documentation

Developer documentation may describe:

* software architecture
* backend
* frontend
* APIs
* services
* storage
* coding practices

Developer documentation should remain implementation-oriented.
Developer documentation should explain the project from high-level architecture down to implementation details.

Each document should build upon the previous one.

Avoid repeating the same concepts in multiple documents.

Cross-reference previously documented architecture whenever appropriate.

---

# Mermaid

Use Mermaid diagrams whenever useful.

Generate valid Mermaid syntax compatible with MkDocs Material.

Prefer:

* flowchart
* sequenceDiagram
* stateDiagram
* classDiagram

Keep diagrams readable and reasonably compact.

---

# Existing Documentation

Before creating or modifying documentation, verify whether the corresponding document already exists.

If the document already exists:

- extend it rather than replacing it;
- update only the sections affected by the implementation changes;
- preserve its writing style, structure and formatting;
- prefer editing existing sections over creating new ones;
- never move information between documents unless necessary;
- keep the generated diff as small as possible;
- avoid unnecessary documentation churn;
- do not create duplicate documents.

Every documentation change should be directly justified by a verified change in the project implementation.

Treat documentation updates as code maintenance, not content rewriting.
---

# Project Inspection

Before writing documentation for an existing subsystem, inspect the current project implementation.

The source code is the primary source of technical truth.

Use the existing documentation only as a writing style reference.

When documenting a subsystem, inspect the relevant backend, frontend, configuration files and existing documentation to understand how the subsystem actually works.

If implementation and documentation differ, prefer the current implementation.

Do not invent features, hardware capabilities or workflows that cannot be confirmed by inspecting the project.

If information cannot be verified, omit it rather than guessing.

---

# Updating index.md

Whenever a new documentation file is created:

1. Open `docs/index.md`.

2. Locate the **Documentation Status** table.

3. Update the corresponding entry.

Change:

```
Planned
```

to

```
Complete
```

for the newly created document.

Do not modify the status of unrelated documents.

---

# Output

When asked to generate documentation:

* create only the requested Markdown file
* preserve the documentation structure
* do not modify unrelated files unless explicitly requested
* always update `docs/index.md` to reflect the completion status of the newly created document
