# DOCUMENTATION.md

# Mini Tracker Documentation Guidelines

This document defines the documentation rules for the Mini Tracker project.

Always read this document before creating or modifying any documentation.

---
# Documentation Language

Mini Tracker documentation is bilingual when the task explicitly requests localized documentation.

The default and canonical documentation language remains **English**.

Italian documentation is allowed for operator-facing manuals, user guides, maintenance guides and other localized documentation when requested by Raffaello or when implementing the multilingual manual.

Language rules:

- English source pages remain the canonical technical reference unless an approved specification says otherwise.
- Italian source pages must be written in clear operational Italian, not as a literal word-for-word machine translation.
- A localized page must preserve the meaning, safety warnings, operational limitations, procedure order, UI labels and verified implementation facts of the canonical page.
- Code identifiers, API paths, configuration keys, filenames, commands, log labels and UI labels must remain exactly as implemented unless the UI itself is localized.
- Mermaid diagrams may be localized when they are part of an Italian page, but node labels must preserve the same operational meaning.
- Direct quotations, UI labels, protocol names, product names and proper names may remain in their original language when appropriate.

When a user request is written in Italian, do not automatically translate all documentation work into Italian. Use Italian only when the requested artifact is explicitly an Italian/localized manual page or part of the multilingual documentation set.

---

# Multilingual Documentation with mkdocs-static-i18n

Mini Tracker uses MkDocs Material for documentation and may use `mkdocs-static-i18n` for multilingual output.

Preferred multilingual structure:

```text
frontend/help/docs/
    index.md          # canonical/default English page
    index.it.md       # Italian localized page

    user/
        dashboard.md
        dashboard.it.md
```

Use the `mkdocs-static-i18n` **suffix** structure for localized pages unless Raffaello explicitly approves a different structure.

For Italian localized pages:

- create the Italian page next to the canonical page using the `.it.md` suffix;
- keep the same directory location as the canonical page;
- keep screenshots and shared image paths reusable whenever the UI image is still valid;
- create localized screenshots only when the image itself is different or the Italian page needs a different visual reference;
- do not delete, rename or replace the canonical English page;
- do not create duplicate unsuffixed Italian pages;
- do not move the documentation tree into language folders unless this is explicitly approved.

`frontend/help/mkdocs.yml` may be updated to configure `mkdocs-static-i18n`, language metadata, navigation translations, search language settings and theme language settings.

When configuring `mkdocs-static-i18n`:

- keep English as the default/root language unless Raffaello explicitly requests Italian as the root language;
- configure Italian as locale `it`;
- prefer `fallback_to_default: true` during incremental translation so untranslated pages still resolve to the canonical English content;
- localize navigation labels through the plugin configuration rather than by changing canonical file paths;
- keep navigation paths relative to `frontend/help/docs/`;
- do not add a dependency unless it is needed for the approved multilingual documentation workflow.

Kiro and Codex must not compile the documentation with MkDocs unless Raffaello explicitly changes the current workflow. Raffaello manually runs MkDocs and verifies generated output.
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
index.it.md

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
    dashboard.it.md
    maps.it.md
    ...

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

For Italian localized pages, use the corresponding English canonical page as the structure and technical reference, then write natural Italian text with the same official product tone.

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

# Documentation Images

Documentation screenshots are stored under:

docs/images/

The directory structure mirrors the documentation structure.

Example:

images/
    user/
        dashboard/
        maps/
        traffic-monitoring/
        mission-planning/


Screenshot filenames must follow this convention:

<section>_<document-name>_<description>.png

Examples:

user_dashboard_overview.png
user_dashboard_map_view.png
user_dashboard_services.png

user_maps_overview.png
user_maps_download_manager.png

user_traffic-monitoring_traffic_sources.png

Reuse existing screenshots whenever possible.

Before creating a new screenshot, verify whether an appropriate image already exists.

When updating documentation, insert screenshots near the section they describe using standard Markdown image syntax.

Example:

![Dashboard Overview](../images/user/dashboard/user_dashboard_overview.png)

Add a short explanatory caption when the screenshot highlights an important interface element.

Screenshots are considered part of the documentation.

When updating an existing document, inspect the corresponding images directory and use available screenshots whenever they improve readability.

Never reference screenshots that do not exist.

Do not create placeholder image references.

Images should complement the text rather than replace it.

Localized pages may reuse existing screenshots when the interface shown is unchanged. Use localized image filenames only when the image content is different for that language.

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

When creating a localized `.it.md` page for an existing canonical page, update `docs/index.md` only if the Documentation Status table includes language-specific status. If the table tracks only canonical documents, do not mark unrelated entries complete only because a localized page was added.

---

# Output

When asked to generate documentation:

* create only the requested Markdown file or localized Markdown file
* preserve the documentation structure
* do not modify unrelated files unless explicitly requested
* update `docs/index.md` only when the status table actually tracks the created canonical or localized document
* do not run MkDocs unless Raffaello explicitly changes the current manual-build workflow
