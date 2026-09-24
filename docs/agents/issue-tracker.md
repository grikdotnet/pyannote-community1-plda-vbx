# Issue tracker: Local Markdown

Issues and specs for this repo live as Markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`.
- Implementation issues are separate files at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`.
- Record triage state in a `Status:` line near the top of each issue file, using the roles in `triage-labels.md`.
- Append conversation under `## Comments`.

When a skill says to publish an issue, create the corresponding file under `.scratch/<feature-slug>/`. When it says to fetch a ticket, read the referenced issue file.

## Wayfinding

- Map: `.scratch/<effort>/map.md`, containing Notes, Decisions-so-far, and Fog.
- Child ticket: `.scratch/<effort>/issues/<NN>-<slug>.md`, with a `Type:` of `research`, `prototype`, `grilling`, or `task`, and a `Status:` of `claimed` or `resolved`.
- Record dependencies as `Blocked by: NN, NN`. A ticket becomes available when every listed dependency is resolved.
- Find the first open, unblocked, unclaimed ticket by number. Claim it by saving `Status: claimed` before work.
- Resolve it by adding an `## Answer`, setting `Status: resolved`, and linking the decision from the map.
