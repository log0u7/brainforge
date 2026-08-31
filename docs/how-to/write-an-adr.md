# How to write an ADR

This guide adds an Architecture Decision Record to `docs/adr/`. ADRs capture
*why* a decision was made, including the options that were rejected. Rules and
the full index live in [docs/adr/index.md](../adr/index.md).

## 1. Decide it is ADR-worthy

Write an ADR when the decision is:

- **architectural**: affects structure, boundaries, dependencies or data
  formats (a new provider type, a cache strategy, a dataset field);
- **hard to reverse**: changing it later costs real work;
- **contested**: reasonable people could pick a different option.

Not ADR-worthy: routine refactors, doc typos, dependency bumps without
behavior change.

## 2. Create the file

```bash
cp docs/adr/template.md docs/adr/00NN-short-title.md
```

Take the next free number; numbers are never reused. Use kebab-case titles.

## 3. Fill every section (MADR 3.0)

- **Status / Date**: `accepted` and today's date for new decisions.
- **Context and Problem Statement**: the forces at play, no solution talk.
- **Decision Drivers**: the 3-6 constraints that actually drove the choice.
- **Considered Options**: include the option that "obviously" won and the one
  that was tempting. This section is the ADR's value.
- **Decision Outcome**: the chosen option and the justification.
- **Consequences**: honest costs, not just benefits, plus mitigations.
- **Confirmation**: the tests, CLI commands or config fields that prove the
  decision still holds. An ADR whose confirmation cannot be written is a wish,
  not a decision.
- **Pros and Cons of the Options**: per option, at least one good and one bad.
- **Links**: related ADRs, doc pages, test files.

Style: English only, no em dashes, under ~150 lines.

## 4. Register it

1. Add a row to the table in `docs/adr/index.md`.
2. Add the page to the `Decision records` section of `mkdocs.yml` (or rely on
   the nav entry pattern of the other ADRs).
3. Link it from the affected reference or explanation pages.

## 5. Review and merge

Open the MR with the `docs` label. Reviewers check that the rejected options
are real alternatives (not strawmen) and that the confirmation section points
at things that exist.

## 6. Superseding a decision

Never edit a superseded ADR's content into the new decision. Instead:

1. Write the new ADR with the new decision.
2. Set the old ADR's status to `superseded by
   [ADR-00NN](00NN-....md)`.
3. Keep the old file for history.

## Example in this repo

[ADR-0005](../adr/0005-rag-fastembed-numpy-sqlite.md) is a good model: four
considered options with honest cons (including the chosen one), a confirmation
pointing at `tests/unit/test_rag.py` and a CLI command, and links to the
reference page it backs.
