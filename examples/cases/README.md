# Example cases

Case files consumed by `brainforge pipeline run <name> --input examples/cases`
and by `brainforge teacher run --case <id> --cases-dir examples/cases`.

A case is a JSON document with:

- `source`: provenance (`type`, `repository`, `commit`, `path`, `date`).
  The `date` (ISO `YYYY-MM-DD`) drives contamination tagging: cases dated
  after the latest teacher `knowledge_cutoff` are the post-cutoff evaluation
  signal; older cases (or cases without a date) are tagged `recitation_risk`.
- `input`: `code` and/or `description` (plus optional `question`).
- `metadata`: free-form, typically `language`.

The three bundled cases span a teacher knowledge cutoff of 2026-06:
command injection (2026-08, post-cutoff), SQL injection (2026-04,
pre-cutoff), and stored XSS (2026-07, post-cutoff).
