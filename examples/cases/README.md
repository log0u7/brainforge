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

Two families ship today:

- **Security cases** (`case-*.json`): command injection (2026-08,
  post-cutoff), SQL injection (2026-04, pre-cutoff), stored XSS (2026-07,
  post-cutoff), spanning the security teacher knowledge cutoff of 2026-06.
- **Ticket cases** (`ticket-*.json`): IT helpdesk triage examples (password
  reset, VPN outage, invoice question) for the `ticket_triage` domain,
  showing the framework is domain-agnostic.

Run them with the mock provider at zero cost:

```bash
uv run brainforge pipeline run security_dataset --input examples/cases --provider mock
uv run brainforge pipeline run ticket_triage --input examples/cases/ticket-vpn-down-001.json --provider mock
```
