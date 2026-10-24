# CTF corpus ingestion

Builds BrainForge CTF cases from the local CTF corpus
(`~/projets/datasets/ctf-corpus`). The corpus is a shared local store:
augure (fact bases) and lictor (lab targets, exploits) can consume the
same material.

```sh
uv run python experiments/ctf_ingest/ingest.py
```

Writes `examples/cases/ctf-*.json` (committed) and
`experiments/ctf_ingest/manifest.jsonl` (committed, one line per case
with its verification status).

## Tiers

| Tier | Sources | expected_flag | Gate behavior |
|---|---|---|---|
| `verified` | lictor lab targets (6), ROP Emporium (6 of 8) | known | judge flag must match `metadata.expected_flag`; mismatch rejects |
| `unverified` | CTF-Archives 2026 dumps | absent, `flag_format` only | standard judge gate only; the flag is not checkable |

`callme` and `ret2csu` (ROP Emporium) ship an `encrypted_flag.dat`: the
plaintext flag is not locally verifiable, so they are skipped until a
solved run records their flags.

## Writeup policy

Writeups are verification references only. They set `expected_flag`
and technique expectations; no generated case embeds writeup text in
its inputs. Teachers analyze the challenge material alone, which keeps
the dataset on the analysis side of the analysis/recitation line.

## Provenance and licensing

Every case carries `source{type, repository, path, date, url}`:

* lictor lab: our own material, dated at corpus ingestion (post-cutoff).
* ROP Emporium: dated 2020-01-01 (pre-cutoff on purpose), so
  `recitation_risk` tagging marks it; the binaries are site-distributed,
  used locally, never republished by this repo.
* CTF-Archives (2026 Zeekr Cybersecurity Quals): release date
  2026-09-19, post-cutoff holdout material, public dump.

Datasets generated from these cases stay local (see the README status
section): nothing here republishes challenge binaries or writeups.
