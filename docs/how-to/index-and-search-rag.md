# How to index and search with RAG

This guide indexes your local material (CVE notes, framework docs, code
conventions) into the vector store and queries it. It assumes a working
install and documents under a directory such as `data/raw/`.

## 1. Choose a backend

| Backend | When |
|---|---|
| `fastembed` (default) | real indexing; ONNX on CPU, downloads the model once (~30 MB) |
| `hashing` | offline environments, CI, quick sanity checks; no download |

## 2. Index a directory

```bash
uv run brainforge rag index data/raw
```

Expected output: `indexed N documents, M chunks -> rag/index`.

What counts as a document: markdown, plain text, JSON, diffs/patches and
common source extensions under the target path (binary and >2 MB files are
skipped).

Options: `--backend fastembed|hashing`, `--index-dir rag/index`.

## 3. Search

```bash
uv run brainforge rag search "CWE-78 subprocess" --k 5
```

Expected: a table with score, source, chunk id and text preview.

## 4. Verify provenance

Each result carries `source` (document path), `doc_id`, `chunk_id`, `hash`
(sha256 of the text) and `score`. The same provenance is copied into every
record generated while that index is active, under
`metadata.rag.chunks` ([dataset format](../reference/dataset-format.md)). To
trace a teacher claim back to its support, search for the claim keywords and
match the chunk hash.

## 5. Use it in a pipeline run

Indexing is enough: `pipeline run` uses `rag/index` automatically when
present. Controls:

```bash
uv run brainforge pipeline run security_dataset --no-rag        # disable
uv run brainforge pipeline run security_dataset --rag-backend hashing
```

Note: chunk texts feed the LLM cache key, so re-indexing invalidates affected
entries ([ADR-0009](../adr/0009-sqlite-response-cache.md)).

## 6. Rebuild after changing sources

The index command clears and rebuilds:

```bash
uv run brainforge rag index data/raw
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `unknown embedding backend` | typo in `--backend` | use `fastembed` or `hashing` |
| `no results (empty or missing index?)` | no `rag/index/index.db` | run the index command |
| slow first index | model download | one-time; hashing for offline |
| irrelevant results | docs too homogeneous or query too vague | check `data/raw` content; narrow the query |

## Next steps

- [Generate a dataset](generate-dataset.md) with RAG active.
- [RAG reference](../reference/rag.md) for chunking and storage details.
