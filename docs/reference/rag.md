# RAG reference

The RAG subsystem indexes local material and injects it into teacher prompts.
Decisions: [ADR-0005](../adr/0005-rag-fastembed-numpy-sqlite.md). Usage
commands live in [how-to: index and search](../how-to/index-and-search-rag.md).

## Pipeline

```text
data/raw/  ->  loaders  ->  chunker  ->  embeddings  ->  vector store
                                                             |
pipeline engine  <-  retrieval with provenance  <-------------+
```

## Loaders

`rag/loaders.py` walks a directory (or single file) and reads text formats:
markdown, plain text, JSON (pretty-printed), diffs/patches and common source
extensions (`.py`, `.c`, `.js`, `.go`, ...). Binary and oversized files
(> 2 MB) are skipped. Every document gets a stable `doc_id`
(`sha1(path)[:12]`).

## Chunking

- Markdown splits on headings first, then paragraphs.
- Other text splits on blank-line paragraphs.
- Long sections are split to ~1200 characters with a 150-character overlap,
  cutting on line boundaries.
- Every chunk stores its byte span (`start`, `end`) and a sha256 `hash`.

## Embedding backends

| Backend | Dependency | Use case |
|---|---|---|
| `fastembed` (default) | fastembed (ONNX, CPU) | production indexing; `BAAI/bge-small-en-v1.5` |
| `hashing` | none | tests, CI, offline; deterministic token hashing |

Both return L2-normalized float32 vectors, so similarity is a dot product.
Select with `--backend fastembed|hashing`.

## Storage

- `rag/index/index.db` (sqlite): chunk rows
  `(chunk_id, doc_id, source, text, hash, start, end)`.
- `rag/index/vectors.npy`: the embedding matrix, row-aligned with chunk ids.
- Search is a numpy matmul (brute force); the store and backend interfaces are
  swappable (sqlite-vec optional extra and backend planned).

## Retrieval and provenance

`Retriever.search(query, k)` returns chunks with full provenance:

| Field | Meaning |
|---|---|
| `text` | chunk text |
| `source` | document path relative to the indexed root |
| `doc_id` | stable document identifier |
| `chunk_id` | `doc_id:sequence` |
| `hash` | sha256 of the chunk text |
| `score` | cosine similarity (rounded) |

## How the engine uses it

When `rag/index/index.db` exists (and `--rag` is on, the default), the engine
searches with the case description and injects numbered chunks into teacher
prompts. The chunk provenance is copied into the record's
`metadata.rag.chunks`, and the chunk texts feed the cache key (`cache_extra`),
so re-indexing invalidates affected cache entries. RAG context is
generation-only: it never enters student training inputs
([ADR-0007](../adr/0007-dataset-format-messages-metadata.md)).
