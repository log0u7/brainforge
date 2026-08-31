# Local RAG

The RAG subsystem indexes local material and feeds it into teacher prompts as
context. It is deliberately lightweight: no torch, no heavy vector database.

## Pipeline

```text
data/raw/  ->  loaders  ->  chunker  ->  embeddings  ->  vector store
                                                             |
pipeline engine  <-  retrieval with provenance  <-------------+
```

## Loaders

`rag/loaders.py` walks a directory (or a single file) and reads text formats:
markdown, plain text, JSON (pretty-printed into text), diffs/patches and common
source-code extensions. Binary and oversized files (2 MB cap) are skipped.

## Chunking

Markdown files split on headings first; everything else splits on blank-line
paragraphs. Long sections are further split to ~1200 characters with a 150
character overlap, cutting on line boundaries. Every chunk stores its byte
span and a sha256 hash.

## Embeddings

| Backend | Dependency | Use |
|---|---|---|
| `fastembed` (default) | fastembed (ONNX, CPU) | production indexing, `BAAI/bge-small-en-v1.5` |
| `hashing` | none (built-in) | tests, CI, offline demos; deterministic token hashing |

Both return L2-normalized vectors, so similarity is a dot product.

```bash
uv run brainforge rag index data/raw --backend fastembed
uv run brainforge rag search "CWE-78 subprocess" --k 5
```

## Storage

`rag/index/index.db` (sqlite) holds chunk rows (text, source, doc id, chunk id,
hash, span); `rag/index/vectors.npy` holds the matrix. Search is a numpy
matmul; the interface is swappable (sqlite-vec planned, see ROADMAP).

## Provenance

Every retrieved chunk carries `source`, `doc_id`, `chunk_id`, `hash` and
`score`. The pipeline engine copies these into the record metadata under
`metadata.rag.chunks`, so any teacher statement can be traced back to the exact
indexed text that supported it.

## Design note

RAG context is used for **generation quality only**: the student training input
is the case itself, so the student learns to analyze code it is given, not to
query a retriever.
