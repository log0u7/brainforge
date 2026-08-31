# 0005 - Local RAG with fastembed embeddings and a numpy/sqlite store

* Status: accepted
* Date: 2026-08-31

## Context and Problem Statement

Teachers need local context (CVE notes, framework docs, code conventions)
injected into their prompts. The index must run on the workstation next to the
training GPU, on CPU, without competing for VRAM, and the retrieval must carry
provenance (which document and chunk supported which statement).

## Decision Drivers

* Zero GPU footprint: embeddings run on CPU while the 3080 trains.
* Minimal dependency weight; the core must install fast in CI.
* Full provenance per chunk (source, doc, hash, score) is a hard requirement
  (ADR-0007).
* Scale target is thousands of chunks, not millions.

## Considered Options

* sentence-transformers + torch, brute-force numpy
* ChromaDB
* FAISS
* Cloud embedding APIs
* fastembed (ONNX) + numpy over sqlite metadata, with a hashing fallback

## Decision Outcome

Chosen option: "fastembed (ONNX) + numpy over sqlite metadata, with a hashing
fallback". `rag/embeddings.py` exposes a swappable `EmbeddingBackend`;
`FastembedBackend` runs `BAAI/bge-small-en-v1.5` as ONNX on CPU, and
`HashingBackend` is a deterministic zero-dependency fallback used by tests and
CI. `rag/store.py` keeps chunk rows in sqlite (`rag/index/index.db`) and the
vector matrix in `rag/index/vectors.npy`; similarity is an L2-normalized dot
product. The interface leaves room for a sqlite-vec backend (optional extra)
when scale demands it.

### Consequences

* No torch in core dependencies: `uv sync` stays light and CI embeds nothing
  heavy.
* Brute-force similarity is O(n) per query; fine at the target scale, and the
  backend/store interfaces are the upgrade path.
* The first fastembed run downloads the model (~30 MB); offline environments
  use the hashing backend.

### Confirmation

* `tests/unit/test_rag.py` covers loaders, chunking, store roundtrip and
  provenance fields using `HashingBackend`.
* `brainforge rag search --backend hashing` works with no model download.

## Pros and Cons of the Options

### sentence-transformers + torch

* Good, because the model zoo is larger.
* Bad, because torch drags hundreds of MB into core deps and wants GPU memory
  the training run needs.

### ChromaDB

* Good, because indexing and search are batteries-included.
* Bad, because a database engine (with its own deps) replaces ~200 lines of
  sqlite+numpy, and provenance schemas become Chroma's.

### FAISS

* Good, because it scales to millions of vectors.
* Bad, because building/linking native FAISS in CI is painful for a scale we
  do not have.

### Cloud embedding APIs

* Good, because quality is high and no local compute is needed.
* Bad, because it breaks the local-first principle, leaks source documents to
  a third party and adds per-call cost.

### fastembed + numpy/sqlite (chosen)

* Good, because ONNX CPU inference is small, fast and deterministic enough;
  provenance stays in our schema.
* Bad, because brute-force search is linear; mitigated by the swappable
  backend/store interfaces (sqlite-vec listed in the roadmap).

## Links

* [RAG reference](../reference/rag.md)
* ADR-0007 (provenance in the dataset)
* `src/brainforge/rag/`
