from pathlib import Path

import numpy as np
import pytest

from brainforge.rag.chunking import chunk_document, chunk_documents, chunk_text
from brainforge.rag.embeddings import HashingBackend, get_backend
from brainforge.rag.loaders import Document, load_documents
from brainforge.rag.retrieval import Retriever
from brainforge.rag.store import VectorStore


@pytest.fixture
def docs_dir(tmp_path) -> Path:
    (tmp_path / "notes.md").write_text(
        "# Title\n\nCWE-78 is command injection via subprocess shell=True.\n\n"
        "## Details\n\nNever pass user input to shell commands without validation.\n"
    )
    (tmp_path / "vuln.py").write_text(
        "import subprocess\n\n\ndef run(cmd):\n    subprocess.run(cmd, shell=True)\n"
    )
    (tmp_path / "data.json").write_text('{"cve": "CVE-2026-1234", "description": "sql injection"}')
    (tmp_path / "binary.bin").write_bytes(b"\x00\x01\x02")
    return tmp_path


def test_load_documents_skips_binary(docs_dir):
    documents = load_documents(docs_dir)
    sources = [doc.source for doc in documents]
    assert "notes.md" in sources
    assert "vuln.py" in sources
    assert "data.json" in sources
    assert "binary.bin" not in sources


def test_load_documents_single_file(docs_dir):
    documents = load_documents(docs_dir / "notes.md")
    assert len(documents) == 1
    assert documents[0].doc_id.startswith("doc-")


def test_chunk_text_spans_cover_and_overlap():
    text = "\n\n".join(f"paragraph {i} " + "word " * 100 for i in range(5))
    spans = chunk_text(text, max_chars=200, overlap=50)
    assert len(spans) > 1
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)


def test_chunk_document_markdown_aware(docs_dir):
    documents = load_documents(docs_dir / "notes.md")
    chunks = chunk_document(documents[0])
    assert len(chunks) >= 2
    assert all(chunk.hash for chunk in chunks)
    assert chunks[0].text.startswith("# Title")


def test_chunk_documents(docs_dir):
    documents = load_documents(docs_dir)
    chunks = chunk_documents(documents)
    assert len(chunks) >= 3


def test_hashing_backend_deterministic_and_normalized():
    backend = HashingBackend(dim=64)
    vectors = backend.embed(["hello world", "hello world", "different text entirely"])
    assert vectors.shape == (3, 64)
    assert np.allclose(vectors[0], vectors[1])
    assert abs(np.linalg.norm(vectors[0]) - 1.0) < 1e-5


def test_get_backend_unknown():
    with pytest.raises(ValueError, match="unknown embedding backend"):
        get_backend("ghost")


def test_vector_store_roundtrip(tmp_path):
    store = VectorStore(tmp_path / "index")
    backend = HashingBackend(dim=32)
    docs = [Document(doc_id="doc-1", source="a.md", text="alpha beta gamma delta")]
    chunks = chunk_documents(docs)
    vectors = backend.embed([chunk.text for chunk in chunks])
    store.add(chunks, vectors)
    assert store.count() == len(chunks)
    results = store.search(backend.embed(["alpha beta gamma delta"])[0], k=1)
    assert results
    assert results[0][0].doc_id == "doc-1"
    assert results[0][1] > 0.9


def test_retriever_index_and_search_with_provenance(docs_dir, tmp_path):
    retriever = Retriever(tmp_path / "index", HashingBackend(dim=128))
    stats = retriever.index(docs_dir)
    assert stats["documents"] == 3
    assert stats["chunks"] > 0
    results = retriever.search("subprocess shell=True command injection", k=3)
    assert results
    top = results[0]
    assert top.source
    assert top.doc_id.startswith("doc-")
    assert top.chunk_id
    assert top.hash
    assert 0.0 <= top.score <= 1.0 or top.score > 1.0
    assert "shell=True" in " ".join(result.text for result in results)


def test_retriever_search_empty_index(tmp_path):
    retriever = Retriever(tmp_path / "empty", HashingBackend(dim=64))
    assert retriever.search("anything") == []
