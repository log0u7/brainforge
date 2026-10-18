# How to ship a trained student to Ollama

Export a trained student as a quantized GGUF and register it with Ollama,
then optionally distribute the weights peer-to-peer with
[llmp2p](https://github.com/log0u7/llmp2p).

## Prerequisites

- A trained student: a run directory under `experiments/` (see the
  [training reference](../reference/training.md)).
- [llama.cpp](https://github.com/ggml-org/llama.cpp) cloned or installed
  (provides `convert_hf_to_gguf.py` and `llama-quantize`).
- Optional: the `ollama` binary on PATH.
- Optional: the `gguf` Python package for the conversion script
  (`pip install gguf`).

## 1. Export to GGUF

```bash
uv run brainforge train export \
  --model experiments/run-20261018-... \
  --format gguf \
  --quant Q4_K_M \
  --output models/export
```

This merges the adapter into `models/export/merged/`, converts to
`models/export/model-q4_k_m.gguf` and writes `models/export/Modelfile`:

```dockerfile
FROM ./model-q4_k_m.gguf
```

llama.cpp is located via `--llama-cpp-path`, the `LLAMA_CPP_DIR` env var,
or PATH; missing tools fail fast with an install pointer.

## 2. Register with Ollama

```bash
uv run brainforge train export \
  --model experiments/run-20261018-... \
  --format gguf --ollama triage-bot
ollama run triage-bot
```

With `--ollama <name>` the command runs `ollama create <name>` itself when
the binary is available (otherwise it reports `skipped` and you register
manually):

```bash
cd models/export && ollama create triage-bot -f Modelfile
```

## 3. Use it as a BrainForge teacher

Any OpenAI-compatible endpoint works, including Ollama's:

```json
{
  "providers": {
    "local": {
      "type": "local",
      "base_url": "http://localhost:11434/v1",
      "api_key": "not-needed"
    }
  }
}
```

Point a role's model at the `local` provider to distill from your own
fine-tuned model in a later pipeline run.

## 4. Distribute with llmp2p (optional)

llmp2p distributes model artifacts over BitTorrent and registers them with
Ollama locally:

```bash
llmp2p import --ollama models/export     # fetch-or-seed the GGUF, ollama create
llmp2p seed                              # keep seeding for peers
```

Peers pull with `llmp2p pull <ref>`; integrity is pinned by manifest and
per-file sha256 (see the llmp2p trust model). BrainForge does not call
llmp2p itself: the handoff is the GGUF directory, and the two tools stay
decoupled.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `llama.cpp not found` | tools not on PATH | pass `--llama-cpp-path` or set `LLAMA_CPP_DIR` |
| conversion script exits non-zero | missing `gguf` package or unsupported arch | `pip install gguf`; check the model is convertible |
| `skipped: ollama binary not found` | ollama not installed | install Ollama, then `ollama create` manually |
| student talks like the base model | adapter not merged before export | use `--format gguf` (it merges) or `train export --format merged` first |
