# How to configure providers

This guide connects BrainForge to real LLM providers: OpenRouter and OpenCode
Zen for teachers, MLGW for local models. It assumes a working install
(`mise install && uv sync`) and the example config in `config/config.json`.

## 1. Collect API keys

| Provider | Where | Environment variable |
|---|---|---|
| OpenRouter | openrouter.ai (API keys) | `OPENROUTER_API_KEY` |
| OpenCode Zen | opencode.ai/auth (copy the API key) | `ZENCODE_API_KEY` |
| MLGW | your local gateway; dev key is `deadbeef` | `MLGW_API_KEY` (defaults to `deadbeef`) |

Put them in your shell profile or a gitignored `.env` loader. Never commit
them; gitleaks runs in CI and pre-commit.

## 2. Check the judge layout

The shipped config satisfies judge independence: security (deepseek) and
coding (qwen) teachers on OpenRouter, the general critic on MLGW, and the
judge (glm) on Zen, a different provider *and* family. If you change models,
keep that property:

- the judge must not share a provider with any teacher;
- the judge must not share a declared `family` with any teacher.

Why: [judge correlation](../explanation/judge-correlation.md).

## 3. Validate

```bash
uv run brainforge config validate
```

Expected: `configuration is valid`. The command loads the config, expands
environment variables, checks references and judge independence, and
dry-builds every provider (an unsupported `api_style` fails here).

## 4. Test connectivity per model

```bash
uv run brainforge models test security
uv run brainforge models test judge
```

Expected: `ok (<n> in / <m> out)` plus a short reply. A 401 means the key did
not expand; check the environment variable name.

## 5. Set teacher knowledge cutoffs

For every teacher model, set `knowledge_cutoff` to its actual training cutoff
(`YYYY-MM`). It drives contamination tagging
([ADR-0003](../adr/0003-contamination-controls-postcutoff-holdout.md)). A
missing cutoff marks *all* records `recitation_risk`
(`unknown_teacher_cutoff`), which is loud on purpose.

## 6. Optional: cost estimates

Add `pricing` to models you want billed visibly:

```json
"judge": {
  "provider": "zen",
  "model": "glm-5.2",
  "family": "glm",
  "pricing": {"input_per_mtok": 1.4, "output_per_mtok": 4.4}
}
```

Per-call costs then appear in `data/logs/usage.jsonl`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `environment variable 'X' is not set` | key missing | export it or add `${X:default}` |
| `judge independence violated` | judge shares provider/family | move the judge to a distinct provider |
| `provider 'zen' does not support api_style` | `responses`/`anthropic` selected | use `chat_completions` models for now |
| `ok` locally but CI fails | CI has no keys | expected: CI uses mock providers only |

## Next steps

- [Generate a dataset](generate-dataset.md) with the real providers.
- [Provider reference](../reference/providers.md) for every option.
