# Configuration reference

Format: JSON (no YAML). Source of truth: pydantic models in
`src/brainforge/config/models.py`; `config/schema.json` is generated from them
(`brainforge config schema --write`) and CI checks it stays in sync.

Defaults: `--config config/config.json`, overridable with `--config`, the
`BRAINFORCE_CONFIG` environment variable, or by editing `config/config.json`.

## Environment expansion

Every string value supports `${VAR}` and `${VAR:default}`. A missing variable
without a default is a hard error at load time. Secrets never live in git.

## Top-level sections

### `providers`

| Field | Type | Notes |
|---|---|---|
| `type` | `openrouter`, `zen`, `mlgw`, `openai`, `local`, `mock` | selects the adapter |
| `base_url` | string | defaults per provider type |
| `api_key` | string | usually `${VAR}` |
| `api_style` | `chat_completions`, `responses`, `anthropic` | only `chat_completions` is implemented |
| `timeout` | float | seconds, default 120 |
| `max_retries` | int | retryable on 429/5xx, default 3 |

### `models`

| Field | Type | Notes |
|---|---|---|
| `provider` | string | key of `providers` |
| `model` | string | provider-side model id |
| `family` | string | model family used for judge-independence checks |
| `knowledge_cutoff` | `YYYY-MM` | teacher knowledge cutoff used for contamination tagging |
| `pricing` | `{input_per_mtok, output_per_mtok}` | optional, enables cost estimates |

### `roles`

`model` (key of `models`) plus `temperature`, `top_p`, `max_tokens`, optional
`system_prompt` override, optional explicit `kind` (`teacher`, `critic`,
`judge`). Kinds are inferred from the role name when absent: names containing
`judge` are judges; names containing `critic` or `general` are critics.

### `pipelines`

| Field | Type | Notes |
|---|---|---|
| `domain` | string | domain pack name (`security`, `coding`, ...) |
| `steps` | list of role names | explicit orchestration; omit to use the pack's mode defaults |
| `mode` | `cheap`, `standard`, `maximum` | step preset when `steps` is omitted |
| `min_confidence` | float | quality-gate threshold, default 0.5 |
| `reject_unresolved` | bool | route `insufficient_information` verdicts to rejected, default true |
| `reject_recitation_risk` | bool | also reject pre-cutoff cases, default false (tag only) |

### `judge_independence`

`enforced` (default, validation error), `warn`, or `off`. Enforced means: the
judge role must resolve to a provider that differs from every other role's
provider, and, when `family` is declared, to a different family as well. See
[the rationale in the dataset guide](dataset.md#judge-correlation).

### `training`

`base_model` (default `Qwen/Qwen3-8B`), `lora_rank`, `lora_alpha`,
`lora_dropout`, `learning_rate`, `epochs`, `batch_size`,
`gradient_accumulation`, `quantization` (`4bit` default), `output_dir`.

## Validation

```bash
uv run brainforge config validate   # references, judge independence, dry-builds providers
uv run brainforge config schema --check
```
