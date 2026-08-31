# Configuration reference

Format: JSON (decision [ADR-0001](../adr/0001-json-config-pydantic-schema.md)).
Source of truth: the pydantic models in `src/brainforge/config/models.py`;
`config/schema.json` is generated from them.

```bash
uv run brainforge config schema --write   # regenerate schema.json
uv run brainforge config schema --check   # fail when out of sync (CI runs this)
```

Resolution order: `--config/-c` flag, then `BRAINFORCE_CONFIG`, then
`config/config.json`.

## Environment expansion

Every string value supports `${VAR}` and `${VAR:default}`:

- a set variable always wins;
- an unset variable with a default resolves to the default (including empty);
- an unset variable without a default is a **hard error at load time**.

Secrets never live in git. See
[how-to: configure providers](../how-to/configure-providers.md).

## `providers`

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `openrouter` \| `zen` \| `mlgw` \| `openai` \| `local` \| `mock` | required | adapter selection |
| `base_url` | string | per type | API base URL |
| `api_key` | string | `null` | usually a `${VAR}` placeholder |
| `api_style` | `chat_completions` \| `responses` \| `anthropic` | `chat_completions` | endpoint dialect (only `chat_completions` implemented) |
| `timeout` | float | `120.0` | request timeout in seconds |
| `max_retries` | int | `3` | retries on 429/5xx with backoff |

Unknown keys are rejected (`extra="forbid"`).

## `models`

| Field | Type | Description |
|---|---|---|
| `provider` | string | key of `providers` |
| `model` | string | provider-side model id |
| `family` | string | model family; used by the judge-independence check |
| `knowledge_cutoff` | `YYYY-MM` | teacher training cutoff; drives contamination tagging |
| `pricing` | `{input_per_mtok, output_per_mtok}` | enables cost estimates in usage logs |

`knowledge_cutoff` must match `YYYY-MM` or validation fails.

## `roles`

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | required | key of `models` |
| `kind` | `teacher` \| `critic` \| `judge` | inferred | overrides kind inference |
| `temperature` | float | `0.7` | sampling temperature |
| `top_p` | float | `null` | nucleus sampling |
| `max_tokens` | int | `null` | completion cap |
| `system_prompt` | string | `null` | overrides the domain pack's system prompt |

Kind inference: names containing `judge` are judges, `critic` or `general` are
critics, anything else is a teacher. Inference keeps configs minimal; explicit
`kind` removes ambiguity for unusual names.

## `pipelines`

| Field | Type | Default | Description |
|---|---|---|---|
| `domain` | string | required | domain pack name |
| `steps` | list of role names | `null` | explicit orchestration; omit for mode presets |
| `mode` | `cheap` \| `standard` \| `maximum` | `standard` | step preset used when `steps` is omitted |
| `min_confidence` | float | `0.5` | quality-gate confidence threshold |
| `reject_unresolved` | bool | `true` | route `insufficient_information` verdicts to rejected |
| `reject_recitation_risk` | bool | `false` | also reject pre-cutoff cases (tag-only otherwise) |

Mode presets come from the domain pack (`security`: cheap = security teacher +
judge, standard = + coding teacher, maximum = + general critic).

## `judge_independence`

`enforced` (default) \| `warn` \| `off`. Enforced makes the load fail when the
judge role shares a provider or a declared `family` with any other role. Rationale:
[judge correlation](../explanation/judge-correlation.md), decision
[ADR-0002](../adr/0002-multi-teacher-ensemble-independent-judge.md).

## `training`

| Field | Type | Default | Description |
|---|---|---|---|
| `base_model` | string | `Qwen/Qwen3-8B` | student base model |
| `lora_rank` | int | `16` | LoRA rank |
| `lora_alpha` | int | `32` | LoRA alpha |
| `lora_dropout` | float | `0.05` | LoRA dropout |
| `learning_rate` | float | `2e-4` | optimizer learning rate |
| `epochs` | int | `3` | training epochs |
| `batch_size` | int | `1` | per-device batch size |
| `gradient_accumulation` | int | `16` | effective batch = 16 |
| `quantization` | `4bit` \| `8bit` \| `none` | `4bit` | QLoRA quantization |
| `output_dir` | string | `experiments` | run output root |

See [ADR-0010](../adr/0010-qlora-training-single-rtx-3080.md) and the
[training reference](training.md).
