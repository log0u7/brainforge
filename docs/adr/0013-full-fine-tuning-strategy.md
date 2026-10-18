# Full fine-tuning as a second strategy after QLoRA

* Status: accepted
* Date: 2026-10-18

## Context and Problem Statement

The workflow is two-phase: a quick QLoRA pass proves the dataset (minutes,
one 10GB GPU), then a heavier pass produces the decision model that sees
the business history. QLoRA alone caps quality; the decision phase wants
every weight trained. The trainer only supported QLoRA.

## Decision Drivers

* The two phases have different budgets and different goals; one flag
  should select between them.
* Full FT changes three things: no quantization config, no LoRA config,
  bf16 compute. Everything else (dataset loading, splits, resume,
  summaries, eval) is identical.
* Silent footguns (e.g. quantization config passed alongside full FT)
  must be impossible.

## Considered Options

* A separate `train full` command.
* A `--full` boolean flag.
* A `strategy` enum (`qlora` | `full`) in config and on the CLI.

## Decision Outcome

Chosen option: "strategy enum", because it composes with the backend
dispatch (`qlora` + `unsloth`, `full` + `trl`) without a flag matrix, and
the config field documents the intent of a run.

### Consequences

* Positive: one code path, three booleans; the summary records the
  strategy so runs are self-describing.
* Negative: full FT needs more VRAM; a mis-set config can OOM.
* Mitigation: the choice is explicit per run (`--strategy`), never
  inferred.

### Confirmation

* `TrainingConfig.strategy` and `train run --strategy`.
* `tests/unit/test_full_finetune.py`: full passes no `peft_config` and no
  `quantization_config` to the trainer (fake modules); qlora behavior is
  byte-identical to before; the CLI flag overrides config.

## Pros and Cons of the Options

### Separate command

* Good, because explicit.
* Bad, because duplicates the dispatch and run-directory logic.

### Boolean flag

* Good, because simple.
* Bad, because it does not extend (KD later would need a third state).

### Strategy enum

* Good, because extensible and composable with backends.
* Bad, because slightly more config surface.

## Links

* [Training reference](../reference/training.md)
* [ADR-0010](0010-qlora-training-single-rtx-3080.md)
