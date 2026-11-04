# datasets/eval - committed protocol inputs

The `datasets/` tree is gitignored (generated corpora are large), EXCEPT
this directory: the files here are small, load-bearing inputs of the
final-bench protocol (`scripts/run_final_bench.sh`) and must survive
reboots. Nothing under /tmp is a protocol input.

## grpo-translated.jsonl

7 records = `datasets/grpo-lab/train.jsonl` (7 records, IDs in the
`lab-<name>-001` namespace) with the IDs translated to the eval
namespace (`ctf-lictor-<name>-001`) so the contamination audit matches
the eval challenge IDs. Challenge content is identical; only the IDs
were mapped. The 7 covered challenges are the mined set:

- ctf-lictor-callme-001
- ctf-lictor-fmtstr-001
- ctf-lictor-orw-001
- ctf-lictor-ret2csu-001
- ctf-lictor-ret2func-001
- ctf-lictor-ret2plt-001
- ctf-lictor-shellcode-001

Regenerate (deterministic ID mapping, one namespace swap):

```bash
python3 - <<'PY'
import json
for line in open("datasets/grpo-lab/train.jsonl"):
    r = json.loads(line)
    r["id"] = r["id"].replace("lab-", "ctf-lictor-", 1)
    print(json.dumps(r))
PY > datasets/eval/grpo-translated.jsonl
```

## sft-mr6/train.jsonl (referenced, NOT committed)

The second contamination source, `datasets/sft-mr6/train.jsonl`, stays
uncommitted (large, generated). The protocol script requires it on
disk and fails with a clear error when missing; regenerate it with
`scripts/assemble_sft_mr6.py` from the bench run dirs.

## uaf sacrifice ledger (to be maintained)

When a challenge is moved from eval to training (e.g. uaf_heap-002
after MR6, mined from its 15 measured base wins), its records enter
`datasets/sft-mr6/train.jsonl` and the clean-derivation in
`scripts/run_final_bench.sh` drops it from the eval automatically.
Log the decision here:
- (2026-11-04, done) uaf_heap-002: sacrifice EVAL -> TRAIN, mined from
  15 executed wins (15 distinct run UUIDs, zero dedup) across all
  seven run-20261103/04 dirs into `datasets/sft-uaf-wins/train.jsonl`
  (local, gitignored like every generated corpus). Records carry the
  eval-namespaced `ctf-lictor-uaf_heap-002-001` ID on purpose: the
  assembly (dataset v2) merges them into the training set that feeds
  the protocol derivation, which then drops the challenge from eval.
  The remaining unseen eval instance for the uaf class:
  `uaf_heap-001` (0/10 both models so far).
