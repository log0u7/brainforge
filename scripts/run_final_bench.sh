#!/usr/bin/env bash
# PROTOCOL: the single entry point for final MR benches (ADR-0021).
#
# Two arms:
#   full   - model + augure + lictor (the production system)
#   model  - model alone, no augure/lictor: the model must solve the
#            challenge from source code alone
#
# WHY THIS SCRIPT EXISTS (lesson from run-20261104-mr6-10rep, 2026-11-04):
#   the re-bench was launched by hand WITHOUT the committed protocol
#   inputs and the /tmp working files the earlier runs had used. The
#   run still completed, but the verdict had to be re-derived after the
#   fact and the contamination gate almost hid a real regression
#   (uaf_heap-002 5/10 -> 0/10) behind a global score. From now on:
#
#   1. NO hand-typed bench commands. This script is the only entry.
#   2. Every protocol input is COMMITTED in the repo (no /tmp):
#      - the challenge manifest        experiments/ctf_ingest/manifest.jsonl
#      - the contamination sources     datasets/sft-mr6/train.jsonl
#                                      datasets/sft-mr6-v2/train.jsonl
#                                      datasets/eval/grpo-translated.jsonl
#   3. The clean list is DERIVED (verified lab challenges minus training
#      IDs), printed at launch, and the final verdict is computed on
#      that clean subset only.
#   4. Exit code 1 from the bench is EXPECTED whenever the run includes
#      mined (leaked) challenges: the contamination audit fails the run
#      on purpose. The verdict below is the clean-only readout; do not
#      confuse exit 1 with a failed run.
#
# Usage:
#   scripts/run_final_bench.sh <tag> [repetitions] [--mode full|model]
#   tag examples: mr6-3ep, mr6-uafwin, mr5-final
#
# Env overrides (defaults keep the MR5/MR6 reference setup):
#   TRAINED_URL / TRAINED_MODEL   (default 9933 / trained-mr8)
#   BASE_URL / BASE_MODEL         (default 9932 / base-qwen3-1.7b)
#
# Outputs:
#   experiments/final-bench-results/run-YYYYMMDD-<tag>-<reps>rep/
#   .../results.json      full run + contamination report
#   stdout                clean-only verdict table (the headline numbers)
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:?usage: run_final_bench.sh <tag> [repetitions] [--mode full|model]}"
REPS="${2:-10}"
shift 2 2>/dev/null || true
MODE="full"
while [[ $# -gt 0 ]]; do
	case "$1" in
	--mode)
		MODE="$2"
		shift 2
		;;
	*)
		shift
		;;
	esac
done

if [[ "$MODE" != "full" && "$MODE" != "model" ]]; then
	echo "PROTOCOL ERROR: invalid mode '$MODE'. Must be 'full' or 'model'." >&2
	exit 2
fi

MANIFEST="experiments/ctf_ingest/manifest.jsonl"
export REPS
export MODE
CASES_DIR="examples/cases"
TRAIN_IDS=(
	"datasets/sft-mr6/train.jsonl"
	"datasets/sft-mr6-v2/train.jsonl"
	"datasets/eval/grpo-translated.jsonl"
)
BASE_URL="${BASE_URL:-http://192.168.1.26:9932/v1}"
BASE_MODEL="${BASE_MODEL:-base-qwen3-1.7b}"
TRAINED_URL="${TRAINED_URL:-http://192.168.1.26:9933/v1}"
TRAINED_MODEL="${TRAINED_MODEL:-trained-mr8}"

for f in "$MANIFEST" "${TRAIN_IDS[@]}"; do
	if [ ! -f "$f" ]; then
		echo "PROTOCOL ERROR: committed protocol input missing: $f" >&2
		exit 2
	fi
done

WORKDIR="experiments/final-bench-results/run-$(date +%Y%m%d)-$TAG-${REPS}rep"
if [ -e "$WORKDIR" ]; then
	echo "PROTOCOL ERROR: workdir already exists: $WORKDIR" >&2
	exit 2
fi

# --- 1. print the derived clean list and the expected totals ----------
CLEAN_IDS="$(
	uv run python - "$MANIFEST" "$CASES_DIR" "${TRAIN_IDS[@]}" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "src")
from brainforge.experiments.final_bench import load_verified_challenges, load_training_ids

manifest, cases_dir = Path(sys.argv[1]), Path(sys.argv[2])
challenges, skipped = load_verified_challenges(manifest, cases_dir)
train = load_training_ids(*[Path(p) for p in sys.argv[3:]])
clean = [c[0] for c in challenges if c[0] not in train]
print(json.dumps({"clean": clean, "n_challenges": len(challenges), "n_clean": len(clean)}))
PY
)"

python3 - "$CLEAN_IDS" <<'PY'
import json, os, sys
d = json.loads(sys.argv[1])
print(f"PROTOCOL: {d['n_challenges']} lab challenges in manifest, {d['n_clean']} CLEAN for the verdict:")
for c in d["clean"]:
    print(f"  - {c}")
print(f"expected totals: {d['n_clean']} challenges x 2 models x {os.environ.get('REPS', '10')} reps")
print(f"MODE: {os.environ.get('MODE', 'full')}")
PY

# --- 2. run the bench ---------------------------------------------------
echo "PROTOCOL: launching bench -> $WORKDIR (mode: $MODE)"
set +e
REPEATED_IDS=()
for id in "${TRAIN_IDS[@]}"; do
	REPEATED_IDS+=(--training-ids "$id")
done
uv run python -m brainforge.experiments.final_bench \
	--manifest "$MANIFEST" --cases-dir "$CASES_DIR" \
	"${REPEATED_IDS[@]}" \
	--base-url "$BASE_URL" --base-model "$BASE_MODEL" \
	--trained-url "$TRAINED_URL" --trained-model "$TRAINED_MODEL" \
	--repetitions "$REPS" \
	--mode "$MODE" \
	--workdir "$WORKDIR"
BENCH_RC=$?
set -e
if [ "$BENCH_RC" -ne 0 ]; then
	echo "PROTOCOL NOTE: bench exit $BENCH_RC - contamination LEAKED is EXPECTED when the"
	echo "  manifest includes mined challenges; the verdict below is clean-only."
fi

# --- 3. clean-only verdict ----------------------------------------------
uv run python - "$WORKDIR/results.json" "${TRAIN_IDS[@]}" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "src")
from brainforge.experiments.final_bench import load_training_ids

res = json.loads(Path(sys.argv[1]).read_text())
train = load_training_ids(*[Path(p) for p in sys.argv[2:]])
clean = [c for c in res["challenges"] if c not in train]
print(f"\nVERDICT (clean-only, {len(clean)} challenges):")
print(f"{'challenge':34} {'base':>7} {'trained':>9} {'delta':>7}")
regress = []
for c in clean:
    b = res["base"]["per_challenge"][c]["successes"]
    t = res["trained"]["per_challenge"][c]["successes"]
    mark = "  <-- REGRESSION" if t < b else ("  +GAIN" if t > b else "")
    if t < b:
        regress.append((c, b, t))
    print(f"{c:34} {b:>5}/10 {t:>7}/10 {t-b:>+7}{mark}")
bt = sum(res["base"]["per_challenge"][c]["successes"] for c in clean)
tt = sum(res["trained"]["per_challenge"][c]["successes"] for c in clean)
n = len(clean) * res["repetitions"]
print(f"{'TOTAL':34} {bt:>4}/{n} {tt:>6}/{n}  ({bt/n:.3f} vs {tt/n:.3f})")
if regress:
    print("\nPER-CHALLENGE REGRESSIONS (metric requires none):")
    for c, b, t in regress:
        print(f"  {c}: {b}/10 -> {t}/10")
PY
echo "PROTOCOL: done. Workdir: $WORKDIR"
