# Final bench results

One directory per completed run (`results.json` full run, `results.md`
headline readout). These directories are data, not scratch: a completed
run is committed before the next run starts (2026-11-15 lesson - the
mr9/mr10 raw results were destroyed by a worktree cleanup because they
were never committed).

Canonical runs:

| run | what it established |
|---|---|
| run-20261103-11ch-10rep | first full lab bench (11 clean x 2 models x 10 reps); harness-robustness evidence |
| run-20261103-11ch-10rep-b | first bench reproduction (docs/explanation/benchmark.md) |
| run-20261103-11ch-clean | clean-only readout reproduction (docs/explanation/benchmark.md) |
| run-20261103-17ch | corpus growth 11 -> 17 challenges |
| run-20261104-mr6-10rep | the MR6 skeleton bench (11 clean x 2 x 10) |
| run-20261104-mr6-v2-30r-30rep | the v2 sleep bench, 30 reps (570 attempts per model) |
| run-20261107-mr7-fix20-10rep | MR7 tooling pass 1 (fix20) |
| run-20261107-mr7-fix36-10rep | MR7 tooling pass 2 (fix36) - base/trained tie |
| run-20261107-mr8-10rep | MR8 full bench |
| run-20261108-mr8-model-10rep | MR8 model-only (the silent-collapse evidence) |
| run-20261108-mr8-model-v3-10rep | model-only v3 (quoted-atom audit, CHANGELOG) |
| run-20261108-mr8-model-v5-10rep | model-only v5, final state before MR9 |

Convention (ADR-0021, scripts/run_final_bench.sh):

- completed run = committed run; the run directory name is
  `run-YYYYMMDD-<tag>-<N>rep`;
- debug and rebench iterations are deleted once the canonical run exists.
