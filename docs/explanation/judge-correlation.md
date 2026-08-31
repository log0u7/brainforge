# Judge correlation: the erroneous-agreement problem

Multi-teacher pipelines are usually justified by disagreement handling:
"independent teachers disagree, a judge arbitrates". That story covers the
visible failure. The dangerous failure is the invisible one.

## Two failure modes

Consider two verdicts a pipeline can produce:

1. **Disagreement**: the security teacher says "confirmed", the coding teacher
   says "not a defect". The judge weighs arguments. This is the case the
   plan-level literature worries about, and the easy one: disagreement is
   visible in the data.
2. **Erroneous agreement**: the security teacher and the judge *both* say
   "confirmed" with high confidence, and both are wrong. No disagreement
   surfaces; the record sails through the gate into the dataset.

## Why teachers and judges correlate

Two LLMs from the same provider or the same model family tend to:

- share pretraining data and, therefore, the same blind spots (families of
  vulnerabilities they consistently miss);
- share alignment and RLHF choices (similar confidence calibration, similar
  tendency to agree with a confident-sounding prior analysis);
- sometimes be the same model behind different endpoints: gateways re-serve
  the same weights, so "OpenRouter model A" and "Zen model B" can be the same
  network.

Correlated verifiers produce a pipeline that agrees with itself into confident
errors. Note the asymmetry with cost: a false "confirmed" poisons the student
with a hallucinated vulnerability; a false "rejected" only loses one example.

## The safeguards

BrainForge addresses correlation at three levels:

1. **Structural separation, enforced**. `config validate` (default:
   `judge_independence: enforced`) requires the judge role to resolve to a
   different provider from every teacher role, *and* a different declared
   `family` when families are declared. Provider-level separation alone is
   insufficient precisely because gateways re-serve shared weights; the family
   check catches "same model, different door".
2. **Adversarial prompting**. The judge system prompt orders an explicit
   re-derivation: conclude from the case evidence *first*, then weigh teacher
   opinions; refuse any claim not traceable to the provided code or data; treat
   teacher consensus as weak evidence when the evidence is absent. This reduces
   prompt-level anchoring on top of the structural separation.
3. **Measured agreement**. Every record stores `teacher_agreement` per teacher;
   experiment reports compute `teacher_judge_agreement_rate` and flag rates
   >= 95%. A judge that agrees with a teacher almost always is not arbitrating;
   that flag means the ensemble is no longer doing verification work.

## What this does not fix

- Families must be declared honestly; a mislabeled `family` defeats the check.
- Different providers can still serve models distilled from each other; the
  agreement-rate monitoring is the backstop.
- All teachers sharing one provider (allowed: only the judge must differ)
  keeps *teacher-teacher* correlation; the general critic mitigates it by
  charter (contradiction hunting), and ablation experiments measure whether
  each teacher actually contributes.

## Related

- [ADR-0002](../adr/0002-multi-teacher-ensemble-independent-judge.md): the
  decision record.
- [How-to: configure providers](../how-to/configure-providers.md): setting up
  a compliant judge.
- `src/brainforge/config/models.py` (`check_judge_independence`),
  `src/brainforge/experiments/manager.py` (`high_agreement_warning`).
