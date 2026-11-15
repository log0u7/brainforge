# Verdict 30 reps (MR6-v2 retrain) - run-20261104-mr6-v2-30r-30rep

## Chiffres (results.json, 570 attempts/modèle, run complet)
| Périmètre | base | trained | Δ | Fisher |
|---|---|---|---|---|
| Tous (19, contamination) | 460/570 (80.7%) | 474/570 (83.2%) | +2.5pp | p=0.317 |
| **Clean (12, hors train)** | **304/360 (84.4%)** | **323/360 (89.7%)** | **+5.3pp** | **p=0.045** |
| Seen/contaminé (7) | 156/210 (74.3%) | 151/210 (71.9%) | -2.4pp | p=0.660 |
| ROP family clean (4: ret2libc-001/002, shellcode-002, srop-001) | 101/120 (84.2%) | 113/120 (94.2%) | +10.0pp | p=0.021 |
| Clean sans callme-002 (11) | 281/330 (85.2%) | 293/330 (88.8%) | +3.6pp | p=0.203 |

Contamination: 7 IDs train dans eval (fmtstr-001, orw-001, ret2csu-001, ret2func-001, ret2plt-001, shellcode-001, uaf_heap-002-001). `contamination.clean: false`.

## Deltas par défi (brut)
| Défi | base | trained | Δ | p |
|---|---|---|---|---|
| callme-002 (clean) | 23/30 | 30/30 | +7 | 0.011 |
| orw-001 (seen, régression) | 9/30 | 1/30 | -8 | 0.012 |
| srop-001 (clean) | 25/30 | 29/30 | +4 | 0.195 |
| ret2libc-002 (clean) | 25/30 | 28/30 | +3 | 0.424 |
| shellcode-002 (clean) | 25/30 | 28/30 | +3 | 0.424 |
| shellcode-001 (seen) | 27/30 | 30/30 | +3 | 0.237 |
| ret2libc-001 (clean) | 26/30 | 28/30 | +2 | 0.671 |
| ret2csu-001 (seen, mort) | 0/30 | 0/30 | 0 | - |
| uaf_heap-001 (clean, mort) | 0/30 | 0/30 | 0 | - |
| 10 défis plafonnés (30/30) | 30/30 | 30/30 | 0 | - |

Note: après correction Holm sur les 7 tests avec delta, aucun ne survit à p<0.05 ajusté. Seuls callme-002 (+7) et orw-001 (-8) <0.05 brut.

## Lecture
1. **Signal transférable réel**: clean 84.4% → 89.7% (+5.3pp, p=0.045). Famille ROP clean +10pp (p=0.021). Le gain global 19-défis est dilué par du vu.
2. **Callme porte la significativité**: sans callme-002, clean devient +3.6pp p=0.203.
3. **Forgetting confirmé**: orw-001 (vu en train) régresse 9/30 → 1/30. Catastrophic forgetting / effet négatif d'entraînement.
4. **Dead zones structurelles**: ret2csu-001 et uaf_heap-001 jamais résolus par aucun modèle (0/60). Dataset actuel n'apprend pas ces techniques.
5. **Plafonds**: 10/19 défis saturés à 30/30. Espace de signal utile ~9 défis.

## Actions
- Analyser trails orw-001: pourquoi trained rate à générer les bons faits / à déclencher suggest→seccomp→orw (vs base réussit).
- Analyser trails ret2csu-001 + uaf_heap-001: tentatives plausibles vs dégénérées.
- Dataset v3: anti-forgetting sur orw-001 (et autres vus régressés), ajouter techniques pour ret2csu/uaf_heap, considérer exclusion ou rebalancing de contamination.
