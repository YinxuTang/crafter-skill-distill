# PPO-AD Teacher training notes

Extracted from internal Step-1 notes. Checkpoint shipped here: `artifacts/teacher/agent-e250.pt`.

## Setup

- Algorithm: Achievement Distillation (PPO-AD)
- Architecture: feed-forward ImpalaCNN + MLP (no LSTM)
- Non-recurrent achievement memory (latent delta of last unlock)
- Config: `artifacts/teacher/ppo_ad.yaml` / `third_party/Achievement-Distillation/configs/ppo_ad.yaml`
- Schedule: 250 epoch × 512 × 8 = **1,024,000** env steps
- Save frequency: every 25 epochs; package keeps **e250 only**

## Training outcome (full training episodes)

- Approx. geometric Crafter score over training episodes: **~21.32**
- Strong early-tree unlock rates (wood / table / sapling / plant / wood tools); rare late-tree items near zero during training aggregate

## Formal eval (matched 20 seeds)

| Metric | Value |
|---|---:|
| BALROG Progression | 62.05% ± 3.49 |
| Official Crafter Score | 32.12 |
| Mean unlocks / episode | ~13.65 / 22 |

Use Progression and Official Score separately; per-episode JSON `score` is unlock count, not the geometric score.

## Reproduce eval

```bash
N_EP=20 SEEDS_FILE=artifacts/teacher/matched_seeds_20.txt bash scripts/eval_teacher.sh
```
