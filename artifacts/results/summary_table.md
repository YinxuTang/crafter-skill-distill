# Results summary

All Student runs use matched_seeds_20 unless noted. Official Score = exp(mean(log(1+s_i)))-1 with s_i in [0,100].

## Teacher (PPO-AD e250)

| Metric | Value |
|---|---:|
| Progression (n=20) | 62.05% ± 3.49 |
| Official Score | 32.12 |
| Train steps | 1,024,000 |

## Student — distilled 72-rule Skill + DAG

| Run | n | Progression | Official Score | Notes |
|---|---:|---:|---:|---|
| 8B_full72_run1 | 20 | 15.00% ± 2.49 | 4.86 | complete |
| 8B_full72_run2 | 20 | 17.05% ± 2.13 | 3.76 | complete |
| 8B_full72_retarget | 20 | 12.05% ± 1.90 | 2.44 | complete; next_achievement targets rewritten |
| 32B_full72_old | 20 | 33.64% ± 3.27 | 11.79 | complete |
| 32B_full72_retarget_r1 | 20 | 34.77% ± 2.80 | 9.31 | complete; next_achievement targets rewritten |
| 32B_full72_retarget_r2 | 20 | 32.95% ± 3.57 | 10.24 | complete; next_achievement targets rewritten |

## Report main-table reference (pre-retarget 8B)

Use `8B_full72_run1` (15.00% / 4.86) for the report primary 72-rule row unless retarget is explicitly discussed.
