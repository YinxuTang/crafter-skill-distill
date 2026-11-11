# Crafter Teacher Skill Distillation

> **中文版请看 [README_zh.md](README_zh.md)**（含完整蒸馏步骤）。
> Distillation step-by-step (v4 Hypothesis DSL + earlier skill-library path) is documented in the Chinese README.


Package a **PPO-AD RL teacher**, a **grounded skill-distillation pipeline**, and **Student evaluation artifacts** for Crafter.

This repo is a **standalone copy** of useful outputs from an internal workspace. It does **not** modify the original project tree.

## What is included

| Piece | Path | GPU? |
|---|---|---|
| Teacher checkpoint (e250) | `artifacts/teacher/agent-e250.pt` | for eval / collect |
| Teacher config + matched seeds | `artifacts/teacher/` | |
| Teacher eval summary | `artifacts/results/summary_table.md` | |
| Distillation code | `skill_distill/` | for discovery / collect |
| 72-rule Skill book | `artifacts/skills/skill_72/` | |
| Retarget Skill book | `artifacts/skills/skill_72_retarget/` | |
| Chinese technical report | `docs/technical-report-zh.md` | |
| Achievement-Distillation (slim) | `third_party/Achievement-Distillation/` | train / eval |

Not included (too large / not needed for smoke use): Decision DB (~597MB), raw teacher trajectories, intermediate checkpoints e025–e225, full BALROG episode dumps.

## Pipeline

```text
PPO-AD train → agent-e250.pt
    → collect trajectories → Decision DB
    → Hypothesis DSL → fidelity / support → revision + predicate ablation
    → static Skill (72 rules)
    → SPRING QA-DAG Student (Qwen3-8B / 32B)
```

## Headline numbers

**Teacher e250** (n=20 matched seeds): Progression **62.05% ± 3.49**, Official Score **32.12**.

**Student 72-rule + DAG** (see `artifacts/results/summary_table.md`):

| Run | Progression | Official Score |
|---|---:|---:|
| Qwen3-8B, 72-rule run1 | 15.00% ± 2.49 | 4.86 |
| Qwen3-8B, 72-rule run2 | 17.05% ± 2.13 | 3.76 |
| Qwen3-8B, retarget | 12.05% ± 1.90 | 2.44 |
| Qwen3-32B, 72-rule (old) | 33.64% ± 3.27 | 11.79 |
| Qwen3-32B, retarget r1 | 34.77% ± 2.80 | 9.31 |
| Qwen3-32B, retarget r2 | 32.95% ± 3.57 | 10.24 |

Official Score uses achievement success rates \(s_i \in [0,100]\):

\[
\mathrm{Score}=\exp\big(\tfrac{1}{22}\sum_i \log(1+s_i)\big)-1
\]

## Usage

### 0) Environment (in-repo only)

```bash
bash scripts/setup_env.sh
# CUDA example:
# TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash scripts/setup_env.sh
```

All scripts use `./.venv` inside this folder. No `rl2lang` paths required.

- **No GPU**: Official Score / reading Skills  
- **GPU**: teacher load, teacher eval, collect, train, Student vLLM  

### 1) Compute Official Score (CPU)

```bash
cd /path/to/crafter-skill-distill
python scripts/compute_official_score.py \
  --run artifacts/results/sample_8b_full72_run1
```

### 2) Smoke-load teacher checkpoint (GPU recommended)

```bash
python scripts/smoke_load_ckpt.py artifacts/teacher/agent-e250.pt
```

### 3) Teacher eval smoke (1 episode, GPU)

```bash
N_EP=1 bash scripts/eval_teacher.sh
```

Formal 20-seed eval (longer):

```bash
N_EP=20 SEEDS_FILE=artifacts/teacher/matched_seeds_20.txt \
  OUT=artifacts/results/teacher_eval_20ep.json \
  bash scripts/eval_teacher.sh
```

### 4) Collect teacher trajectories (GPU)

```bash
CKPT=artifacts/teacher/agent-e250.pt SPLIT=exploration \
  bash scripts/collect_teacher_skills.sh
```

### 5) Student eval with Skill + SPRING (GPU + BALROG + vLLM)

Not fully vendored here. Point BALROG at:

```bash
AGENT_TYPE=skill_spring SKILLS_MODE=static \
SKILLS_PATH=$PWD/artifacts/skills/skill_72/SKILL.md \
MODEL=Qwen/Qwen3-8B N_EP=20 \
SEEDS_FILE=$PWD/artifacts/teacher/matched_seeds_20.txt \
bash /path/to/run_spring_ablate.sh
```

### 6) Optional: retrain teacher from scratch (GPU, long)

```bash
cd third_party/Achievement-Distillation
python train.py --exp_name ppo_ad  # uses configs/ppo_ad.yaml
```

This package already ships `agent-e250.pt`; retraining is optional.

## Groundedness (short)

- Rule **conditions** may only use observable Decision-DB fields (`allowed_fields`).
- Hypotheses are scored with **support / fidelity** against the teacher.
- The English Skill is a **soft prompt** for the Student; runtime does not hard-execute predicates.

## Limits

- `next_achievement` rules name a target but do not specify a primitive action.
- Student decoding uses temperature 1.0 → high run-to-run variance.
- Retargeting next-achievement text did **not** help 8B vs pre-retarget 72-rule runs.

## Cite

- Crafter (Hafner et al.)
- Achievement Distillation / PPO-AD (Moon et al.)
- BALROG / SPRING-style agents
- Qwen3

## License

Upstream `third_party/Achievement-Distillation` keeps its original LICENSE. Other packaging files: MIT unless noted.
