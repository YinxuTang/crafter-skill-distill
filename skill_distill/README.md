# Crafter grounded skill distillation

This package turns PPO-AD teacher rollouts into a cited natural-language skill
book and serves validated skills to the BALROG SPRING DAG.

## 1. Collect disjoint teacher data

```bash
python -m skill_distill.make_seed_splits \
  --out-dir data/skill_distill/seeds \
  --exclude data/matched_seeds_20.txt

SPLIT=train FORK_HORIZON=9 bash scripts/collect_teacher_skills.sh
SPLIT=validation FORK_HORIZON=9 bash scripts/collect_teacher_skills.sh
```

`matched_seeds_20.txt` is deliberately excluded from skill construction. The
collector records policy logits/value, obs-A text, structured state deltas, and
same-process counterfactual branches. Cross-process action-prefix replay is not
treated as authoritative because Crafter object update order can diverge.

## 2. Build and validate the skill book

For a ChatGPT-written book:

```bash
export OPENAI_API_KEY=...
export SKILL_SUMMARIZER_MODEL=gpt-4.1-mini
bash scripts/build_skill_library.sh
```

`OPENAI_BASE_URL` can point to another OpenAI-compatible endpoint. The API may
rewrite only natural-language fields. Machine predicates, dependencies,
actions, versions, and evidence spans remain unchanged and are checked after
the response.

Without an API key the build script uses a deterministic template renderer.
The resulting book remains grounded but has less cross-trajectory abstraction.

Outputs:

- `data/skill_distill/library/skills.v1.jsonl`: canonical serving format.
- `data/skill_distill/library/skill-book.v1.txt`: human-readable book.
- `data/skill_distill/library/skills.v1.validation.json`: promotion report.

## 3. Run the skill-aware SPRING agent

```bash
AGENT_TYPE=skill_spring \
SKILLS_MODE=retrieve_verify \
SKILLS_PATH=$HOME/rl2lang/data/skill_distill/library/skills.v1.jsonl \
SPRING_C=full \
bash scripts/run_spring_ablate.sh
```

Retrieval is injected only into q5-q7. The online verifier rejects observable
inventory/workstation violations and reports missing postconditions on the next
step. It does not expose teacher-only coordinates or the full semantic map.
