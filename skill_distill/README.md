# skill_distill (in-repo)

See the package root [README_zh.md](../README_zh.md) for Chinese usage.

Quickstart after `bash scripts/setup_env.sh`:

```bash
SPLIT=exploration bash scripts/collect_teacher_skills.sh
bash scripts/build_skill_library.sh   # needs collected raw trajectories
```

The shipped 72-rule book is under `artifacts/skills/skill_72/` (v4 Hypothesis-DSL route).
