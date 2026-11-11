# Crafter Teacher → Skill 蒸馏（中文）

本目录是**自包含**工作区：代码、Teacher checkpoint、72 条 Skill、结果与脚本都在这里。  
**不要依赖**外面的 `rl2lang` 路径。

## 环境（只在本目录内）

本机已带 **`envs/ad/`**（约 7GB，已在 `.gitignore`，不进 GitHub）。  
克隆仓库的人请在本目录创建环境：

```bash
cd /path/to/crafter-skill-distill
bash scripts/setup_env.sh
# H100 / CUDA 12.1：
# TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash scripts/setup_env.sh
```

所有脚本经 `scripts/_env.sh` 启动，**会清空外部 PYTHONPATH**，只使用本目录的 `envs/ad` 或 `.venv`。

## 包内能跑什么

| 步骤 | 命令 | GPU |
|---|---|---|
| 算 Official Score | `python scripts/compute_official_score.py --run artifacts/results/sample_8b_full72_run1` | 否 |
| 加载 Teacher | `python scripts/smoke_load_ckpt.py` | 建议有 |
| Teacher 1 局冒烟 | `N_EP=1 bash scripts/eval_teacher.sh` | 是 |
| Teacher 20 seed | `N_EP=20 SEEDS_FILE=artifacts/teacher/matched_seeds_20.txt bash scripts/eval_teacher.sh` | 是 |
| 采集轨迹 | `SPLIT=exploration bash scripts/collect_teacher_skills.sh` | 是 |
| 旧版 skill library | `bash scripts/build_skill_library.sh`（需先有 raw 轨迹） | 视步骤 |

## 蒸馏步骤（v4，主结果路线）

产物已在 `artifacts/skills/skill_72/`。若要在本目录重跑：

1. **采集**（上表）→ `artifacts/teacher/raw/...`
2. **建 Decision DB**  
   `python -m skill_distill.analysis.build_decision_db --help`
3. **Critical bank**  
   `python -m skill_distill.analysis.rollout_critical_bank --help`
4. **提假设 + 验证**  
   `python -m skill_distill.hypothesis.propose_conditioned ...`
5. **消融 + 分类**  
   `python -m skill_distill.hypothesis.simplify ...`  
   `python -m skill_distill.hypothesis.classify ...`
6. **编译 Skill** → 英文 `SKILL.md`（现成文件可直接用）

完整 Decision DB 体积很大，默认不入库；采集后可在本目录生成。

## 主结果

见 `artifacts/results/summary_table.md`。

- Teacher e250：Progression 62.05% ± 3.49，Official Score 32.12  
- 8B 72 条 run1：15.00% / 4.86（报告主表）  
- 32B 72 条旧编译：33.64% / 11.79  

## 文件结构

```text
artifacts/teacher/     # ckpt、seeds、评测
artifacts/skills/      # 72 条 Skill
artifacts/results/     # 主表 + sample 评测 json
skill_distill/         # 蒸馏代码
third_party/Achievement-Distillation/
scripts/               # 只依赖本目录 .venv
docs/                  # 中文技术报告
```

## Groundedness / 限制

条件限可观测字段；fidelity 对 Teacher 数据验证；英文 Skill 对 Student 为软提示。  
`next_achievement` 不给一步动作；Student temperature=1.0 方差大。
