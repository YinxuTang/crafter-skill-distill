# Crafter Teacher → Skill 蒸馏（中文说明）

本仓库是从内部工程**复制**出来的独立包（不修改原 `rl2lang` 目录），包含：

1. **PPO-AD Teacher** 最终 checkpoint 与评测结果  
2. **Skill 蒸馏代码**（`skill_distill/`）与 **72 条 Skill 产物**  
3. **Student 主表数字**与用法脚本  

英文版见 [README.md](README.md)。技术细节见 [docs/technical-report-zh.md](docs/technical-report-zh.md)。

---

## 现在好了吗？

| 项 | 状态 |
|---|---|
| 独立文件夹打包 | ✅ `/home/tiger/crafter-skill-distill` |
| Teacher ckpt + 评测摘要 | ✅ |
| 72 条 Skill（含 retarget） | ✅ |
| 蒸馏**代码**与**步骤说明** | ✅（见下文） |
| 冒烟：Official Score / GPU 加载 / Teacher 1 局 | ✅ |
| Decision DB / 原始轨迹（完整重跑蒸馏所需大数据） | ❌ 未打进包（太大） |
| 推送到 GitHub | ❌ 未推（需你确认） |

一句话：**读结果、用 Skill、评 Teacher 已经可用；要从零完整重跑蒸馏，还需自备 Decision DB（或重新采集）+ LLM API。**

---

## 总流程

```text
① 训练 / 准备 PPO-AD Teacher（本包已提供 agent-e250.pt）
② 采集 Teacher 轨迹（GPU）
③ 建 Decision Database + critical bank
④ LLM 在 Hypothesis DSL 里提规则 → fidelity/support 验证
⑤ 反例修订 + 谓词消融 + 分类筛选
⑥ 编译成英文静态 Skill（本包已提供 72 条）
⑦ 注入 SPRING QA-DAG，评 Qwen3-8B / 32B
```

---

## 主结果（摘要）

**Teacher e250**（20 个 matched seeds）：Progression **62.05% ± 3.49**，Official Score **32.12**。

**Student · 蒸馏 72 条 + DAG**（详见 `artifacts/results/summary_table.md`）：

| 配置 | Progression | Official Score |
|---|---:|---:|
| 8B · 72 条 run1（报告主表用） | 15.00% ± 2.49 | 4.86 |
| 8B · 72 条 run2 | 17.05% ± 2.13 | 3.76 |
| 8B · retarget | 12.05% ± 1.90 | 2.44 |
| 32B · 72 条（旧编译） | 33.64% ± 3.27 | 11.79 |
| 32B · retarget r1 / r2 | 34.77% / 32.95% | 9.31 / 10.24 |

---

## 用法命令

### 环境

- **不用 GPU**：读 Skill、算 Official Score  
- **要 GPU**：加载 Teacher、评 Teacher、采轨迹、训 PPO、跑 Student（vLLM）  
- 若本机有 `~/rl2lang/envs/ad/bin/python`，脚本会优先用它  

```bash
cd /home/tiger/crafter-skill-distill   # 或你的克隆路径
```

### A. 算 Official Score（CPU）

```bash
python scripts/compute_official_score.py \
  --run /path/to/balrog_run_dir
```

### B. 冒烟加载 Teacher（GPU）

```bash
python scripts/smoke_load_ckpt.py artifacts/teacher/agent-e250.pt
```

### C. 评 Teacher（GPU；冒烟 1 局）

```bash
N_EP=1 bash scripts/eval_teacher.sh
```

正式 20 seed：

```bash
N_EP=20 SEEDS_FILE=artifacts/teacher/matched_seeds_20.txt \
  OUT=artifacts/results/teacher_eval_20ep.json \
  bash scripts/eval_teacher.sh
```

### D. Student + Skill（需自备 BALROG / vLLM）

```bash
AGENT_TYPE=skill_spring SKILLS_MODE=static \
SKILLS_PATH=$PWD/artifacts/skills/skill_72/SKILL.md \
MODEL=Qwen/Qwen3-8B N_EP=20 \
SEEDS_FILE=$PWD/artifacts/teacher/matched_seeds_20.txt \
bash /path/to/run_spring_ablate.sh
```

---

## 蒸馏步骤（你要的 distill pipeline）

本包已带上 **72 条最终 Skill**（`artifacts/skills/skill_72/`）。  
下面是复现蒸馏时的步骤与命令；**大数据（Decision DB）未入库**，需先有轨迹与 DB。

### 步骤 1 — 采集 Teacher 轨迹（GPU）

```bash
CKPT=artifacts/teacher/agent-e250.pt SPLIT=exploration \
  bash scripts/collect_teacher_skills.sh
# 还可 SPLIT=revision / acceptance 等 holdout
```

输出：episode jsonl（状态、动作、logits 等）。

### 步骤 2 — 构建 Decision Database

```bash
python -m skill_distill.analysis.build_decision_db \
  --raw artifacts/teacher/raw/teacher_e250_exploration \
  --out artifacts/decision_db/decisions.csv
```

（参数名以模块 `--help` 为准；核心是把逐步决策摊平为可查询表。）

### 步骤 3 — Critical bank（关键状态）

```bash
python -m skill_distill.analysis.rollout_critical_bank \
  --help   # 按项目默认 horizon=24 / n_samples=5 / γ=0.95
```

### 步骤 4 — 提出假设并验证（需 LLM）

每轮大致：

```bash
python -m skill_distill.hypothesis.propose_conditioned \
  --decisions artifacts/decision_db/decisions.csv \
  --critical-bank artifacts/critical_states/rollout_bank_action_stratified \
  --baseline-report artifacts/baselines/mvp_baseline_report.json \
  --existing-classified artifacts/hypotheses/registry_classified.json \
  --out-dir artifacts/hypotheses/cycle0 \
  --n-hypotheses 10 \
  --refine-rounds 1 \
  --split exploration \
  --counterexample-split revision \
  --evidence-cycle 0
```

程序在 Decision DB 上算 **support / fidelity**；revision split 挖反例后做**至多一轮**修订。

### 步骤 5 — 谓词消融 + 分类

```bash
python -m skill_distill.hypothesis.simplify \
  --evaluated artifacts/hypotheses/cycle0/evaluated_round1.json \
  --decisions artifacts/decision_db/decisions.csv \
  --out artifacts/hypotheses/cycle0/simplified.json \
  --splits exploration revision \
  --min-fidelity 0.55 \
  --max-fidelity-drop 0.03

python -m skill_distill.hypothesis.classify \
  --evaluated artifacts/hypotheses/cycle0/simplified.json \
  --decisions artifacts/decision_db/decisions.csv \
  --split exploration \
  --out artifacts/hypotheses/cycle0/classified.json
```

多轮 cycle（0–11）后去重筛选：keep + undercovered → **72 条**实验 Skill。

### 步骤 6 — 编译英文 Skill

结构化规则 → 英文规则书。本包已放好：

- `artifacts/skills/skill_72/SKILL.md`（报告主表用）  
- `artifacts/skills/skill_72_retarget/SKILL.md`（`next_achievement` 写明具体 target）  
- `artifacts/skills/skill_72/skill_72.json`（结构化）

### 另一条更早的「轨迹片段 → skill library」流水线

代码同样在 `skill_distill/`（segment → pair → fork → extract → summarize → validate）。  
可用适配脚本：

```bash
bash scripts/build_skill_library.sh
```

这与 v4 Hypothesis DSL 发现是**不同路线**；当前主结果来自 **v4 · 72 条**。

---

## Groundedness（简要）

- 条件只能用可观测字段（`allowed_fields`，约 113 个）  
- 每条规则用 Teacher 数据算 support / fidelity  
- 英文 Skill 对 Student 是**软提示**，运行时不会强制执行谓词  

---

## 限制

- `next_achievement` 只给目标成就，不给一步动作  
- Student `temperature=1.0`，方差大  
- retarget 后 8B 未必更好（见上表）  

---

## 许可

`third_party/Achievement-Distillation` 遵循上游 LICENSE；其余包装文件默认 MIT。
