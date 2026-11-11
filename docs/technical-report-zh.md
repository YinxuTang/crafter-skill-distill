# Crafter RL Teacher 

## 1. 概述

使用**一个** Crafter Teacher（PPO Achievement Distillation, PPO-AD）。做法是：

1. 采集该 Teacher 的轨迹；
2. 把轨迹变成逐决策数据库；
3. 用 LLM 在受限 DSL 里提出策略假设；
4. 用程序验证、挖反例、简化、去重；
5. 编译成英文静态 Skill；
6. 在固定九问 QA-DAG 下评估 Student。

环境是 Crafter：17 个离散动作，22 项成就。Student 为 `Qwen3-8B` 与 `Qwen3-32B`。

---

## 2. Pipeline 总览

```text
Teacher 轨迹采集
    ↓
Decision Database
    ↓
关键状态识别
    ↓
构造 LLM Scientist 输入
    ↓
LLM 提出 Hypothesis DSL
    ↓
程序验证与 fidelity 评估
    ↓
Revision 反例 + 一次有界修订
    ↓
谓词消融与简化
    ↓
多轮 discovery、去重与筛选
    ↓
编译英文静态 Skill
    ↓
注入固定 QA-DAG，评估 8B / 32B
```

数据规模（当前）：

- exploration：500 episodes / 128,853 decisions  
- revision：80 episodes / 19,798 decisions  
- acceptance：80 episodes / 19,673 decisions  
- Decision DB 合计：168,324 行  

cycle0–11 后得到 127 条简化候选，去重后 120 条；保留 keep 54 + undercovered 18，编译成实验用 **72 条 Skill**。

---

## 3. 各步骤在做什么

### 3.1 采集 Teacher 轨迹

**做什么**：用固定 Teacher checkpoint 在隔离 seed 上滚动，保存每一步状态与动作。

**输入**：Teacher checkpoint、Crafter 配置、seed、split。

**输出（episode summary 样例）**：

```json
{
  "episode_id": "ep_00000_seed_730986878",
  "seed": 730986878,
  "num_steps": 308,
  "episode_return": 13.1,
  "progression": 0.63636,
  "achievements": {
    "collect_wood": 19,
    "make_wood_pickaxe": 1,
    "make_stone_pickaxe": 1,
    "collect_coal": 1,
    "collect_diamond": 0
  },
  "done": true
}
```

每一步还会保存动作、value、inventory、可见物体、距离/方向、成就等字段，供后续建库。

---

### 3.2 构建 Decision Database

**做什么**：把轨迹展成“一行一个决策”，并补齐可观测状态特征。

**输入**：Teacher 轨迹。

**输出（decision row 样例）**：

```json
{
  "split": "exploration",
  "episode_id": "ep_00000_seed_730986878",
  "seed": 730986878,
  "step": 69,
  "action": "Make Stone Pickaxe",
  "front": "table",
  "is_night": false,
  "health": 9,
  "food": 9,
  "drink": 6,
  "energy": 7,
  "inv_wood": 1,
  "has_wood_pickaxe": true,
  "visible_stone": true,
  "dist_stone": 2,
  "dir_stone": "north",
  "ach_make_wood_pickaxe": true
}
```

规则条件只能用当前可观测字段。`action`、`reward`、`seed`、`next_achievement` 等不能写进 condition，只能当标签或评估目标。

---

### 3.3 识别关键状态

**做什么**：不是所有 Teacher 动作都值得蒸馏。对候选状态做 repeated continuation rollout，估计 Teacher 动作相对替代动作的优势：

\[
\Delta Q(s)=Q_{\text{teacher}}(s,a_T)-\max_{a\neq a_T}Q(s,a)
\]

具体做法写清楚如下：

1. 在当前状态 \(s\) 上，强制执行候选动作 \(a\)（Teacher 动作或替代动作）作为**第 1 步**；
2. 之后交给同一个 Teacher 自己继续采样，再往前滚 **23 步**；
3. 因此一次 continuation 一共向前看 **horizon = 24 步**；
4. 回报按 \(\gamma=0.95\) 折扣累计；若 24 步内 episode 未结束，再用 Teacher value head bootstrap；
5. 每个候选动作独立重复 **5 次**（`n_samples=5`），取均值、标准误和 95% CI；
6. 替代动作取 Teacher logits 里 top-k（默认 k=3）中、除 Teacher 动作外最好的那个。

也就是说：这里的“优势”不是一步 TD，而是 **强制第一步动作后，再看未来约 24 步折扣回报** 的差距。

**输出规模**：

```json
{
  "critical_correct": 2218,
  "critical_negative": 5052,
  "ambiguous_large": 5978,
  "indifferent": 31500,
  "unstable": 253
}
```

**critical_correct 样例**：

```json
{
  "seed": 472246823,
  "step": 15,
  "action": "Place Table",
  "best_alt_action": "Move South",
  "teacher_gap_rollout": 3.9718,
  "class": "critical_correct"
}
```

负 gap 用来提醒：Teacher 做过某动作，不等于它该进 Skill。

---

### 3.4 构造 LLM Scientist 输入

**做什么**：把 DSL、状态样例、关键状态、历史规则等打包成 Scientist 的 context。

**输入片段（示意）**：

```json
{
  "task": "Propose executable hypotheses that describe the PPO-AD teacher's decision policy.",
  "dsl": {
    "allowed_fields": [
      "front", "health", "food", "drink", "energy",
      "inv_wood", "has_wood_pickaxe",
      "visible_stone", "dist_stone", "dir_stone",
      "ach_make_wood_pickaxe"
    ],
    "forbidden_fields": [
      "action", "reward", "seed", "step",
      "next_achievement", "next_interaction_target"
    ],
    "ops": ["==", "!=", "<", "<=", ">", ">="]
  }
}
```

说明：`allowed_fields` 这里只是截断示意。真实 context 里共有 **113** 个允许字段，主要包括：

- `ach_*`（成就是否解锁）
- `visible_*` / `dist_*` / `dir_*`（可见性、距离、方向）
- `has_*` / `inv_*`（持有与数量）
- `front`、`health`、`food`、`drink`、`energy`、`is_night`、`sleeping` 等

---

### 3.5 LLM 提出 Hypothesis DSL

**做什么**：Scientist（GPT-6.1-Sol）每轮提出 10 条可执行假设。只能输出合法 JSON，不能自己发明字段。

**输出样例**：

```json
{
  "hypothesis_id": "hyp_first_stone_progress_from_grass_with_close_stone_after_wood_tools",
  "condition": {
    "all": [
      {"field": "front", "op": "==", "value": "grass"},
      {"field": "ach_collect_stone", "op": "==", "value": false},
      {"field": "has_wood_pickaxe", "op": "==", "value": true},
      {"field": "has_wood_sword", "op": "==", "value": true},
      {"field": "dist_stone", "op": "<=", "value": 3},
      {"field": "sleeping", "op": "==", "value": false}
    ]
  },
  "behavior": {
    "type": "next_achievement",
    "target": "collect_stone"
  }
}
```

Behavior 可以是下面四种，分别表示规则在预测什么：

| 类型 | 含义 | 例子 |
|---|---|---|
| `action` | 预测**当前这一步**应执行的单个动作 | `{"type":"action","action":"Do"}` |
| `action_distribution` | 预测当前这一步属于一个**可接受动作集合**即可 | `{"type":"action_distribution","actions":["Move North","Move South"]}` |
| `next_interaction_target` | 预测接下来不久会去交互的对象，不一定是当前一步动作 | `{"type":"next_interaction_target","target":"water"}` |
| `next_achievement` | 预测固定展望窗口内下一个新解锁的成就 | `{"type":"next_achievement","target":"collect_stone"}` |

更具体一点：

- `action`：最直接。条件命中时，Teacher 这一步就该做这个动作。
- `action_distribution`：允许等价动作。例如“往北或往南都行”，Teacher 做集合里任一动作都算命中。
- `next_interaction_target`：看 Decision DB 的 hindsight 标签。默认在未来 **64 步**内，下一个交互目标是什么（如 `water` / `tree` / `stone_pickaxe`）。它描述“接下来要去碰什么”，不是立刻输出 `Move West`。
- `next_achievement`：同样用 hindsight 标签。默认在未来 **64 步**内，下一个新解锁成就是什么。这是观察到的进度事件，不是 Teacher 脑子里的隐藏意图。

后两种更抽象：`next_interaction_target` 会写成 “prioritize interacting with …”；`next_achievement` 现已写明具体 target（如 `pursue \`collect_wood\` as the next achievement`），但仍不给一步动作，对小模型往往不如直接 `action` 好用。

约束：最多 8 个谓词，嵌套深度 ≤ 3，动作必须属于 Crafter 的 17 个合法动作。

---

### 3.6 程序验证与 Fidelity

**做什么**：在 exploration 上统计规则是否真的描述 Teacher。

\[
\mathrm{support}(h)=|\{x:C(x)=1,\ B(x)\text{ eligible}\}|
\]

\[
\mathrm{fidelity}(h)=
\frac{|\{x:C(x)=1,\ B(x)\text{ matches Teacher}\}|}
{\mathrm{support}(h)}
\]

**输出样例**：

```json
{
  "hypothesis_id": "hyp_interact_with_first_cow_without_sword_despite_high_food",
  "support": 372,
  "independent_seeds": 129,
  "hits": 328,
  "fidelity": 0.8817,
  "support_action_histogram": {
    "Do": 328,
    "Move East": 9,
    "Move South": 9,
    "Move North": 7,
    "Place Table": 18
  }
}
```

---

### 3.7 Revision 反例与一次有界修订

**做什么**：在 revision split 找“条件命中但 Teacher 没按规则做”的例子，再让 LLM 修一轮。每条最多 12 个反例，每 cycle 只 refine 一次，避免过拟合。

**反例包样例**：

```json
{
  "hypothesis_id": "hyp_first_coal_progress_from_path_despite_close_stone",
  "support": 45,
  "hits": 24,
  "misses": 21,
  "fidelity": 0.5333,
  "miss_outcome_histogram": {
    "place_stone": 11,
    "collect_stone": 4,
    "place_table": 2
  },
  "examples": [
    {
      "front": "path",
      "inventory": {"stone": 1, "wood": 4, "wood_pickaxe": 1},
      "nearby": {
        "stone": {"distance": 1, "direction": "west"},
        "coal": {"distance": 2, "direction": "north-east"}
      },
      "actual_behavior_outcome": "place_table"
    }
  ]
}
```

---

### 3.8 谓词消融与简化

**做什么**：LLM 提出的条件经常比数据真正需要的更复杂。这一步用程序做**贪心谓词消融**：在不伤 fidelity / support 的前提下，尽量删掉多余条件，只保留必要谓词。

只对顶层 `{"all": [p1, p2, ...]}` 形式的条件做消融；若只剩 1 个谓词就停止。behavior 不变，只改 condition。

#### 怎么做

对每条 hypothesis：

1. **先算原规则分数**  
   在 `exploration` 和 `revision` 两个 split 上分别计算：
   - `support`：条件命中且 behavior 可评估的样本数；
   - `independent_seeds`：这些样本覆盖多少独立 seed；
   - `fidelity`：其中 Teacher 行为与规则预测一致的比例。

2. **尝试删掉某一个谓词**  
   设当前条件是 `all = [p1, p2, ..., pk]`。依次尝试删掉其中一个 `pi`，得到候选条件：
   - 若还剩多个谓词：`all = [除 pi 外的其余谓词]`；
   - 若只剩一个：直接变成那个单谓词。

3. **用同一套阈值检查候选是否可接受**  
   对每个 split，候选必须同时满足：
   - `fidelity ≥ max(0.55, 原规则 fidelity − 0.03)`  
     即绝对不低于 0.55，且相对原规则最多掉 0.03；
   - `support` 不低于原规则；
   - `independent_seeds` 不低于原规则。  
   任一 split 不满足，就拒绝这次删除。

4. **在所有可接受删除里，贪心选最好的一次**  
   排序优先级：
   1. 两个 split 的 `support` 总增益最大；
   2. 若并列，平均 `fidelity` 更高；
   3. 再并列，优先删更靠前的谓词。  
   选中后真正删掉这个谓词，更新当前条件。

5. **重复 2–4，直到删不动**  
   也就是：再也找不到“删了还不掉分”的谓词，或条件只剩 1 个谓词。

默认阈值：

```text
min_fidelity = 0.55
max_fidelity_drop = 0.03
消融数据 = exploration + revision
```

#### 简化示意

```json
{
  "before": {
    "all": [
      {"field": "front", "op": "==", "value": "cow"},
      {"field": "ach_eat_cow", "op": "==", "value": false},
      {"field": "has_wood_sword", "op": "==", "value": false},
      {"field": "food", "op": ">=", "value": 8},
      {"field": "is_night", "op": "==", "value": false}
    ]
  },
  "after": {
    "field": "front",
    "op": "==",
    "value": "cow"
  },
  "behavior": {
    "type": "action",
    "action": "Do"
  }
}
```

含义：后面几个条件被逐个证明“可删”，最后只剩 `front == cow → Do`。  
这说明 LLM 给的解释性条件，常常比数据真正支持的条件更细；消融是在用程序把规则压到更短、更稳的版本。

---

### 3.9 多轮 discovery、去重与筛选

**做什么**：每轮轮转 evidence → 提 10 条 → 评估 → 反例 → 修订 → 简化 → 合并进 Registry。最后做结构去重，并按证据状态筛选。

```text
127 raw simplified candidates
    ↓ 结构去重
120 unique candidates
    ↓ 证据筛选
54 keep + 18 undercovered + 48 reject
    ↓ 编译 keep + undercovered
72-rule experimental Skill
```

---

### 3.10 编译英文静态 Skill

**做什么**：把结构化规则翻译成 Student 实际读取的英文规则书。

结构化输入：

```json
{
  "condition": {
    "all": [
      {"field": "front", "op": "==", "value": "table"},
      {"field": "inv_stone", "op": "==", "value": 1},
      {"field": "has_coal", "op": "==", "value": true},
      {"field": "has_wood", "op": "==", "value": true},
      {"field": "ach_make_stone_pickaxe", "op": "==", "value": false}
    ]
  },
  "behavior": {
    "type": "action",
    "action": "Make Stone Pickaxe"
  }
}
```

编译输出：

```text
When facing a table with coal, stone, and wood, use `Make Stone Pickaxe`
if it has not yet been crafted.
```

评估时采用 static 注入：整本 Skill 作为固定 QA-DAG 的知识上下文，每个问题都能看到整本 Skill。

每个环境 step 固定问 9 个问题（可见物体、上一步、条件是否满足、子任务、候选动作、最终动作等），最后映射到 17 个 Crafter 动作之一。

---

## 4. Skill 可以怎样人为分类

程序筛选只回答“证据够不够”；人工还需要再看“这条到底算什么知识”。
下面是按**语义**重新人工归类后的类别；它和早期自动标签不完全相同。

| 类型 | 含义 | 数量 |
|---|---|---:|
| `policy_skill` | 有真正取舍：多种资源/敌人/时机之间做优先级，或目标不在正前方。 | 23 |
| `basic_mechanics` | 接近明确游戏机制：面前物体的标准交互（如 zombie/cow/tree/stone → `Do`），或配方式放置/制造。 | 15 |
| `basic_navigation` | 主要就是朝某个可见目标方向移动。 | 15 |
| `basic_affordance` | 面前是什么，就优先与什么交互；预测目标几乎等于 `front`。 | 8 |
| `下一成就目标` | 只预测“接下来要解锁哪个成就”，**不给出具体一步动作**。 | 11 |

也可以按证据状态看：

| 状态 | 含义 | 数量 |
|---|---|---:|
| keep | 当前阈值下证据足够 | 54 |
| undercovered | fidelity 看起来不错，但 support / seed 不够 | 18 |

72 条实验 Skill = keep + undercovered。

特别说明 **下一成就目标**：

- 这类规则的 behavior 类型是 `next_achievement`；
- 它只说“未来约 64 步内下一个新解锁成就是什么”；
- **不告诉 Student 现在该 `Move` 还是 `Do`**；
- 所以单独看英文 Skill 时会觉得“啥也没说”。这不是策略细节，更像进度标签。

筛选全程只看结构化规则，例如：

```json
{
  "condition": { "...": "..." },
  "behavior": { "type": "next_achievement", "target": "collect_wood" }
}
```

程序在 Decision DB 上检查：条件命中时，hindsight 标签 `next_achievement` 是否等于这个明确 target。  
例如第 48 条对 `collect_wood` 的 fidelity 约 0.98、support 51，所以能过 keep。

早期英文 Skill 曾误编译成笼统的 `pursue the next achievement implied by this state`，丢掉了具体 target。  
**现已全部重译**：英文写成 `pursue \`collect_wood\` as the next achievement.`，附录中文也按结构化条件重新生成。

如果以后要压缩 Skill，更合理的顺序是：先保留真正的 `policy_skill`，再考虑 mechanics / navigation / affordance；`下一成就目标` 即使写明了 target，也仍然不给具体一步动作，对小模型帮助可能有限。

---

## 5. 实验

### 5.1 两个指标分别代表什么

**Progression**

一局里解锁了多少种不同成就，再除以 22：

\[
\mathrm{progression}_e=\frac{k_e}{22}
\]

对多局取平均，再乘 100 写成百分比。它衡量的是：

> 平均每局开了多少成就。

**Official Crafter Score（几何分数）**

先算每项成就在全部 episode 上的成功率 \(s_i\in[0,100]\)，再做：

\[
\mathrm{Score}=
\exp\left(
\frac{1}{22}\sum_{i=1}^{22}\log(1+s_i)
\right)-1
\]

它衡量的是：

> 22 项成就是否被较均衡地覆盖。

所以 Progression 接近、Score 更高，通常意味着：平均每局成就数差不多，但覆盖到了更多种类、尤其是更稀有的成就。

主表比较五组：

1. 原始模型  
2. 原始模型 + DAG  
3. 手册 + 动作条件 + DAG  
4. 蒸馏 28 条 + DAG  
5. 蒸馏 72 条 + DAG  

每组 20 episodes。8B 的 72 条取 run 1；32B 的 72 条尚未跑完，先留空。

---

### 5.2 Qwen3-8B

| 配置 | Progression | Official Score |
|---|---:|---:|
| 原始模型 | 8.41% ± 0.74 | 1.29 |
| 原始模型 + DAG | 15.45% ± 1.22 | 1.80 |
| 手册 + 动作条件 + DAG | 25.45% ± 2.86 | 9.39 |
| 蒸馏 28 条 + DAG | 17.05% ± 1.60 | 2.70 |
| 蒸馏 72 条 + DAG，run 1 | 15.00% ± 2.43 | 4.86 |

简要说明：

- DAG 本身已经把 Progression 从 8.41% 提到 15.45%。
- 手册 + 动作条件 + DAG 仍然最强。
- 蒸馏 28 条相对“只有 DAG”略好。
- 蒸馏 72 条 run 1 的 Progression 接近“只有 DAG”，但 Official Score 更高（4.86），说明偶尔覆盖到更深/更广的成就。

---

### 5.3 Qwen3-32B

| 配置 | Progression | Official Score |
|---|---:|---:|
| 原始模型 | 18.18% ± 2.01 | 4.60 |
| 原始模型 + DAG | 12.73% ± 0.89 | 1.37 |
| 手册 + 动作条件 + DAG | 37.05% ± 3.33 | 16.53 |
| 蒸馏 28 条 + DAG | 28.18% ± 3.29 | 13.04 |
| 蒸馏 72 条 + DAG |  |  |

简要说明：

- 手册 + 动作条件 + DAG 仍最强。
- 蒸馏 28 条明显强于原始模型与“只有 DAG”。
- 蒸馏 72 条尚未完成，主表先留空。

---

## 附录：最终 72 条 Skill（人工重分类 + 中英对照）

以下同时给出：

1. 纠正后的**英文 Skill**（`next_achievement` 已写明具体 target）；
2. 按结构化条件/行为生成的**中文翻译**（便于阅读，不是实验输入）。

编号仍用 1–72，并标注 `[keep]` / `[undercovered]`。

实验注入时的总说明：

```text
# Crafter Behavior Skill

Use the most specific matching rule below. If no rule matches, fall back to the base policy. These rules describe action preferences only; they do not define whether an action is legal.
```

分类汇总：

| 语义类型 | keep | undercovered | 合计 |
|---|---:|---:|---:|
| 策略技能 | 13 | 10 | 23 |
| 基础机制 | 13 | 2 | 15 |
| 基础导航 | 14 | 1 | 15 |
| 基础交互 | 8 | 0 | 8 |
| 下一成就目标 | 6 | 5 | 11 |

### A.1 策略技能（policy_skill）：23 条

有真正取舍：多种资源/敌人/时机之间做优先级，或目标不在正前方。

其中 keep 13 条，undercovered 10 条。

**英文**

```text
1. [keep] When holding a sapling, before placing plant, before collecting wood, use `Place Plant`.
2. [keep] When holding exactly one sapling, before collecting wood, before placing plant, use `Place Plant`.
5. [keep] When facing cow, holding wood sword, holding exactly one wood, during the day, while awake, use `Do`.
6. [keep] When facing grass, holding wood sword, while awake, with cow to the South, with a cow exactly 2 steps away, with no tree visible, use `Move South`.
9. [keep] When facing grass, holding wood sword, before collecting drink, while awake, with water to the West, with a water exactly 2 steps away, with no cow visible, with no stone visible, use `Move West`.
10. [keep] When before collecting wood, with no tree visible, with cow to the South-West, with no stone visible, with no sand visible, use `Move South`.
11. [keep] When facing path, holding at least 2 stone, holding at most 3 stone, with coal to the North, with a coal exactly 2 steps away, with no skeleton visible, use `Move North`.
12. [keep] When facing grass, holding wood sword, before eating cow, during the day, with no tree visible, with cow to the North-West, use one of `Move North`, `Move West`.
14. [keep] When facing path, holding exactly 4 stone, holding coal, before placing furnace, with no zombie visible, with no skeleton visible, prioritize interacting with `furnace`.
15. [keep] When facing grass, holding a sapling, before placing plant, at night, with food ≥ 2, use `Place Plant`.
17. [keep] When facing path, holding exactly one stone, with no coal, holding at least 1 wood, before placing stone, use `Place Stone`.
18. [keep] When no wood pickaxe, with tree to the South-East, with cow to the East, use one of `Move East`, `Move South`.
19. [keep] When facing grass, with a tree within 2 steps, prioritize interacting with `tree`.
55. [undercovered] When facing table, with no wood, with no wood pickaxe, with tree to the West, with table to the South, use `Move West`.
56. [undercovered] When facing furnace, holding stone pickaxe, before collecting drink, while awake, with water visible, with a water within 1 step, with drink ≤ 2, prioritize interacting with `water`.
57. [undercovered] When facing grass, holding wood pickaxe, with no wood, before collecting coal, while awake, with table to the North-East, with coal to the West, with a coal within 3 steps, use `Move West`.
58. [undercovered] When facing path, holding stone sword, with skeleton to the East, with a skeleton within 2 steps, with health ≥ 5, with no zombie visible, use `Move East`.
59. [undercovered] When facing grass, holding wood sword, before eating cow, while awake, with cow to the South, with no tree visible, with no stone visible, use `Move South`.
60. [undercovered] When facing grass, holding wood sword, while awake, with tree to the South-East, with cow to the West, with a cow within 2 steps, with a table more than 1 step away, use `Move West`.
61. [undercovered] When facing grass, before eating cow, with cow to the South, with no tree visible, with no stone visible, with energy=9, use `Move South`.
62. [undercovered] When facing grass, holding wood sword, before collecting drink, with water to the West, with a water within 2 steps, with no tree visible, with no stone visible, prioritize interacting with `water`.
63. [undercovered] When facing grass, holding wood pickaxe, before collecting stone, with a stone within 2 steps, with a tree more than 2 steps away, with a coal more than 2 steps away, with a cow more than 2 steps away, with a skeleton more than 2 steps away, prioritize interacting with `stone`.
64. [undercovered] When facing grass, holding exactly 2 wood, after collecting drink, after eating cow, with tree to the East, with a tree exactly 2 steps away, with no stone visible, prioritize interacting with `tree`.
```

**中文**

```text
1. [keep] 持有树苗，还没放置植物，还没收集木头时，使用 `Place Plant`。
2. [keep] 手里正好 1 树苗，还没收集木头，还没放置植物时，使用 `Place Plant`。
5. [keep] 正面对着牛，持有木剑，手里正好 1 木头，白天，醒着时，使用 `Do`。
6. [keep] 正面对着草地，牛在南边，牛正好 2 步，持有木剑，看不见树，醒着时，使用 `Move South`。
9. [keep] 正面对着草地，水在西边，水正好 2 步，还没收集饮水，持有木剑，醒着，看不见牛，看不见石头时，使用 `Move West`。
10. [keep] 还没收集木头，看不见树，牛在西南边，看不见石头，看不见沙子时，使用 `Move South`。
11. [keep] 正面对着小路，煤在北边，煤正好 2 步，手里至少 2 石头，手里至多 3 石头，看不见骷髅时，使用 `Move North`。
12. [keep] 正面对着草地，看不见树，持有木剑，还没吃牛，牛在西北边，白天时，使用 `Move North`、`Move West` 之一。
14. [keep] 正面对着小路，手里正好 4 石头，还没放置熔炉，持有煤，看不见僵尸，看不见骷髅时，优先去与 `furnace` 交互。
15. [keep] 正面对着草地，持有树苗，还没放置植物，夜晚，食物 ≥ 2 时，使用 `Place Plant`。
17. [keep] 正面对着小路，手里正好 1 石头，还没放置石头，没有煤，手里至少 1 木头时，使用 `Place Stone`。
18. [keep] 没有木镐，树在东南边，牛在东边时，使用 `Move East`、`Move South` 之一。
19. [keep] 正面对着草地，树距离 ≤ 2 时，优先去与 `tree` 交互。
55. [undercovered] 正面对着工作台，没有木头，没有木镐，树在西边，工作台在南边时，使用 `Move West`。
56. [undercovered] 正面对着熔炉，还没收集饮水，看得见水，水距离 ≤ 1，饮水 ≤ 2，持有石镐，醒着时，优先去与 `water` 交互。
57. [undercovered] 正面对着草地，工作台在东北边，煤在西边，煤距离 ≤ 3，持有木镐，没有木头，还没收集煤，醒着时，使用 `Move West`。
58. [undercovered] 正面对着小路，骷髅在东边，骷髅距离 ≤ 2，持有石剑，生命 ≥ 5，看不见僵尸时，使用 `Move East`。
59. [undercovered] 正面对着草地，牛在南边，还没吃牛，持有木剑，看不见树，看不见石头，醒着时，使用 `Move South`。
60. [undercovered] 正面对着草地，树在东南边，牛在西边，牛距离 ≤ 2，持有木剑，工作台距离 > 1，醒着时，使用 `Move West`。
61. [undercovered] 正面对着草地，牛在南边，还没吃牛，看不见树，看不见石头，体力 = 9 时，使用 `Move South`。
62. [undercovered] 正面对着草地，水在西边，水距离 ≤ 2，还没收集饮水，持有木剑，看不见树，看不见石头时，优先去与 `water` 交互。
63. [undercovered] 正面对着草地，还没收集石头，持有木镐，石头距离 ≤ 2，树距离 > 2，煤距离 > 2，牛距离 > 2，骷髅距离 > 2 时，优先去与 `stone` 交互。
64. [undercovered] 正面对着草地，树在东边，树正好 2 步，手里正好 2 木头，已经收集饮水，已经吃牛，看不见石头时，优先去与 `tree` 交互。
```

### A.2 基础机制（basic_mechanics）：15 条

接近明确游戏机制：面前物体的标准交互（如 zombie/cow/tree/stone → `Do`），或配方式放置/制造。

其中 keep 13 条，undercovered 2 条。

**英文**

```text
3. [keep] When facing skeleton, with no stone sword, use `Do`.
4. [keep] When facing zombie, use `Do`.
7. [keep] When facing cow, use `Do`.
8. [keep] When facing cow, holding stone sword, use `Do`.
13. [keep] When facing path, holding exactly 4 stone, with no stone pickaxe, before placing furnace, with a table more than 1 step away, use `Place Furnace`.
16. [keep] When facing path, holding exactly 4 stone, with no stone pickaxe, before placing furnace, with no zombie visible, with no skeleton visible, use `Place Furnace`.
20. [keep] When holding exactly 3 wood, with no wood pickaxe, use `Place Table`.
21. [keep] When facing grass, holding wood pickaxe, with no wood sword, holding wood, with a table within 1 step, use `Make Wood Sword`.
22. [keep] When facing tree, use `Do`.
23. [keep] When facing table, holding exactly one stone, holding wood, holding stone pickaxe, before crafting a stone sword, use `Make Stone Sword`.
24. [keep] When facing table, holding exactly one stone, holding exactly one wood, holding coal, before crafting a stone pickaxe, during the day, use `Make Stone Pickaxe`.
25. [keep] When facing table, holding coal, holding stone, holding wood, before collecting drink, before crafting a stone pickaxe, use `Make Stone Pickaxe`.
54. [keep] When facing stone, use `Do`.
65. [undercovered] When facing stone, holding exactly 2 stone, with no stone pickaxe, with coal to the North, use `Do`.
72. [undercovered] When facing grass, holding exactly 2 wood, with no wood pickaxe, before placing table, during the day, with tree to the South-East, with a tree at least 2 steps away, with cow to the North-East, use `Place Table`.
```

**中文**

```text
3. [keep] 正面对着骷髅，没有石剑时，使用 `Do`。
4. [keep] 正面对着僵尸时，使用 `Do`。
7. [keep] 正面对着牛时，使用 `Do`。
8. [keep] 正面对着牛，持有石剑时，使用 `Do`。
13. [keep] 正面对着小路，手里正好 4 石头，还没放置熔炉，没有石镐，工作台距离 > 1 时，使用 `Place Furnace`。
16. [keep] 正面对着小路，手里正好 4 石头，还没放置熔炉，没有石镐，看不见僵尸，看不见骷髅时，使用 `Place Furnace`。
20. [keep] 手里正好 3 木头，没有木镐时，使用 `Place Table`。
21. [keep] 正面对着草地，工作台距离 ≤ 1，持有木镐，没有木剑，持有木头时，使用 `Make Wood Sword`。
22. [keep] 正面对着树时，使用 `Do`。
23. [keep] 正面对着工作台，手里正好 1 石头，持有木头，持有石镐，还没制作石剑时，使用 `Make Stone Sword`。
24. [keep] 正面对着工作台，手里正好 1 石头，手里正好 1 木头，持有煤，还没制作石镐，白天时，使用 `Make Stone Pickaxe`。
25. [keep] 正面对着工作台，还没收集饮水，持有煤，持有石头，持有木头，还没制作石镐时，使用 `Make Stone Pickaxe`。
54. [keep] 正面对着石头时，使用 `Do`。
65. [undercovered] 正面对着石头，煤在北边，手里正好 2 石头，没有石镐时，使用 `Do`。
72. [undercovered] 正面对着草地，手里正好 2 木头，还没放置工作台，没有木镐，树在东南边，树距离 ≥ 2，牛在东北边，白天时，使用 `Place Table`。
```

### A.3 基础导航（basic_navigation）：15 条

主要就是朝某个可见目标方向移动。

其中 keep 14 条，undercovered 1 条。

**英文**

```text
26. [keep] When facing path, before defeating skeleton, with skeleton to the West, with a skeleton within 2 steps, use `Move West`.
27. [keep] When facing path, before placing furnace, with skeleton to the West, with a skeleton exactly 2 steps away, use `Move West`.
28. [keep] When before collecting wood, before collecting a sapling, with tree to the South-West, use one of `Do`, `Move South`.
29. [keep] When facing stone, holding at least 2 wood, holding exactly one stone, with stone to the East, with health ≥ 8, use one of `Do`, `Move East`, `Make Stone Pickaxe`.
30. [keep] When facing stone, with stone to the North, with energy=9, use one of `Do`, `Move North`.
31. [keep] When facing grass, before collecting wood, before collecting a sapling, before eating cow, with tree to the South, with a tree exactly 2 steps away, use `Move South`.
32. [keep] When facing grass, with no wood pickaxe, with tree to the North, use `Move North`.
33. [keep] When facing path, with skeleton to the South, use `Move South`.
34. [keep] When facing grass, with no wood, with no wood pickaxe, after placing table, with tree to the South, use `Move South`.
35. [keep] When facing grass, with no wood sword, with tree to the North, use `Move North`.
36. [keep] When facing grass, holding wood pickaxe, before crafting a wood sword, with coal to the South, use `Move South`.
37. [keep] When facing grass, holding exactly 4 wood, with no stone, during the day, with stone to the North-East, use one of `Move North`, `Move East`.
38. [keep] When facing grass, holding wood sword, with no stone sword, before defeating zombie, while awake, with zombie to the North, use `Move North`.
39. [keep] When facing grass, holding exactly 4 wood, with no stone, with stone to the North-East, use one of `Move North`, `Move East`.
66. [undercovered] When facing grass, before collecting wood, after placing plant, with tree to the North, use `Move North`.
```

**中文**

```text
26. [keep] 正面对着小路，还没打倒骷髅，骷髅在西边，骷髅距离 ≤ 2 时，使用 `Move West`。
27. [keep] 正面对着小路，骷髅在西边，骷髅正好 2 步，还没放置熔炉时，使用 `Move West`。
28. [keep] 还没收集木头，还没收集树苗，树在西南边时，使用 `Do`、`Move South` 之一。
29. [keep] 正面对着石头，石头在东边，手里至少 2 木头，手里正好 1 石头，生命 ≥ 8 时，使用 `Do`、`Move East`、`Make Stone Pickaxe` 之一。
30. [keep] 正面对着石头，石头在北边，体力 = 9 时，使用 `Do`、`Move North` 之一。
31. [keep] 正面对着草地，还没收集木头，还没收集树苗，还没吃牛，树在南边，树正好 2 步时，使用 `Move South`。
32. [keep] 正面对着草地，没有木镐，树在北边时，使用 `Move North`。
33. [keep] 正面对着小路，骷髅在南边时，使用 `Move South`。
34. [keep] 正面对着草地，已经放置工作台，没有木头，没有木镐，树在南边时，使用 `Move South`。
35. [keep] 正面对着草地，没有木剑，树在北边时，使用 `Move North`。
36. [keep] 正面对着草地，煤在南边，持有木镐，还没制作木剑时，使用 `Move South`。
37. [keep] 正面对着草地，手里正好 4 木头，没有石头，石头在东北边，白天时，使用 `Move North`、`Move East` 之一。
38. [keep] 正面对着草地，还没打倒僵尸，持有木剑，没有石剑，僵尸在北边，醒着时，使用 `Move North`。
39. [keep] 正面对着草地，手里正好 4 木头，没有石头，石头在东北边时，使用 `Move North`、`Move East` 之一。
66. [undercovered] 正面对着草地，还没收集木头，已经放置植物，树在北边时，使用 `Move North`。
```

### A.4 基础交互（basic_affordance）：8 条

面前是什么，就优先与什么交互；预测目标几乎等于 `front`。

其中 keep 8 条，undercovered 0 条。

**英文**

```text
40. [keep] When facing stone, holding stone pickaxe, before placing furnace, with diamond visible, prioritize interacting with `stone`.
41. [keep] When facing zombie, prioritize interacting with `zombie`.
42. [keep] When facing tree, prioritize interacting with `tree`.
43. [keep] When facing stone, with energy=8, prioritize interacting with `stone`.
44. [keep] When facing stone, with energy ≥ 6, prioritize interacting with `stone`.
45. [keep] When facing water, before collecting drink, with no skeleton visible, prioritize interacting with `water`.
46. [keep] When facing stone, with no stone pickaxe, with stone to the West, with diamond visible, prioritize interacting with `stone`.
47. [keep] When facing iron, prioritize interacting with `iron`.
```

**中文**

```text
40. [keep] 正面对着石头，持有石镐，还没放置熔炉，看得见钻石时，优先去与 `stone` 交互。
41. [keep] 正面对着僵尸时，优先去与 `zombie` 交互。
42. [keep] 正面对着树时，优先去与 `tree` 交互。
43. [keep] 正面对着石头，体力 = 8 时，优先去与 `stone` 交互。
44. [keep] 正面对着石头，体力 ≥ 6 时，优先去与 `stone` 交互。
45. [keep] 正面对着水，还没收集饮水，看不见骷髅时，优先去与 `water` 交互。
46. [keep] 正面对着石头，石头在西边，没有石镐，看得见钻石时，优先去与 `stone` 交互。
47. [keep] 正面对着铁时，优先去与 `iron` 交互。
```

### A.5 下一成就目标（原 progression_pattern）：11 条

只预测“接下来要解锁哪个成就”，不给出具体一步动作。英文现已写明具体 target（如 `collect_wood`），不再用笼统的 pursue the next achievement implied by this state。

其中 keep 6 条，undercovered 5 条。

**英文**

```text
48. [keep] When facing plant, before collecting wood, with tree toward West, with a tree within 5 steps, pursue `collect_wood` as the next achievement.
49. [keep] When facing grass, holding a sapling, before placing plant, after defeating zombie, with a tree more than 2 steps away, pursue `place_plant` as the next achievement.
50. [keep] When facing grass, during the day, with a skeleton exactly 2 steps away, with a stone more than 2 steps away, pursue `defeat_skeleton` as the next achievement.
51. [keep] When facing path, holding exactly 4 stone, before placing furnace, pursue `place_furnace` as the next achievement.
52. [keep] When facing path, holding exactly one stone, with no stone pickaxe, before placing stone, before collecting coal, with a coal exactly 2 steps away, pursue `place_stone` as the next achievement.
53. [keep] When facing sand, holding wood pickaxe, before collecting drink, with water to the North, with a water exactly 2 steps away, with no skeleton visible, pursue `collect_drink` as the next achievement.
67. [undercovered] When holding exactly 4 stone, before placing furnace, with a tree more than 2 steps away, with a iron more than 2 steps away, with no skeleton visible, with a table more than 1 step away, pursue `place_furnace` as the next achievement.
68. [undercovered] When facing path, holding wood pickaxe, holding exactly 2 stone, before collecting coal, after placing stone, with a coal within 2 steps, with a stone more than 2 steps away, with no skeleton visible, pursue `collect_coal` as the next achievement.
69. [undercovered] When facing path, holding wood pickaxe, holding exactly 2 stone, before collecting coal, with a coal within 1 step, with a stone more than 1 step away, with a table more than 2 steps away, with no skeleton visible, pursue `collect_coal` as the next achievement.
70. [undercovered] When facing grass, holding wood sword, before collecting drink, with water to the West, with a water exactly 2 steps away, with no tree visible, with no stone visible, with no table visible, pursue `collect_drink` as the next achievement.
71. [undercovered] When facing grass, holding at least 6 wood, before collecting stone, with stone to the West, pursue `collect_stone` as the next achievement.
```

**中文**

```text
48. [keep] 正面对着植物，还没收集木头，树方向含西，树距离 ≤ 5 时，下一目标是「收集木头」(`collect_wood`)，但不指定具体一步动作。
49. [keep] 正面对着草地，持有树苗，还没放置植物，已经打倒僵尸，树距离 > 2 时，下一目标是「放置植物」(`place_plant`)，但不指定具体一步动作。
50. [keep] 正面对着草地，骷髅正好 2 步，白天，石头距离 > 2 时，下一目标是「打倒骷髅」(`defeat_skeleton`)，但不指定具体一步动作。
51. [keep] 正面对着小路，手里正好 4 石头，还没放置熔炉时，下一目标是「放置熔炉」(`place_furnace`)，但不指定具体一步动作。
52. [keep] 正面对着小路，手里正好 1 石头，还没放置石头，煤正好 2 步，还没收集煤，没有石镐时，下一目标是「放置石头」(`place_stone`)，但不指定具体一步动作。
53. [keep] 正面对着沙子，水在北边，水正好 2 步，还没收集饮水，持有木镐，看不见骷髅时，下一目标是「收集饮水」(`collect_drink`)，但不指定具体一步动作。
67. [undercovered] 还没放置熔炉，手里正好 4 石头，树距离 > 2，铁距离 > 2，看不见骷髅，工作台距离 > 1 时，下一目标是「放置熔炉」(`place_furnace`)，但不指定具体一步动作。
68. [undercovered] 正面对着小路，还没收集煤，持有木镐，已经放置石头，手里正好 2 石头，煤距离 ≤ 2，石头距离 > 2，看不见骷髅时，下一目标是「收集煤」(`collect_coal`)，但不指定具体一步动作。
69. [undercovered] 正面对着小路，还没收集煤，持有木镐，煤距离 ≤ 1，石头距离 > 1，手里正好 2 石头，工作台距离 > 2，看不见骷髅时，下一目标是「收集煤」(`collect_coal`)，但不指定具体一步动作。
70. [undercovered] 正面对着草地，水在西边，水正好 2 步，还没收集饮水，持有木剑，看不见树，看不见石头，看不见工作台时，下一目标是「收集饮水」(`collect_drink`)，但不指定具体一步动作。
71. [undercovered] 正面对着草地，手里至少 6 木头，还没收集石头，石头在西边时，下一目标是「收集石头」(`collect_stone`)，但不指定具体一步动作。
```
