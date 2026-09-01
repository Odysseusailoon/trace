# Agent 决策步粒度熵对齐 — 实验记录(2026-08-31 凌晨)

## 问题

ARPO/AEPO/Tree-GRPO 族用 rollout 中的 token 熵决定在哪分叉。这个廉价信号能不能指到我们重放测量出的因果分叉步?(MCQ token 级已测:不能。本实验补 agent 决策步粒度。)

## 方法

- 对 4 条有重放真值的轨迹(t4 失败/成功、t7 失败/成功),用**原 seed + deterministic inference 重生成**,请求加 `logprobs, top_logprobs=20`,得到生成时逐 token 的 top-20 截断熵——正是熵启发式方法在 rollout 中可见的信号。
- 复现验证:4/4 final 逐字一致;tool 调用 SQL 原文逐字符一致(transcripts 里是 fallback 解析后的表示,本实验存的是解析前原文,表示层差异)。
- 分叉真值(各轨迹自己的重放 o_d + persistence):t4_10000 → turn 0(persist .54, o 22%→0%);t4_10015 → turn 0(persist .16, o 22%→100%);t7_10003 → turn 1(persist .10, o 14%→100%);t7_10000 → **无争议步**(persist 全程 1.0,走廊型,排除出对齐统计)。
- 脚本:节点 `scripts/entropy_agent.py`,本地分析内联(见 git);数据 `data/reports/entropy_agent_raw.json`。

## 结果

### 步级对齐(n=3 条有分叉轨迹)

| 轨迹 | fork turn | argmax(meanH) | argmax(maxH) | argmax(firstH) |
|---|---|---|---|---|
| t4_10000 | 0 | 0 ✓ (.099 vs .021) | 0 ✓ | 0 ✓(.007 vs .004,噪声级) |
| t4_10015 | 0 | 0 ✓ (.023 vs .021,平) | 0 ✓ | 0 ✓(.007 vs .000,噪声级) |
| t7_10003 | 1 | **0 ✗** (.067 > .050) | **0 ✗** | 1 ✓(**.562**,唯一实信号) |

### 关键观察

1. **有真信号的一例**:t7_10003 turn1 首 token(`The`,发工具还是直接作答的决策点)H=0.562,与 persistence 0.10 方向一致;同一边界在注定路径 t7_10000 上 H=0.000(persistence 1.0)。t4_10000 的 intdiv 因果 token ` COUNT`(分位 0.99)、` /`(0.96)也在高熵区。**熵不是零信号。**
2. **路径条件性(最重要的失配)**:t4_10015 走的是 16%-persistence 少数路径——测量口径下全数据最有争议的决策(switch rate 84%)——但它自己这条 rollout 全程熵平(fork turn meanH 0.023 ≈ 非分叉 turn 0.021)。**熵是沿单条路径的局部量,分叉是状态的分布属性;熵启发式在"进入了光滑少数路径"的 rollout 上什么都看不见。**
3. **装饰性峰值**:各轨迹最高熵 token 多为别名/标点(` Average` 0.906、`Track` 0.757、`.` 0.597),因果 token 排其后——熵驱动的分叉预算会花在措辞变异上。
4. **严重度全盲**:t7 走廊态 92% 注定失败,meanH 0.091、firstH 0.001。token 熵完全反映不了 o_d 级别的命运信息。

## 结论口径(写论文用)

> Token entropy is a **path-conditional, severity-blind** local signal; measured forks (o_d) are a **state property**. On agent decision steps the two partially overlap — entropy can flag a tool-vs-answer decision *on the trajectory that took the minority branch* — but it goes blind on smooth minority paths, spends its budget on decorative variation, and cannot rank fate-lock severity.

比 MCQ 侧的"entropy is a poor proxy"一刀切更准,也更防审稿人反例。**Plan B Gate 1 含义**:对齐率不是 0 也不是 1——熵拿得到部分信号但拿不到状态级分叉,Gate 1 的"显著 < 1"方向上成立,但 Plan B 若做,方法叙事应改为"熵可作预筛、测量作确认"的两级结构。

## 限制

- n=4 episode(3 条有分叉)、11 turn;top-20 截断熵;enable_thinking=False;单模型(Qwen3-8B)单温度(T=1)。
- 步级 argmax 统计在 2-turn 轨迹上接近抛硬币,真正有分辨力的只有 t7_10003(5 turn)。
- 硬化路径:对更多重放轨迹批量跑(每条只需一次带 logprobs 的重生成,便宜),W2 可选。

## 成本

节点本次唤醒 ~25 min(含权重重下),估 +$0.5。数据已拉回本地,节点已 STOP。
