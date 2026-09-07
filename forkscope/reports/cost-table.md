# 成本表:naive FPA vs forkscope(2026-08-31)

> naive FPA = HF 式重采样,无 KV 复用:每条续写请求全额 prefill 自己的前缀(Forking Fast 开源实现即此形态,手动 DynamicCache 裁剪只在单 case 内有效)。
> forkscope = 同一请求流打在 SGLang RadixAttention 上,嵌套前缀共享 trunk。
> decode 两边完全相同(相同 seeds、逐 token 一致,A/B 已验证)——差异全部在 prefill。

## 1. 决策步重放的精确 token 记账(本地可复算,`scripts/make_cost_table.py`)

前缀长度 = 逐边界用 Qwen3-8B chat template(含 tools schema)精确 tokenize;K=50。

| 轨迹 | 边界数 D | 各边界前缀 tokens | naive prefill | trunk-once | ↓(理想) | ↓(实测 98.5% 命中锚定) |
|---|---|---|---|---|---|---|
| t4_10000 | 2 | 572, 643 | 60,750 | 643 | **94.5×** | 66.7× |
| t4_10015 | 2 | 572, 660 | 61,600 | 660 | 93.3× | 66.7× |
| t7_10000 | 2 | 563, 636 | 59,950 | 636 | 94.3× | 66.7× |
| t7_10003 | 5 | 563→871 | 179,600 | 871 | **206.2×** | 66.7× |
| **合计** | 11 | — | **361,900**(5.8 PFLOP) | **2,810**(0.045 PFLOP) | **129×** | 66.7× |

- **比率结构**:理想比 ≈ (K×ΣL_d)/L_max ≈ K×D×(平均前缀/最长前缀)——**随重放深度线性增长**。5 步轨迹 206×,2 步轨迹 94×;更长的真实 agent 轨迹(20+ 步)理论上 >10³×。
- **实测锚定**:全管线 prefix cache 命中 98.5%(含冷前缀、跨 case),对应 prefill 计算 ↓66.7×——理想值的下界,实打实测到的。

## 2. 受控 A/B(radix on/off,同负载相同 seeds,Day 3 实测)

| 指标 | radix OFF(=naive) | radix ON | 比 |
|---|---|---|---|
| prefill 计算 tokens | 106.7K | 7.1K | **↓14.9×** |
| wall | 8s | 6s | 1.33× |
| decode | 逐 token 一致 | 逐 token 一致 | = |

负载:40 branches × 8 续写 × ≤128 new(token 级枚举,浅前缀)——比率低于重放场景是结构性的(浅前缀、少共享)。

## 3. 诚实注记(审稿人会问的)

1. **wall 增益 < FLOP 增益**:玩具规模下 decode 占主导,wall 只差 1.33×。prefill 份额随上下文长度增长——长轨迹 agent(数千 token 前缀 × 深重放)才是 prefill 主导的区间,也正是 FPA 的目标场景。省下的 prefill 直接兑换为并发容量/吞吐,不完全兑换为单请求延迟。
2. **66.7× 与 129× 的关系**:前者是整个 Day 2-3 请求流(含所有冷启动)的实测平均;后者是重放工作负载的理想记账。论文表里两个都放,分别标"measured (whole pipeline)"和"analytical (replay workload)"。
3. **对比对象是引擎无关的 naive,不是"关掉 radix 的 SGLang"**:Forking Fast 的 HF 实现里 KV 复用要手写且只在单 case 内成立;serving 引擎让嵌套重采样的共享自动、跨 case、免费——这是"FPA 便宜到可日常用"claim 的机制。
4. 尺度感:4 条轨迹全部重放的 naive prefill = 5.8 PFLOP ≈ 单卡 H100 十几秒——绝对量小,claim 的重点是**比率随规模的增长结构**(200 episodes × 多任务 × 长轨迹),不是这 4 条省了几毛钱。
