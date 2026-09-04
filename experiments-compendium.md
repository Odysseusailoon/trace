# forkscope 实验汇编(Experiments Compendium)

> 2026-08-31 汇编 · 覆盖至今全部已完成实验:token 级 / agent 级 / 环境 / 熵 / 系统与预算
> 写作纪律遵循 ml-paper-writing:每个实验先声明**支撑的 claim**,再给 setup / 表 / 读表指引 / 统计 / 诚实边界
> 口径纪律:t4 统一用 46%(首步 intdiv 占比)/ 0%(intdiv 簇正确率)/ 100%(patched);弃用旧分类器 63%
> Claim 索引:仪器论文三贡献 **P1** 测量层 / **P2** 发现层 / **P3** 应用层;闭环论文子命题 **C1-C5**(paper-core-claims.md)

---

## 1. 实验设施(Agent Environment)

**模型与服务**:Qwen3-8B,SGLang(`--enable-deterministic-inference`),单卡 H100(givemeanode)。全部随机性由 (任务, 步, 序号) 派生 seed 控制,断点续跑 bit 级一致,A/B 两侧 decode 逐 token 相同。

**数据库**:Chinook(音乐商店 SQLite,11 表,Artist 275 / Album 347 / Track 3503 行),查询真实执行、只读。

**三个工具**(OpenAI function-calling schema):

| 工具 | 行为 | 陷阱设计 |
|---|---|---|
| `sql_query` | 任意只读 SQL,返回前 30 行 JSON;描述附带 schema 文本 | schema 列名 prime 错误写法(t4) |
| `calculator` | 正则白名单后沙箱 eval | — |
| `web_search` | 确定性 mock,固定语料关键词匹配 | 语料含诱饵数字(t10:67.3% 紧邻正确原料 19.3/28.6) |

**任务**:t1-t10,全部带可执行 gold(对库直接算出,非人工标注)。代表性陷阱:t4 schema-priming(intdiv 截断为 1 且不报错)、t2 陷阱表、t8 off-by-one、t10 诱饵数字。

**Episode 机制**:system(分析师人设 + "The answer is" 收尾)→ 任务 → ≤8 轮 chat completions(tools、T=1.0、non-thinking、per-episode seed);工具真执行、结果回填;正则抓答案按容差判对错 + 具名错误模式识别。

**规模**:每任务 200 条独立 episode;决策步重放 K=50;token 级重采样 S=20(t0 锚点 100)。

---

## 2. Token 级实验(MCQ / Forking Fast 复现线)

### E-T1 · 决策型分叉稀有性探针 —— 支撑 P2(agent 是 FPA 自然栖息地)

201 题 MCQ 探针(MMLU 类,thinking 续写):决策型分叉检出 **~1%**。原论文(大模型)设定在 8B 上不可复现——这不是复现失败,是把 FPA 推向 agent 域的动机证据。

### E-T2 · o_t 管线复现 + 平滑(V3)—— 支撑 P1(测量层)

Forking Fast 管线逐段移植(枚举 p≥0.05 分支 / 纯祖先采样 / regex+fallback 结局提取)。两个深挖 case:

| Case | o_t 结构 | 判定 |
|---|---|---|
| college_physics[4] | 变点清晰 | 决策型(MCQ 侧少数样本) |
| virology[5] | B/C 拉锯、TVD 0.12-0.30 摆动 ×11 处 | 漂移型——平滑模型**正确地拒绝**硬找分叉 |

V3 平滑恢复(PELT-L2 + 核加权 Dirichlet pooling):合成 22/22 通过,TVD −36%,等效样本量 **3.8×**(论文报 4-15×,同向)。

### E-T3 · 统计有效性(V0/V1/V2)—— 支撑 P1(噪声模型与保证)

| 检验 | 内容 | 结果 |
|---|---|---|
| V0 | 管线自洽 | 16/16 |
| V1 | 重采样噪声 = exact iid Multinomial null | agent 侧 t4 p=0.28 / t7 p=0.09 / t10 p=0.22 / Day2-t4 p=0.19,全部无法拒绝;**MCQ 旧悬案收口**:比值 0.916/0.915 按 null 重算 p=0.41/0.31,不显著(比值带 [0.95,1.05] 是多位置尺子,T=1 场景错配) |
| V2 | replicate-TVD ∝ 1/√S | t4 斜率 **−0.493**(理论 −0.5);t7/t10 partial(小 T) |

Seed 段敏感性:t4 32/18/24%、t7 12/18/16%,无坏批。Watch-item:t7 收集批 23.5% vs 重批 15.3%(~1.9σ)轻度过散,大 T 重验已预注册。

### E-T4 · Token 级熵对齐 —— 支撑 C1 前身 / O4

`entropy_v1_followup.json`:两个 MCQ case 共 **22 个测量分叉,熵中位分位数 0.25-0.26,仅 1/22 落在熵 top-20%**;熵 top-decile 位置命中分叉窗口 **5/150**。限:2 case、token 级、top-10 截断熵。

---

## 3. Agent 级实验(决策步重放线)

### E-A1 · t4:第一条 SQL 定生死 —— 支撑 P2(决策型·首步锁定)+ C2(coverage)

200 episode,按第一条 SQL 聚类:

| 第一条 SQL 写法 | 占比 | 最终正确率 |
|---|---|---|
| `AVG(...) GROUP BY`(正确) | 18% | **100%** |
| `COUNT(*)/COUNT(DISTINCT ArtistId)`(intdiv→1) | **46%** | **0%** |
| 错表 `FROM Artist` | 4% | 0% |
| 其它 | 33% | 39% |

MI(首步; 结局) = **0.568 bits**。因果确认(镜像重放,K=50):失败轨迹 o_0[correct]=0.22 → o_1=**0.00**;成功轨迹同一 o_0 → o_1=**1.00**——同一状态、一个决策、两种命运。正确写法 persistence 仅 **0.16**(少数派,C2 的 coverage 证据)。无救机制:intdiv 返回合法的 1,零错误信号,自愈 **2/126**;两次独立采集 45%/46%,发现稳健。图:`fig_t4_mirror.png`。

**因果升级(9/1 预注册批,schema 消融网格 3×200 ep)**:列清单措辞因果驱动 intdiv,剂量反应完整——删整个 schema 描述(V1):intdiv 45.5%→**14%**、correct 17.5%→**60%**;列序重排 ArtistId 提前(V3):intdiv **65%**、correct 11%(恶化);只删 Album 行列名(V2):簇标签 intdiv 坍缩到 7.5% 但结局 1.000 仍有 **52%**——模型改写 `COUNT(DISTINCT x)/COUNT(DISTINCT y)` 掉同一个整数除法陷阱(被聚类器归入 sql:other)。**簇标签 ≠ 失败模式,结局才是**;prime 机制是"给出列结构"本身而非特定列名。"schema-as-adversarial-context" 升级为受控因果发现。重放扩样 13 条(t4 三簇 5+5+3,K=50)落盘;fork-state 续写证实 t4 d=0 前缀 50 续写未锁定(30 intdiv/12 correct/8 other)。

### E-A2 · t7:中段分叉,逐步定位 —— 支撑 P1+P2(决策型·中段)

98% episode 写同一条 SQL(MI≈0.02,首步无分歧),但 75% 把秒当分钟答。重放定位:

| d | 状态 | p_correct | persistence | 判定 |
|---|---|---|---|---|
| 0 | 写 SQL 前 | 0.08 | **1.00** | 必经之路,非分叉 |
| 1 | SQL 结果进上下文 | 0.00/0.14(败/成) | 1.0 / **0.10** | **分叉**:仅 ~10% 续写调 calculator 换算 |
| 2 | 换算结果进上下文 | 1.00 | 0.92 | 锁定 |

关键细节:失败/成功轨迹 SQL 与返回值**完全相同**,差别只在第一轮 assistant 的措辞——早期文字调制后续决策分布。图:`fig_t7_localization.png`。

**因果升级(9/1 批)**:重放扩样 8 条(wrong_unit×5、correct×3)落盘。fork-state 续写(d=1 前缀,50 续写):**50/50 wrong_unit_seconds,完全锁定**——分叉图因果层的硬证据,单轨迹 artifact 排除。

### E-A3 · t10:诱饵锁定 —— 支撑 P2(三分类扩展)

**187/200** 抄语料诱饵 67.3(全流媒体占比)而非计算 19.3/28.6=67.5。失败模式单一度 94%,与 t4 schema-priming 同族:上下文紧邻错误候选锁定结局。

**因果闭合(9/1 批,语料消融 200 ep)**:删掉语料中 "67.3%" 那句后 **correct 198/200(99.0%)**——诱饵率完全坍缩。最便宜的反事实干预($1),"诱饵锁定"从相关升级为因果。

### E-A4 · 失败三分类 —— 支撑 P2(分类学)+ C2(边界:何时可放大)

| 类型 | 案例 | o_t 结构 | 干预点 | coverage |
|---|---|---|---|---|
| 知识层 | econometrics[6] | 全程自信错,0 分叉 | 模型/prompt(需注入) | ≈0 |
| 漂移层 | virology[5] | 拉锯无落点 | 采样/聚合 | — |
| 决策·首步 | t4 | d=0 锁定 | schema/工具描述 | 18-22% |
| 决策·中段 | t7 | d=1 锁定 | 该步前上下文 | ~10% |
| 诱饵锁定 | t10 | 语料数字锁定 | 检索去混淆 | — |

### E-A5 · 分布错配发现 —— 支撑 P2(方法学,首次记录)

同一模型**裸续写**(无 chat template/tool schema)100% 写对 t4 的 SQL——schema 本身是陷阱来源;裸 input_ids 续写测的是另一个模型。Agent FPA 必须在带 tool schema 的 chat 分布里做。

---

## 4. RSI 闭环实验(应用层)

### E-R1 · 测量报告驱动自修复 —— 支撑 P3 + C1(执行级信号)

t4 分叉报告(统计事实,非日志)交给同一模型,只允许改 `sql_query` 描述,200 vs 200:

| 指标 | control | patched |
|---|---|---|
| 最终正确率 | 27.0% | **100.0%** |
| 错误写法族占比(宽口径¹) | 64% | **0%** |

¹ rsi_loop 宽口径 64% ≠ canonical 46%,口径注释见报告。Agent 诊断复述精确命中因果链("schema text primes... COUNT(DISTINCT ArtistId)")。

### E-R2 · 去泄露三臂对照 —— 支撑 P3 诚实性 + C1(信息价值分层)

便签版:agent 只写 ≤60 词警告,代码 append(结构上不可能泄露正确写法):

| 臂 | t4 correct | t4 首步错表 | t7 correct |
|---|---|---|---|
| control | 27.0% | 9.5% | 17.0% |
| 完整报告(含改法) | **100%** | 0% | — |
| 只警告(便签) | **15.5% ↓**(z≈2.8) | **32.5% ↑** | **37.5%**(2.2×,z≈4.6) |

三层结论:t7 纯定位就有效(修复动作模型本来会做);t4 纯警告**有害**(错误位移,打地鼠);因果机制授权的修复才到 100%。**测量价值分层:定位是下限,机制是上限。**

过程教训(入 discussion):v1"只能警告"约束 → 模型删光 schema → 81-97% 瘫痪 → **RSI patch 必须 append-only**(对 DGM 类普适)。

### E-R3 · Patch 生成种子稳健性(9/1 预注册批)—— 支撑 P3 稳健性 + C1 精确化

单一样本 patch 的方差洞修复:5 个独立 patch 种子 × 100 ep(原生 agentenv runner,对照臂同 runner n=100):

| 臂 | intdiv | correct | wrong_other |
|---|---|---|---|
| 对照 | 51% | 29% | 17% |
| patch 种子×5 | **1-4%**(5/5 过 <10% 预注册阈值) | 7-15% | 81-89% |

**假设成立,但故事改写**:警告便签稳健地消除 intdiv 陷阱(51%→≤4%),但 correct 不升反降(29%→7-15%)——**消除陷阱 ≠ 产生能力**,被替代的质量全部流进 wrong_other,与 E-R2"便签有害"同向。RSI claim 精确化:agent 写的诊断便签能稳健消除特定陷阱模式,正确路径仍需机制级授权。

**测量纪律记录**:本批首跑曾"全绿"——rsi_loop runner 在 server 未带 `--tool-call-parser` 时收到文本 `<tool_call>` 而非结构化 tool_calls,全部 no_answer、intdiv 率 0.00 假通过。换原生 runner 重跑才得上表。**任何全绿先验测量通道。**

---

## 5. 熵对齐(Agent 决策步粒度,8/31)

### E-E1 · 熵 vs 测量分叉 —— 支撑 C1/O4(升级版口径)

方法:原 seed + deterministic inference 重生成 4 条有重放真值的轨迹(**4/4 final 逐字复现**,SQL 逐字符一致),top-20 logprobs 取逐 token 截断熵(= ARPO 族 rollout 内可见信号)。

| 轨迹 | fork turn(真值) | argmax(meanH) | argmax(firstH) | 备注 |
|---|---|---|---|---|
| t4_10000 | 0 | 0 ✓(.099 vs .021) | 0 ✓(噪声级差距) | intdiv 因果 token ` COUNT` 熵分位 0.99 |
| t4_10015 | 0 | 0 ✓(.023 vs .021,平) | 0 ✓(噪声级) | **16%-persistence 少数路径全程熵平** |
| t7_10000 | 无争议步(走廊) | — | — | 92% 注定失败,meanH 仅 .091 |
| t7_10003 | 1 | **0 ✗** | **1 ✓(.562,唯一实信号)** | 同边界注定路径 firstH=.000 |

三个失配模式(各有实测数):**路径条件性**(最有争议的决策在自己的 rollout 上熵平)/ **装饰性峰值**(` Average` .906 > 因果 ` COUNT` .633)/ **严重度全盲**。合并口径:**token 熵是路径条件的、严重度盲的局部信号;o_d 是状态属性**。限:4 episode / 11 turn / 3 条有分叉真值。

---

## 6. 系统与成本

### E-S1 · RadixAttention 实测 —— 支撑 P1(成本层)

| 指标 | 数值 | 口径 |
|---|---|---|
| 全 pipeline 前缀命中 | **98.5%** | Day 2-3 全请求流 |
| 重放负载实测命中 | **99.2%** | 9/1 审计批 /metrics 前后差(6.53M prompt,6.47M cached)——"anchored"列升级为 measured |
| radix on/off A/B prefill 计算 | 7.1K vs 106.7K = **↓14.9×** | 同负载相同 seeds,decode 逐 token 一致;A/B 负载内前缀命中 93.3% |
| wall(玩具规模) | 6s vs 8s(1.33×) | decode 主导;时钟差随轨迹长度增长 |

### E-S2 · 成本记账表(`reports/cost-table.md`,tokenizer 精确计数)

决策步重放(K=50),naive(每请求全额 prefill)vs forkscope:

| 轨迹 | D | naive prefill | trunk-once | ↓理想 | ↓98.5% 锚定 |
|---|---|---|---|---|---|
| 2 步轨迹 ×3 | 2 | ~60K tok | ~650 | **93-95×** | 66.7× |
| t7 成功(5 步) | 5 | 179.6K | 871 | **206×** | 66.7× |
| 合计 | 11 | 361.9K(5.8 PFLOP) | 2.8K | **129×** | 66.7× |

比率结构 ≈ K×D,随重放深度线性增长。诚实注记:wall 增益 < FLOP 增益(省的是吞吐/容量);对比对象是引擎无关 naive(Forking Fast HF 形态);绝对量小,claim 在增长结构。

### 预算台账(GPU,checkpoint 制)

| 时点 | 累计 | 增量事项 |
|---|---|---|
| Day 3 晨(8/28) | $18.67 | t4/t7/t10 复现 + 决策步重放上线 |
| Day 3 傍晚 | $23.87 | RSI 闭环 + t10 诱饵 + 轨迹回传 |
| Day 3 深夜 | $24.47 | 熵对齐 MCQ 侧 + MCQ 悬案收口 |
| Day 3 深夜 II | $24.67 | RSI 去泄露三臂 |
| 8/31 凌晨 | **~$26.0** | agent 熵对齐(节点 20min,$1.34;费率 $0.0666/min,15min 最低) |
| 9/1 | **~$28.7** | 预注册审计批:schema/语料消融 800ep + 重放扩样 21 条 + forkstate dump + patch 种子×5(节点 ~40min,$2.7) |

零 GPU 完成:成本表、全部统计分析、三份 lit review、四篇精读。单卡 H100 $4/hr;周五 33% off(下批排 9/4)。

---

## 7. 总索引:实验 × Claim × 数据文件

| 实验 | 支撑 | 状态 | 数据 |
|---|---|---|---|
| E-T1 分叉稀有性 | P2 | ✅ | 201 题探针 |
| E-T2 o_t 复现+V3 | P1 | ✅ | report_*_4/5.md, forks_*.png |
| E-T3 V0/V1/V2 | P1 | ✅(t7 watch) | agent_v1v2.json, validation_*.json |
| E-T4 token 熵对齐 | C1/O4 | ✅ | entropy_v1_followup.json |
| E-A1 t4 首步锁定 | P2/C2 | ✅ | validation_t4, replay_t4_*, fig_t4_mirror |
| E-A2 t7 中段定位 | P1/P2 | ✅ | replay_t7_*, fig_t7_localization |
| E-A3 t10 诱饵 | P2 | ✅ | validation_t10 |
| E-A4 三分类 | P2/C2 | ✅ | 综合 |
| E-A5 分布错配 | P2 | ✅ | Day 2 记录 |
| E-R1 RSI 闭环 | P3/C1 | ✅ | rsi_loop.json, rsi-loop-report.md |
| E-R2 去泄露三臂 | P3/C1 | ✅ | rsi_noleak2.json |
| E-R3 patch 种子×5 | P3/C1 | ✅ | rsi_patch_seeds.json, data/audit/ |
| E-A1c schema 消融 | P2 因果层 | ✅ | data/audit/t4_V1-V3(600 ep) |
| E-A3c 语料消融 | P2 因果层 | ✅ | data/audit/t10_NODECOY(200 ep) |
| E-A*c 重放扩样+forkstate | P1/P2 因果层 | ✅ | data/audit/replay_*(21 条), forkstate_*(2) |
| E-E1 agent 熵对齐 | C1/O4 | ✅ | entropy_agent_raw.json, entropy-agent-alignment.md |
| E-S1 radix A/B | P1 | ✅ | v5_radix 记录 |
| E-S2 成本记账 | P1 | ✅ | cost_table.json, cost-table.md |
| *待做(W1)* | | | judge baseline / 自动判定器 / 回归矩阵 |

## 8. 汇编级诚实边界

1 模型(Qwen3-8B)/ 1 环境族(Chinook+mock)/ 决策锁定实证 2 例(t4、t7);"普遍存在"未证(E2b 外部 benchmark 排 W2)。熵对齐两侧均为截断熵(top-10/20),与 80/20 的全词表口径对比前需给 bound。t7 过散 watch-item 未关。V1 是"无法拒绝"而非"证明成立"(T=1 功效低)。

**9/1 批后已关**:t10 诱饵因果(语料消融)、t4 schema-priming 因果(消融网格)、patch 单样本方差(5 种子)、重放单轨迹 artifact(21 条扩样 + forkstate 50 续写)、成本表命中率外推(99.2% 实测)。**新增边界**:schema 消融揭示 canonical 聚类器在 V2 型输入(COUNT(DISTINCT)/COUNT(DISTINCT) 变体)下错位——结局口径优先于簇标签;RSI patch 消除陷阱不产生能力(correct 反降),E-R1 的 100% 依赖机制级授权而非警告。盲审计 50 条已抽样待人工标注(分类器循环性的最后一环)。
