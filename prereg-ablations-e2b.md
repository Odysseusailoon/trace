# 预注册:E2b 外部 benchmark 泛化 + 9/1 消融批(2026-08-31 定稿,跑前冻结)

> 目的:把 O3 的教训制度化——以下假设、判据、排除规则在数据采集**之前**写死;跑完只报结果,不改尺子。改动只允许追加注明日期的 amendment。

## A. 9/1 消融批(今天要跑的)

### A1. t4 schema 消融(假设:schema 列名措辞因果地提升 intdiv 率)
- 变体(工具描述文本,DB 不动):V0 control(现行);V1 NOSCHEMA(只列表名,无列名);V2 NOALBUMCOLS(仅 Album 行去列名);V3 REORDER(Album 列序 (AlbumId, Title, ArtistId) → (ArtistId, AlbumId, Title))。
- n=200/变体,seed0=10000(与 canonical 采集配对),T=1.0,其余不动。
- **主判据(冻结)**:V1、V2 的 intdiv 首步占比显著低于 V0 的 45.5%(两比例 z,α=0.05,双侧,对 3 个比较 Bonferroni)。V3 为探索性(方向不预设)。
- 预测:V1/V2 intdiv 大降(priming 成立);若不降,schema-priming 降级为"与 schema 呈现相关"。
- 副指标(探索性):整体正确率、错误位移(其他错误族占比)。

### A2. t10 语料消融(假设:诱饵数字因果地锁定 67.3 答案)
- 变体:V0 control;V1 NODECOY——把诱饵 snippet 改为 "Subscription streaming revenue reached $19.3 billion in 2023."(删除 "67.3% of total revenue" 短语;**保留 19.3 原料,28.6 在另一 snippet 不动**——任务仍可解)。
- n=200,seed0=10000。
- **主判据(冻结)**:V1 的 67.3-答案率显著低于 V0 的 93.5%(z,α=0.05)。
- 预测:67.3 率坍缩(<20%);正确率(67.5)上升为副指标。

### A3. 镜像重放扩样(假设:o_d 跳变是簇级系统现象,非单轨迹 artifact)
- t4:intdiv 结局 ×5、avg_groupby-correct ×5、sql:other ×3;t7:wrong_unit ×5、correct ×3(受 correct 池限制,7.5% × 200 = ~15 条可选)。K=50,seed 派生同现行。
- **主判据(冻结)**:E2c 判定器(α=0.05,Bonferroni,persistence<0.9)在 ≥80% 的 intdiv/avg 轨迹上于 d=0 检出分叉;t7 wrong_unit 轨迹 0 检出、correct 轨迹在 d=1 检出 ≥60%(correct 轨迹里走 no-calculator 路线的除外,排除规则:persistence_1=1.0 者不计入分母)。

### A4. RSI patch 种子(假设:27%→100% 与"便签有害"对 patch 生成随机性稳健)
- 泄露臂、便签臂各 3 个独立 patch(诊断+生成温度 1.0,seed 变化),每 patch 眼 200 ep。
- **主判据(冻结)**:泄露臂 3/3 patch ≥90% correct;便签臂效应方向(t4 correct 下降)3 份 patch 中 ≥2 份复现。检验按 patch 分层报告,不合并。

## B. E2b 外部 benchmark(W2 跑,现在冻结设计)

- 数据:BIRD-mini 挑 10 题(挑选标准:涉及聚合/单位/schema 歧义;**在看模型输出之前**由题面选定并存档)+ GSM8K 随机 10 题(calculator 工具)。
- 每题 200 ep 收集 → 失败聚类;仅对失败率 ∈ [20%, 90%] 的题投入重放(排除规则:太易/太难无信息)。
- **主判据(冻结)**:(1) 分叉检出率:重放的题中 E2c 判定器检出 ≥1 个分叉的比例;(2) coverage:检出分叉处正确簇占比的中位数。报数,不设通过线——但 coverage 中位数 <2% 时,论文 claim 须缩为"存在性"。
- 三分类占比为探索性。

## C. 全局排除与报告规则

- Episode error(网络/超时)不计入分母,计数单独报;error 率 >5% 的批整批重跑。
- 所有比例配 Wilson 95% CI;多重比较一律 Bonferroni;patch 级效应报分层。
- 本文档 git 提交时间即预注册时间戳。

---

## Amendment A1 (2026-09-01, appended before BIRD data collection)

### 范围
把 §B 的 BIRD-mini 线正式开跑,并追加两个探索性测量。GSM8K 线不动,另行安排。

### 选题(冻结)
- 确定性规则:`forkscope/scripts/select_bird_e2b.py`(聚合/比例/单位主题 + gold 为单标量的
  单层聚合 SQL + 每库 ≤2 题 + 难度下限 3 simple/3 moderate + question_id 升序取前 10)。
- 存档:`forkscope/data/bird/selection_e2b.json` — 10 题 + 10 条冻结替补队列。
- 替补规则(冻结):gold SQL 在库上执行结果非单行单列数值 → 按替补队列顺序替换,记录于日志。
- 选题只基于题面与 gold SQL 文本,未接触任何模型输出。

### Episode 协议(相对 Chinook 线的差异,冻结)
- 工具:仅 sql_query(schema 描述由 sqlite_master 自动生成,同 "Table(Col,...)" 风格,
  无陷阱表注入)+ calculator。web_search 移除(其语料是 Chinook 专用诱饵)。
- Prompt = BIRD question + evidence(官方 external-knowledge 协议)+ 3 位小数指示语。
- 判对(冻结):answer 抽取同 final_number;correct iff |v-g|/max(|g|,1e-9) < 0.01;
  若题面含 percentage/ratio,额外接受 v/100 或 v*100 同容差。其余为具名/other 失败。
- n=200/题/臂,seed0=10000,T=1.0,并发 24,错误 episode 不计分母单独报(同 §C)。

### 两个臂
- **NT 臂(confirmatory,对应 §B 判据)**:non-thinking,协议同上。
- **TH 臂(exploratory,新)**:enable_thinking=True,max_tokens=6000;多轮历史中
  剥离 <think> 块后回填(Qwen3 官方多轮协议),原始 thinking 文本保留在 steps 供分析。
  预测(方向性,不设通过线):TH 臂失败率低于 NT 臂;若某题 NT 失败率 ∈ [20%,90%] 而
  TH <5%,记为 "thinking 解毒" 证据。

### 全词表熵(exploratory,新)
- 对两臂全部 episode 的全部生成 token,HF teacher-forced forward 计算精确全词表熵,
  同时记录 top-10 截断熵(与 E-T4/E-E1 口径搭桥)。
- 已知边界:teacher-forcing 用本地 chat template 渲染,与 server 渲染可能有微小差异
  (2026-09-01 zoom 实验已记录此 caveat),渲染一致性检查随批报告。

### 判据
§B 主判据不变(分叉检出率 + coverage,基于 NT 臂)。TH 臂与熵均为探索性,只报数。

## Amendment A2 (2026-09-01, appended before any BIRD episode collection)

### 变更:选题改为两段式(题面段 + 不确定性段)
动机:纯题面选题不约束结局熵,可能选中 o_0 退化(≈0 或 ≈1)的题,fork 质量为零,
200 ep 无信息。参照 Zur et al. (arXiv:2511.04527) 的筛选协议。

1. **题面段**(A1 已冻结,不变):确定性规则(select_bird_e2b.py)产出 eligible 队列,
   question_id 升序。全量 eligible 列表随本 amendment 存档(selection_e2b.json 扩充)。
2. **不确定性段**(新,冻结):
   - 对 eligible 队列中 gold 可执行(单行单列数值)的前 60 题,各跑 10 条 NT episode
     (seed0=90000,协议同 A1,仅 n=10)。
   - 统计量:**众数答案出现次数**(答案抽取同 final_number,按 gold 同精度舍入分箱;
     no_answer 计为独立类)。用众数频率而非正确率:双错误答案分裂的题同样有 fork 质量
     (anti-mode 结构是研究对象,不是排除对象)。
   - 保留众数 ∈ [4,6]/10 的题,按 question_id 序取前 10 为正式集;不足 10 则顺延
     下一批 60 题重复。screen 全部结果(60 题 × 10 ep 的众数分布)落盘存档。
3. A1 中原 10 题选集保留存档并在论文附录报告(题面段单独选题的对照);主分析集
   以两段式结果为准。A1 其余协议(工具/判对/臂/n/熵)不变。
4. EG-1 / DE-1 / SR-1(experiment-design-2026-09-01.md)的 BIRD 采样框一律继承
   两段式正式集。
5. screen 预算 ~600 episodes(约一题全量的 3 倍),计入 E2b 预算。
