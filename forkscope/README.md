# forkscope

**哪一步决定了 agent 的结果?** 在 SGLang 上对 agent 轨迹做分叉点分析(Forking Paths Analysis):在每个决策步边界重放 K 条完整续写(工具真执行),测出结局分布 o_d 在哪一步坍缩——用测量回答归因,而不是让 LLM 读日志去猜。

基于 Bigelow et al. *Forking Fast*(arXiv:2608.19611)并扩展到 agent 工具轨迹。

## 三个数字

- t4(text-to-SQL):第一条 SQL 把同一起点(22% 正确)锁到 **0% 或 100%**;46% 的 episode 被 schema 措辞诱导进整数除法陷阱,自愈 2/126
- RSI 闭环:agent 读分叉归因报告、改自己的工具描述、重测——正确率 **27%→100%**(200 vs 200)
- RadixAttention 实测 A/B:重采样负载的 prefill 计算量 **↓14.9×**(93.3% 前缀命中)

## 快速上手

```bash
# 1. 起 server(H100,~2min)
HF_HUB_CACHE=/scratch/hf python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B --port 30000 \
  --enable-metrics --enable-deterministic-inference

# 2. 收 episodes(每任务 200 条,~7min)
python -m agentenv.collect --tasks t4_artist_album_ratio --n 200

# 3. 首步锁定分析(表 + MI + 判定)
python -m agentenv.analyze --tasks t4_artist_album_ratio

# 4. 决策步重放(o_d 曲线,单轨迹因果)
python -m agentenv.replay --task t4_artist_album_ratio --pick sql:intdiv --outcome intdiv_1.000 --k 50

# 5. 统计验证(V1 多项噪声 / V2 1/√S)
python scripts/agent_stats.py --tasks t4_artist_album_ratio

# 6. RSI 闭环(报告→自补丁→重测)
python scripts/rsi_loop.py
```

MCQ token 级管线(复现 Forking Fast):`scripts/run_pipeline.py`、`scripts/run_stats.py`、`scripts/make_report.py`。

## 讲解材料

- `notebooks/demo.ipynb` — 3 分钟故事线(可 Restart & Run All)
- `../forkscope-report.html` — 完整发现报告(自包含,浏览器直接打开)
- `../RFC.md` — 设计文档 + Day 1-3 实现纪要
- `../paper-reports/` — 11 篇相关文献的深度报告
- `../objections-and-rebuttals.md` — 质疑与回应(FAQ/rebuttal 素材)

## Repo 地图

```
agentenv/    工具环境(Chinook SQLite + calculator + mock search,4 类陷阱)
             runner/collect/analyze/replay —— agent 侧四件套
src/forkscope/  MCQ token 级管线(base_path→fork_enum→resampler→smoothing→stats)
scripts/     入口脚本(含 rsi_loop.py、agent_stats.py、entropy_v1_followup.py)
data/reports/  所有图表、validation/replay/transcripts JSON——报告数字均可由此复算
```

## 已知边界

1 个 8B 模型、10 任务环境;决策锁定案例 2 个(t4 首步、t7 中段)+ 诱饵锁定 1 个(t10);RSI patch 有答案泄露嫌疑(去泄露版排期中);t7 轻度过散监控中。完整诚实清单见报告 §14。
