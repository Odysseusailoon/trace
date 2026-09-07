# forkscope

Experiment code for the [demo website](../demo/en.html).
Run the commands below from this directory.

## Setup

```bash
python -m pip install -e '.[dev]'
export FORKSCOPE_DB="$PWD/data/agentenv/chinook.sqlite"
```

Collection and replay require a running SGLang server with deterministic
inference and the Qwen tool-call parser. GPU serving, tokenization, and dataset
loading also require SGLang, transformers, and datasets in the experiment environment.
The statistical tests and saved results can be inspected without a model server.

## Chinook collection and replay

```bash
python -m agentenv.collect --tasks t4_artist_album_ratio t7_avg_track_len_min t10_search_plus_calc --n 200
python -m agentenv.analyze --tasks t4_artist_album_ratio
python -m agentenv.replay --task t4_artist_album_ratio --pick sql:intdiv --outcome intdiv_1.000 --k 50
python scripts/agent_stats.py --tasks t4_artist_album_ratio
```

## Other experiments

| Experiment | Entry points |
|---|---|
| BIRD | `scripts/select_bird_e2b.py`, `scripts/collect_bird.py`, `scripts/entropy_bird_fullvocab.py` |
| Token resampling and extraction audit | `scripts/run_pipeline.py`, `scripts/branch_stats.py`, `scripts/probe_len.py` |
| Token entropy and t4 zoom | `scripts/entropy_agent_analyze.py`, `scripts/zoom_t4_token.py` |
| Entropy-gate recall on Forking Fast data | `scripts/eg0_cot_recall.py` (run from the repository root) |
| Tool-description patches | `scripts/rsi_loop.py`, `scripts/rsi_noleak.py`, `scripts/rsi_patch_seeds.py` |
| Replay rule, confidence intervals, and ablations | `scripts/fork_rule.py`, `scripts/ci_report.py`, `scripts/audit_batch.py` |
| MoE routing follow-up described in the writeup | `scripts/moe_routing.py` |

Scripts document their inputs and arguments. [The data inventory](data/README.md)
distinguishes saved results from inputs that must be collected or downloaded.

## Local checks

```bash
python -m pytest tests -q
```

Figure scripts run from `xfigs/` and save PNG/SVG files under `xfigs/output/`.
The original MMLU reports remain marked invalidated; use `configs/dense.yaml`
for the corrected sampling configuration.
