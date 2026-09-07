# Recorded experiment data

| Website experiment | Files in this checkout |
|---|---|
| Chinook t4/t7/t10, Qwen3-8B | `agentenv/chinook.sqlite`; `reports/validation_*.json` and `.md`; selected episodes in `reports/transcripts.json` |
| Chinook t4/t7/t10, Qwen3.6-35B-A3B | `raw36/episodes_*.jsonl`; `reports36/validation_*.json` and `.md` |
| Decision-step replay | `reports/replay_*.json`; expanded replays and continuation dumps in `audit/` |
| Schema and corpus ablations | `audit/t4_V1_NOSCHEMA/`, `audit/t4_V2_NOALBUMCOLS/`, `audit/t4_V3_REORDER/`, `audit/t10_NODECOY/` |
| Token entropy and t4 zoom | `reports/entropy_agent_raw.json`, `reports/entropy_v1_followup.json`, `reports/zoom_t4_*.json` |
| BIRD selection and four model/thinking conditions | `bird/mini_dev_sqlite.json`, `bird/selection_e2b.json`, `bird_analysis_20260902.json` |
| Entropy-gate recall | `reports/eg0_cot_recall.json` |
| Tool-description patch | `reports/rsi_loop.json`, `reports/rsi_patch_seeds.json` |
| Sampling statistics and cost accounting | `reports/agent_v1v2.json`, `reports/cost_table.json` |
| SAE probe described in the English demo | [Measurement notes and figures](sae/README.md) |
| Original MMLU run | `aggregated/`, `reports/report_*`, and the corresponding figures |

The original MMLU reports are marked invalidated. The historical measurements
are retained to document the extraction and sampling failures; their fork counts
are not validated results.

BIRD episode traces and its SQLite databases, the full 8B collection, the
no-leak patch run, and SAE extraction code/raw activations are not included in
this checkout. The SAE notes preserve the available measurements and identify
the original external output location. A summary is not a substitute for the
missing raw traces.

The entropy-gate script also reads the upstream Forking Fast checkout at
`../../forking-fast/`, including `data/s200/` and `otrecon/`. That checkout is
kept locally as an experiment dependency and is excluded from this repository.
Its source is [ericb-goodfire/forking-fast](https://github.com/ericb-goodfire/forking-fast).

The [preregistration](../../prereg-ablations-e2b.md) retains the original sampling
rules and amendments. Figures and their rendering scripts are in
[`../xfigs/`](../xfigs/).
