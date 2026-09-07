# trace

Code and recorded results for the Forkscope demo: resampling LLM continuations
at token positions and tool-call boundaries to measure outcome distributions.

## Website

- [Reproduction writeup](demo/repro.html)
- [English demo](demo/en.html) / [Chinese demo](demo/index.html)
- [Literature](demo/lit.html)
- [Interpretability proposal](demo/moonshot.html)

Preview from the repository root:

```bash
python3 demo/serve_nocache.py
```

Open http://127.0.0.1:8901/en.html. Deployment instructions are in
[demo/README.md](demo/README.md).

## Experiments and data

| Experiment | Implementation | Recorded results |
|---|---|---|
| Chinook t4/t7/t10: collection, replay, schema and corpus ablations | [agentenv](forkscope/agentenv/), [audit batch](forkscope/scripts/audit_batch.py) | [8B](forkscope/data/reports/), [35B](forkscope/data/reports36/), [audit](forkscope/data/audit/) |
| BIRD: selection and thinking/non-thinking comparison | [selection](forkscope/scripts/select_bird_e2b.py), [collection](forkscope/scripts/collect_bird.py) | [selection](forkscope/data/bird/), [analysis](forkscope/data/bird_analysis_20260902.json) |
| Token resampling, entropy gates, and sampling statistics | [pipeline](forkscope/src/forkscope/), [scripts](forkscope/scripts/) | [reports](forkscope/data/reports/), [tests](forkscope/tests/) |
| Tool-description patch and SAE probe | [patch loop](forkscope/scripts/rsi_loop.py), [controls](forkscope/scripts/rsi_noleak.py) | [patch](forkscope/data/reports/rsi_loop.json), [SAE notes and figures](forkscope/data/sae/) |

[Run instructions](forkscope/README.md) · [Data inventory](forkscope/data/README.md) ·
[Sampling preregistration and amendments](prereg-ablations-e2b.md) ·
[Figure sources](forkscope/xfigs/)

The original MMLU fork counts were invalidated by the sampling and extraction
audit. Their reports retain that notice; the corrected configuration is
[dense.yaml](forkscope/configs/dense.yaml).
