#!/usr/bin/env bash
# Launch SGLang server for forkscope. Model/port from env or defaults.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-30000}"
MEM_FRAC="${MEM_FRAC:-0.85}"
VENV="${VENV:-/home/dev/venv}"

export HF_HUB_CACHE="${HF_HUB_CACHE:-/scratch/hf}"

exec "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  --port "$PORT" \
  --mem-fraction-static "$MEM_FRAC" \
  --enable-metrics \
  --enable-deterministic-inference \
  --log-level info
