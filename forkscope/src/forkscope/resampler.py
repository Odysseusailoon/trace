"""Stage 3: resampling engine.

Submits branch tasks to SGLang sorted by t ascending (maximizes radix cache
hits), S samples per branch via server-side n=S, seeds bound to (case, t, w)
for bit-level reproducibility, writes branch records incrementally for resume.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

from .base_path import BasePath
from .client import SGLangClient
from .fork_enum import Branch


def task_seed(case_id: str, t: int, tok_id: int) -> int:
    h = hashlib.sha256(f"{case_id}:{t}:{tok_id}".encode()).digest()
    return int.from_bytes(h[:4], "little")


class Resampler:
    def __init__(self, client: SGLangClient, out_dir: str | Path):
        self.client = client
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _record_path(self, case_id: str) -> Path:
        return self.out_dir / f"branches_{case_id}.jsonl"

    def _done_keys(self, case_id: str) -> set[tuple[int, int]]:
        p = self._record_path(case_id)
        done: set[tuple[int, int]] = set()
        if p.exists():
            with open(p) as f:
                for line in f:
                    rec = json.loads(line)
                    done.add((rec["t"], rec["tok_id"]))
        return done

    async def run_branch(
        self,
        base: BasePath,
        branch: Branch,
        n_samples: int,
        max_new_tokens: int,
        stop: list[str] | None = None,
    ) -> dict:
        prefix = branch.prefix_ids(base)
        n = n_samples if branch.t > 0 else max(n_samples, 1)  # caller handles N0
        seed = task_seed(branch.case_id, branch.t, branch.tok_id)
        resp = await self.client.sample_continuations(prefix, n=n, max_new_tokens=max_new_tokens,
                                                      seed=seed, stop=stop)
        items = resp if isinstance(resp, list) else [resp]
        answers = []
        cont_lens = []
        for it in items:
            meta = it.get("meta_info", {})
            ids = [lp[1] for lp in (meta.get("output_token_logprobs") or [])]
            text = it.get("text", "")
            answers.append(text)
            cont_lens.append(meta.get("completion_tokens", len(ids) if ids else None))
        return {
            "case_id": branch.case_id,
            "t": branch.t,
            "tok_id": branch.tok_id,
            "tok_p": branch.tok_p,
            "is_base": branch.is_base,
            "n": n,
            "seed": seed,
            "continuations": answers,
            "cont_lens": cont_lens,
        }

    async def run(
        self,
        base: BasePath,
        branches: list[Branch],
        samples_per_branch: int,
        max_new_tokens: int,
        samples_t0: int | None = None,
        stop: list[str] | None = None,
        progress_every: int = 50,
    ) -> Path:
        """Run all pending branches for a case, sorted by t ascending. Resumable."""
        done = self._done_keys(base.case_id)
        todo = [b for b in sorted(branches, key=lambda b: b.t) if (b.t, b.tok_id) not in done]
        path = self._record_path(base.case_id)
        t_start = __import__("time").time()

        lock = asyncio.Lock()
        counter = {"n": 0}

        async def worker(b: Branch):
            n = samples_t0 if (b.t == 0 and samples_t0) else samples_per_branch
            rec = await self.run_branch(base, b, n, max_new_tokens, stop=stop)
            async with lock:
                with open(path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                counter["n"] += 1
                if counter["n"] % progress_every == 0 or counter["n"] == len(todo):
                    dt = __import__("time").time() - t_start
                    rate = counter["n"] / dt if dt else 0
                    eta = (len(todo) - counter["n"]) / rate if rate else float("inf")
                    print(f"[resample] {counter['n']}/{len(todo)} branches "
                          f"({rate:.1f}/s, eta {eta/60:.1f} min)", flush=True)

        # bounded fan-out: client semaphore already caps HTTP concurrency,
        # but keep task count sane for very large branch tables
        chunk = 256
        for i in range(0, len(todo), chunk):
            await asyncio.gather(*(worker(b) for b in todo[i : i + chunk]))
        return path
