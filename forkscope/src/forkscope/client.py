"""Thin async client for the SGLang HTTP server.

Everything rides /generate with input_ids (never text) so prefixes are
exact and RadixAttention can share them. Secrets/retries kept minimal.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


class SGLangClient:
    def __init__(self, base_url: str, concurrency: int = 32, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self._sem = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, max=20))
    async def _post(self, path: str, payload: dict) -> Any:
        async with self._sem:
            r = await self._client.post(path, json=payload)
            r.raise_for_status()
            return r.json()

    async def get(self, path: str) -> Any:
        r = await self._client.get(path)
        r.raise_for_status()
        return r.json()

    async def generate(
        self,
        input_ids: list[int],
        sampling_params: dict,
        return_logprob: bool = False,
        top_logprobs_num: int | None = None,
    ) -> Any:
        payload: dict[str, Any] = {
            "input_ids": input_ids,
            "sampling_params": sampling_params,
        }
        if return_logprob:
            payload["return_logprob"] = True
            payload["logprob_start_len"] = 0
        if top_logprobs_num is not None:
            payload["top_logprobs_num"] = top_logprobs_num
        return await self._post("/generate", payload)

    async def greedy_with_topk(self, input_ids: list[int], max_new_tokens: int, top_k: int = 10) -> Any:
        return await self.generate(
            input_ids,
            {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "top_k": 1,
                "top_p": 1.0,
            },
            return_logprob=True,
            top_logprobs_num=top_k,
        )

    async def sample_continuations(
        self,
        prefix_ids: list[int],
        n: int,
        max_new_tokens: int,
        seed: int | None = None,
        stop: list[str] | None = None,
    ) -> Any:
        """n continuations, each with its own seed (seed, seed+1, ...).

        SGLang's n>1 expands one sampling_params dict, so a single
        sampling_seed would make all n draws identical. Per-copy seeds give
        distinct, reproducible draws; the prefix prefill is still shared via
        the radix cache (verified in V5).
        """
        async def one(i: int):
            params: dict[str, Any] = {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": max_new_tokens,
            }
            if seed is not None:
                params["sampling_seed"] = seed + i
            if stop:
                params["stop"] = stop
            resp = await self.generate(prefix_ids, params)
            return resp[0] if isinstance(resp, list) else resp

        return await asyncio.gather(*(one(i) for i in range(n)))

    async def metrics_text(self) -> str:
        r = await self._client.get("/metrics")
        r.raise_for_status()
        return r.text

    async def health(self) -> bool:
        try:
            await self.get("/get_model_info")
            return True
        except Exception:
            return False
