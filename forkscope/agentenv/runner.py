"""Agent loop runner: SGLang chat completions + tool calls.

v2 (Day 3): the Day-2 version appended each assistant tool-call message TWICE
(once as the raw API message, once per parsed call), so every tool call showed
up twice in the conditioning context. Fixed here — each assistant turn is
appended exactly once, raw <tool_call> fallbacks are normalized to structured
calls so replay conditioning is consistent.

Also adds continue_episode(): run the loop from an arbitrary message prefix —
the primitive for decision-step replay FPA.
"""
from __future__ import annotations

import json
import re

import httpx

from agentenv.tools import TOOLS, call_tool

MAX_ROUNDS = 8

SYSTEM = (
    "You are a data analyst with access to a music-store SQLite database, "
    "a calculator, and web search. Solve the user's task using tools. "
    "When you have the final answer, reply with 'The answer is <answer>.'")


def initial_messages(task: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task["prompt"]},
    ]


def _fallback_tool_calls(content: str | None) -> list[dict]:
    """Model wrote a raw <tool_call> JSON in content (tool parser missed it)."""
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content or "", re.S)
    if not m:
        return []
    try:
        fn = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []
    args = fn.get("arguments", {})
    if isinstance(args, dict):
        args = json.dumps(args)
    return [{"id": "call_fb", "type": "function",
             "function": {"name": fn.get("name"), "arguments": args}}]


async def continue_episode(base_url: str, messages: list[dict], seed: int,
                           start_round: int = 0, max_tokens: int = 2000,
                           tools: list[dict] | None = None,
                           enable_thinking: bool = False) -> dict:
    """Run the agent loop starting from an existing message prefix."""
    messages = [dict(m) for m in messages]
    tools = tools if tools is not None else TOOLS
    steps: list[dict] = []
    async with httpx.AsyncClient(base_url=base_url, timeout=600) as c:
        for rnd in range(start_round, MAX_ROUNDS):
            r = await c.post("/v1/chat/completions", json={
                "model": "default",
                "messages": messages,
                "tools": tools,
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": max_tokens,
                "seed": seed,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            })
            r.raise_for_status()
            j = r.json()
            msg = j["choices"][0]["message"]
            fin = j["choices"][0].get("finish_reason")

            tcs = msg.get("tool_calls") or []
            fallback = False
            raw_content = msg.get("content")
            # Qwen3 multi-turn protocol: prior turns' <think> blocks are not part
            # of the conditioning context. Keep the raw text in steps for
            # entropy/fork analysis; strip it from the message history.
            content = raw_content
            if enable_thinking and content:
                content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.S)
            if not tcs:
                tcs = _fallback_tool_calls(content)
                fallback = bool(tcs)

            assistant_msg: dict = {"role": "assistant"}
            if content and not fallback:
                assistant_msg["content"] = content
            if tcs:
                assistant_msg["tool_calls"] = tcs
            messages.append(assistant_msg)
            steps.append({"round": rnd, "role": "assistant",
                          "content": raw_content, "finish": fin,
                          "tool_calls": tcs or None,
                          "fallback_parse": fallback})

            if not tcs:
                break
            for tc in tcs:
                args = tc["function"].get("arguments", "{}")
                if isinstance(args, dict):
                    args = json.dumps(args)
                out = call_tool(tc["function"]["name"], args)
                steps.append({"round": rnd, "role": "tool",
                              "name": tc["function"]["name"],
                              "args": args, "result": out})
                messages.append({"role": "tool",
                                 "tool_call_id": tc.get("id", "call_0"),
                                 "content": out})

    final = next((s["content"] for s in reversed(steps)
                  if s["role"] == "assistant" and s.get("content")), None)
    return {"seed": seed, "steps": steps, "messages": messages, "final": final}


async def run_episode(base_url: str, task: dict, seed: int,
                      max_tokens: int = 2000, tools: list[dict] | None = None,
                      enable_thinking: bool = False) -> dict:
    ep = await continue_episode(base_url, initial_messages(task), seed,
                                max_tokens=max_tokens, tools=tools,
                                enable_thinking=enable_thinking)
    ep["task_id"] = task["id"]
    return ep


def outcome_of(ep: dict, gold: dict) -> str:
    """Rough outcome label for probe filtering (kept from Day 2)."""
    text = (ep.get("final") or "").lower()
    for val in [str(v) for v in gold.values() if isinstance(v, (int, float))]:
        if val and val in text:
            return "gold"
    for v in gold.values():
        if isinstance(v, str) and len(v) > 2 and v.lower() in text:
            return "gold_partial"
    return "wrong"
