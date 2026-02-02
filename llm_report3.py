# -*- coding: utf-8 -*-
"""
LLM report generator (OpenAI Responses API).

Goals:
- Stable interface: run_llm_report(prompt) -> str
- Backwards compatibility: generate_report(...) alias
- Optional rich wrapper prompt for better structure
- Optional non-fatal call helper: try_run_llm_report(...)
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_URL = "https://api.openai.com/v1/responses"


def _extract_response_text(data: dict) -> str:
    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"].strip()

    texts = []
    for item in (data.get("output") or []):
        for c in (item.get("content") or []):
            if not isinstance(c, dict):
                continue

            if isinstance(c.get("text"), str) and c["text"].strip():
                texts.append(c["text"])
                continue

            if c.get("type") in ("output_text", "text") and isinstance(c.get("value"), str):
                v = c["value"].strip()
                if v:
                    texts.append(v)

    out = "\n".join(texts).strip()
    if out:
        return out

    raise RuntimeError("Could not extract text from LLM response")


def wrap_prompt_for_richer_report(raw_prompt: str) -> str:
    return f"""
You are a buy-side technical analyst writing an internal investment committee memo.

Output requirements:
- Use clear section headings (Markdown).
- Keep the memo factual and consistent with the provided strategy description and metrics.
- Do NOT invent prices, dates, catalysts, fundamentals, or any performance statistics beyond what is provided.
- If a detail is not provided, explicitly state "Not provided in the backtest output".

Include these sections:
1) Executive Summary (3-6 bullets)
2) Strategy Definition (rules, timing, execution, costs)
3) Performance Summary (Strategy vs Buy & Hold; highlight CAGR, Sharpe, Max Drawdown, Total Return)
4) Risk & Limitations (modeling, costs, dividends, slippage, regime risk, parameter risk)
5) Regime Interpretation (when it works/fails based on trend/momentum intuition; no new data)
6) Actionable Monitoring Checklist (what to watch daily/weekly)
7) Research Extensions (3-6 concrete next steps)

SOURCE MATERIAL (single source of truth):
{raw_prompt}
""".strip()


def run_llm_report(
    prompt: str,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.2,
    max_output_tokens: int = 1400,
    rich_wrap: bool = True,
    timeout: int = 60,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    final_prompt = wrap_prompt_for_richer_report(prompt) if rich_wrap else prompt

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": final_prompt,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    resp = requests.post(
        OPENAI_API_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    return _extract_response_text(resp.json())


# Backwards-compatible alias for older code
def generate_report(prompt: str, model="gpt-4.1-mini", temperature=0.2, max_tokens=1200) -> str:
    return run_llm_report(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        rich_wrap=True,
    )


def try_run_llm_report(*args, **kwargs) -> tuple[str | None, str | None]:
    """
    Non-fatal wrapper: returns (report_text_or_none, error_or_none)
    Useful when you want the backtest to finish even if LLM fails (quota, network, etc.)
    """
    try:
        return run_llm_report(*args, **kwargs), None
    except Exception as e:
        return None, repr(e)


def main() -> None:
    outdir = Path("outputs")
    prompt_path = outdir / "llm_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError("Run run_demo3.py first to generate outputs/llm_prompt.txt")

    prompt = prompt_path.read_text(encoding="utf-8")
    print(f"Loaded prompt ({len(prompt)} chars)")

    report, err = try_run_llm_report(prompt)
    if err:
        raise RuntimeError(f"LLM generation failed: {err}")

    report_path = outdir / "llm_report.md"
    report_path.write_text(report or "", encoding="utf-8")
    print("DONE ->", report_path)


if __name__ == "__main__":
    main()

