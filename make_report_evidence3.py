# -*- coding: utf-8 -*-
"""
Evidence generation helper for Technical Agent v3.

This script scans the outputs folder and core source files to generate
structured evidence files that can be referenced directly in the coursework report.

Expected usage:
1) Run run_demo3.py first
2) Then run: python make_report_evidence3.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
AGENT_FILE = ROOT_DIR / "src" / "agent.py"
RUN_DEMO_FILE = ROOT_DIR / "run_demo3.py"


def read_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def write_text(file_path: Path, content: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")


def list_output_files(output_dir: Path) -> str:
    if not output_dir.exists():
        return "outputs folder not found. Please run run_demo3.py first."

    files = sorted([f for f in output_dir.rglob("*") if f.is_file()])

    lines = ["Generated output artifacts:"]
    for f in files:
        relative_path = f.relative_to(ROOT_DIR)
        lines.append(f"- {relative_path.as_posix()} ({f.stat().st_size} bytes)")

    return "\n".join(lines)


def extract_train_end_default(run_demo_text: str) -> str | None:
    match = re.search(r'--train_end".*default\s*=\s*"([^"]+)"', run_demo_text)
    return match.group(1) if match else None


def load_metrics(metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def format_metrics_table(metrics: dict) -> str:
    rows = []

    for split in ["train", "test", "full"]:
        if split not in metrics:
            continue

        m = metrics[split]
        rows.append(
            (
                split,
                m.get("CAGR"),
                m.get("Sharpe"),
                m.get("MaxDrawdown"),
                m.get("TotalReturn"),
                m.get("BuyHoldCAGR"),
                m.get("BuyHoldSharpe"),
                m.get("BuyHoldMaxDrawdown"),
                m.get("BuyHoldTotalReturn"),
            )
        )

    def pct(value):
        return "Not available" if value is None else f"{value * 100:.2f}%"

    def num(value):
        return "Not available" if value is None else f"{value:.3f}"

    table_lines = []
    table_lines.append("| Split | Strategy CAGR | Strategy Sharpe | Strategy MaxDD | Strategy Return | Buy&Hold CAGR | Buy&Hold Sharpe | Buy&Hold MaxDD | Buy&Hold Return |")
    table_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        split, cagr, sharpe, mdd, ret, bh_cagr, bh_sharpe, bh_mdd, bh_ret = r
        table_lines.append(
            f"| {split} | {pct(cagr)} | {num(sharpe)} | {pct(mdd)} | {pct(ret)} | "
            f"{pct(bh_cagr)} | {num(bh_sharpe)} | {pct(bh_mdd)} | {pct(bh_ret)} |"
        )

    return "\n".join(table_lines)


def extract_relevant_code_snippets(agent_code: str) -> str:
    keywords = [
        "RSI", "rsi", "rolling", "ewm",
        "entry", "exit", "signal", "position",
        "transaction", "cost", "return",
        "generate_signals", "backtest",
        "grid_search_rsi_params",
        "compute_split_metrics"
    ]

    lines = agent_code.splitlines()
    selected_indices = set()

    for i, line in enumerate(lines):
        for kw in keywords:
            if kw in line:
                for j in range(max(0, i - 6), min(len(lines), i + 8)):
                    selected_indices.add(j)

    selected_lines = [lines[i] for i in sorted(selected_indices)]

    if not selected_lines:
        return "No relevant code snippets found in src/agent.py."

    return "\n".join(selected_lines)


def generate_workflow_diagram() -> str:
    return """
PIPELINE WORKFLOW (ASCII DIAGRAM)

[Yahoo Finance OHLCV Data]
            |
            v
[Indicator Calculation]
 RSI (Wilder), Moving Averages, MACD
            |
            v
[RSI Parameter Grid Search (Train Only)]
            |
            v
[Signal Generation]
 Entry: RSI cross-up + MA bullish + MACD bullish
 Exit : RSI cross-down
            |
            v
[Execution Engine]
 Signal at close(t) -> Execute at open(t+1)
            |
            v
[Backtest Engine]
 Returns, transaction costs, equity curve
            |
            v
[Performance Evaluation]
 CAGR, Sharpe, Max Drawdown, Total Return
            |
            v
[Output Artifacts]
 CSV / JSON / PNG / LLM Report
""".strip()


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_demo_text = read_text(RUN_DEMO_FILE) if RUN_DEMO_FILE.exists() else ""
    agent_text = read_text(AGENT_FILE) if AGENT_FILE.exists() else ""

    train_end_default = extract_train_end_default(run_demo_text) or "Not detected"
    metrics_data = load_metrics(OUTPUT_DIR / "metrics.json")

    # 1) Output artifact list
    write_text(OUTPUT_DIR / "project_artifacts.txt", list_output_files(OUTPUT_DIR))

    # 2) Workflow diagram
    write_text(OUTPUT_DIR / "workflow_ascii.txt", generate_workflow_diagram())

    # 3) Code evidence snippets
    write_text(OUTPUT_DIR / "agent_code_snippets.txt", extract_relevant_code_snippets(agent_text))

    # 4) Metrics table
    if metrics_data:
        write_text(OUTPUT_DIR / "metrics_table.md", format_metrics_table(metrics_data))
    else:
        write_text(OUTPUT_DIR / "metrics_table.md", "metrics.json not found. Please run run_demo3.py first.")

    # 5) Summary file for coursework reference
    summary_lines = []
    summary_lines.append("# Evidence Summary")
    summary_lines.append("")
    summary_lines.append(f"- Project root directory: {ROOT_DIR.as_posix()}")
    summary_lines.append(f"- Default train_end parameter: {train_end_default}")
    summary_lines.append("- Execution assumption: signals generated at close(t), executed at open(t+1).")
    summary_lines.append("- Returns computed using open-to-open price changes.")
    summary_lines.append("- Transaction costs applied when position changes.")
    summary_lines.append("- Dividends, slippage and market impact are not modeled.")
    summary_lines.append("")
    summary_lines.append("## Output Files")
    summary_lines.append("See: outputs/project_artifacts.txt")
    summary_lines.append("")
    summary_lines.append("## Performance Table")
    summary_lines.append("See: outputs/metrics_table.md")
    summary_lines.append("")
    summary_lines.append("## Workflow Diagram")
    summary_lines.append("See: outputs/workflow_ascii.txt")
    summary_lines.append("")
    summary_lines.append("## Implementation Evidence")
    summary_lines.append("See: outputs/agent_code_snippets.txt")

    write_text(OUTPUT_DIR / "evidence_summary.md", "\n".join(summary_lines))

    print("Evidence files generated successfully:")
    print("- outputs/evidence_summary.md")
    print("- outputs/metrics_table.md")
    print("- outputs/project_artifacts.txt")
    print("- outputs/workflow_ascii.txt")
    print("- outputs/agent_code_snippets.txt")


if __name__ == "__main__":
    main()
        
