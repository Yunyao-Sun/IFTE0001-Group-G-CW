# -*- coding: utf-8 -*-
"""
Technical Agent Demo (v3): Yahoo Finance pipeline + train-time RSI parameter selection
Indicators: RSI + MA + MACD

Member-specific entry file: run_demo3.py (for version separation inside group).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from src.agent import (
    BacktestConfig,
    prepare_base_df,
    generate_signals,
    backtest,
    compute_split_metrics,
    plot_equity,
    grid_search_rsi_params,
    summarize_latest_signals,
)

try:
    from llm_report3 import try_run_llm_report
except Exception:
    try_run_llm_report = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Technical Agent v3 (RSI+MA+MACD) with Yahoo Finance (Academic Version)")
    p.add_argument("--ticker", type=str, default="BP", help="Asset display name")
    p.add_argument("--yahoo_symbol", type=str, default="BP.L", help="Yahoo Finance ticker")
    p.add_argument("--years", type=int, default=10, help="Rolling history window")
    p.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    p.add_argument("--train_end", type=str, default="2020-12-31", help="Train end date (YYYY-MM-DD)")
    p.add_argument("--no_param_search", action="store_true", help="Skip RSI param search; use cfg defaults")

    # Optional overrides
    p.add_argument("--rsi_entry", type=float, default=None)
    p.add_argument("--rsi_exit", type=float, default=None)

    return p.parse_args()


def fetch_yahoo_ohlcv(symbol: str, years: int) -> pd.DataFrame:
    end = datetime.utcnow().date()
    start = end - timedelta(days=int(years * 365.25))

    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo Finance returned empty data for symbol={symbol}")

    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def build_strategy_summary(cfg: BacktestConfig, selected: dict | None, latest_signals: dict | None) -> str:
    sizing = (
        "Position sizing (vol targeting): ENABLED (default)\n"
        f"- vol_window: {cfg.vol_window} days\n"
        f"- target_vol_annual: {cfg.target_vol_annual:.4f}\n"
        f"- max_leverage: {cfg.max_leverage:.4f}\n"
        f"- min_vol_annual: {cfg.min_vol_annual:.4f}\n"
    ) if cfg.use_position_sizing else "Position sizing (vol targeting): DISABLED (position=0/1)\n"

    sel_txt = ""
    if selected:
        sel_txt = (
            "\nParameter selection (train-only grid search):\n"
            f"- Objective (train): {selected.get('objective')}\n"
            f"- Train end: {selected.get('train_end')}\n"
            f"- Selected rsi_entry: {selected.get('selected_rsi_entry')}\n"
            f"- Selected rsi_exit : {selected.get('selected_rsi_exit')}\n"
            f"- Best train score   : {selected.get('best_train_score')}\n"
            f"- Train trades (best): {selected.get('train_trades_best')}\n"
        )

    latest_txt = ""
    if latest_signals:
        latest_txt = (
            "\nLatest indicator snapshot (for report):\n"
            f"- RSI: {latest_signals.get('RSI_value')} ({latest_signals.get('RSI_state')})\n"
            f"- MA trend: {latest_signals.get('MA_trend')} (fast={latest_signals.get('MA_fast')}, slow={latest_signals.get('MA_slow')})\n"
            f"- MACD trend: {latest_signals.get('MACD_trend')} (dif={latest_signals.get('MACD_dif')}, dea={latest_signals.get('MACD_dea')}, hist={latest_signals.get('MACD_hist')})\n"
        )

    return (
        "Indicators used: RSI (Wilder) + MA crossover + MACD\n"
        f"- RSI window: {cfg.rsi_window}\n"
        f"- MA fast/slow: {cfg.ma_fast}/{cfg.ma_slow}\n"
        f"- MACD fast/slow/signal: {cfg.macd_fast}/{cfg.macd_slow}/{cfg.macd_signal}\n"
        "\nSignal timing & execution:\n"
        "- Signals computed on day t close\n"
        "- Trades executed at day t+1 open (next-day open execution)\n"
        "\nSignal logic (long-only, fused):\n"
        f"- Entry: RSI crosses ABOVE {cfg.rsi_entry} AND MA_fast > MA_slow AND MACD DIF > DEA\n"
        f"- Exit : RSI crosses BELOW {cfg.rsi_exit}"
        + (" OR MA trend turns bearish OR MACD turns bearish\n" if cfg.exit_on_trend_break else "\n")
        + f"{sel_txt}"
        + f"{latest_txt}\n"
        + f"{sizing}"
    )


BACKTEST_ASSUMPTIONS = (
    "Backtest assumptions:\n"
    "- Execution: next-day open\n"
    "- PnL model: open-to-next-open\n"
    "- Long-only strategy\n"
    "- Transaction costs: applied on turnover (bps)\n"
)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Download data
    raw_df = fetch_yahoo_ohlcv(args.yahoo_symbol, args.years)
    raw_price_path = outdir / "raw_prices.csv"
    raw_df.to_csv(raw_price_path, index=False)

    # 2) Prepare cfg
    cfg = BacktestConfig(train_end=args.train_end)

    if args.rsi_entry is not None:
        cfg.rsi_entry = float(args.rsi_entry)
    if args.rsi_exit is not None:
        cfg.rsi_exit = float(args.rsi_exit)

    # 3) Prepare base df (prices + returns + indicators)
    base_csv = outdir / "prices_for_agent.csv"
    raw_df.to_csv(base_csv, index=False)
    base_df = prepare_base_df(base_csv, cfg)

    # 4) Optional param search (train-only)
    selected_info = None
    if not args.no_param_search:
        entry_grid = list(range(30, 56, 5))   # 30..55
        exit_grid = list(range(55, 86, 5))    # 55..85
        objective = "Sharpe"

        best_cfg, table = grid_search_rsi_params(
            base_df, cfg, entry_grid=entry_grid, exit_grid=exit_grid, objective=objective, min_trades_train=3
        )
        cfg = best_cfg

        (outdir / "param_search_results.csv").write_text(table.to_csv(index=False), encoding="utf-8")
        selected_info = {
            "objective": objective,
            "train_end": cfg.train_end,
            "selected_rsi_entry": cfg.rsi_entry,
            "selected_rsi_exit": cfg.rsi_exit,
            "best_train_score": float(table.iloc[0]["objective_score"]) if len(table) else None,
            "train_trades_best": float(table.iloc[0]["train_Trades"]) if len(table) else None,
        }
        (outdir / "selected_params.json").write_text(json.dumps(selected_info, indent=2, ensure_ascii=False), encoding="utf-8")

    # 5) Run final backtest
    df = generate_signals(base_df, cfg)
    df = backtest(df, cfg)
    split_metrics = compute_split_metrics(df, cfg)

    (outdir / "metrics.json").write_text(json.dumps(split_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(outdir / "backtest_full.csv", index=False)

    # 6) Plot
    plot_equity(df, outdir, ticker=args.ticker)

    # 7) Build strategy summary + latest signal snapshot (for LLM prompt)
    latest_signals = summarize_latest_signals(df, cfg)
    strategy_summary = build_strategy_summary(cfg, selected_info, latest_signals)
    (outdir / "strategy_config.txt").write_text(strategy_summary, encoding="utf-8")
    (outdir / "backtest_assumptions.txt").write_text(BACKTEST_ASSUMPTIONS, encoding="utf-8")

    # 8) Optional LLM report
    if try_run_llm_report is not None:
        llm_prompt = (
            f"Asset: {args.ticker}\n"
            f"Yahoo symbol: {args.yahoo_symbol}\n\n"
            f"{strategy_summary}\n\n"
            f"{BACKTEST_ASSUMPTIONS}\n\n"
            f"Metrics (train/test/full):\n{json.dumps(split_metrics, indent=2, ensure_ascii=False)}\n\n"
            "Task: Write a concise professional technical analysis report.\n"
            "- Summarize trend, momentum, and RSI condition\n"
            "- Give a recommendation (Buy/Hold/Sell) with risk considerations\n"
            "- Reference the provided indicator snapshot and metrics\n"
        )
        report, err = try_run_llm_report(llm_prompt)
        if report:
            (outdir / "llm_report.md").write_text(report, encoding="utf-8")
        elif err:
            (outdir / "llm_report_error.txt").write_text(str(err), encoding="utf-8")

    print("\n===== PIPELINE FINISHED (Technical Agent v3) =====")
    print("Raw Yahoo data:", raw_price_path)
    if not args.no_param_search:
        print("Param search table:", outdir / "param_search_results.csv")
        print("Selected params:", outdir / "selected_params.json")
    print("Backtest data:", outdir / "backtest_full.csv")
    print("Metrics:", outdir / "metrics.json")
    if try_run_llm_report is not None:
        print("LLM report:", outdir / "llm_report.md")


if __name__ == "__main__":
    main()

