# -*- coding: utf-8 -*-
"""
Technical Agent (v3): RSI + MA + MACD (with train-time RSI parameter selection)

What v3 adds vs RSI-only v2:
- Indicators: RSI(14, Wilder) + MA crossover (fast/slow) + MACD(12,26,9)
- Signal fusion (long-only):
  Entry requires RSI cross-up + trend confirmation (MA fast > MA slow) + momentum confirmation (MACD DIF > DEA)
  Exit on RSI cross-down, and (optional) on trend/momentum break
- A helper to summarize the latest indicator states for LLM reporting

Everything else is kept compatible with the v2 pipeline:
- Next-day open execution
- Open-to-next-open returns
- Transaction cost on turnover
- Position sizing: volatility targeting (ENABLED by default)
- Metrics: CAGR, Sharpe, Max Drawdown, Hit Rate, Total Return
- Grid-search RSI entry/exit on TRAIN split, then evaluate on TEST + FULL
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRADING_DAYS = 252


# =========================
# Config
# =========================

@dataclass
class BacktestConfig:
    # costs & metrics
    tx_cost_bps: float = 10.0
    risk_free_rate: float = 0.0

    # split
    train_end: str = "2020-12-31"

    # RSI
    rsi_window: int = 14
    rsi_entry: float = 40.0
    rsi_exit: float = 70.0

    # MA (trend filter)
    ma_fast: int = 20
    ma_slow: int = 50

    # MACD (momentum confirmation)
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Signal fusion controls
    # If True: exit also when MA trend turns bearish or MACD turns bearish
    exit_on_trend_break: bool = True

    # Position sizing (Vol targeting) - ENABLED by default
    use_position_sizing: bool = True
    vol_window: int = 20
    target_vol_annual: float = 0.15
    max_leverage: float = 1.0
    min_vol_annual: float = 0.01


# =========================
# Data Loader
# =========================

def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    col_map = {}
    for c in df.columns:
        lc = str(c).lower().strip()
        if lc == "date":
            col_map[c] = "Date"
        elif lc == "open":
            col_map[c] = "Open"
        elif lc == "high":
            col_map[c] = "High"
        elif lc == "low":
            col_map[c] = "Low"
        elif lc == "close":
            col_map[c] = "Close"
        elif lc == "volume":
            col_map[c] = "Volume"
    df = df.rename(columns=col_map)

    if "Date" not in df.columns:
        raise ValueError(f"Missing 'Date' column in {path}. Got columns: {list(df.columns)}")

    if "Open" not in df.columns:
        if "Close" not in df.columns:
            raise ValueError(f"Missing both 'Open' and 'Close' columns in {path}. Got columns: {list(df.columns)}")
        df["Open"] = df["Close"]

    if "Close" not in df.columns:
        raise ValueError(f"Missing 'Close' column in {path}. Got columns: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# =========================
# Returns & Indicators
# =========================

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Open-to-next-open returns (used by backtest)
    out["ret_oo"] = out["Open"].shift(-1) / out["Open"] - 1.0

    # For vol targeting we want realized daily returns; using close-to-close is fine for proxy
    out["ret_cc"] = out["Close"].pct_change()

    return out


def add_rsi_wilder(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing is EMA with alpha=1/window
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_indicators(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Compute RSI + MA + MACD columns."""
    out = df.copy()
    close = out["Close"]

    # RSI
    out["RSI"] = add_rsi_wilder(close, cfg.rsi_window)

    # MA crossover
    out["MA_fast"] = close.rolling(cfg.ma_fast).mean()
    out["MA_slow"] = close.rolling(cfg.ma_slow).mean()
    out["MA_trend_up"] = out["MA_fast"] > out["MA_slow"]

    # MACD (DIF/DEA/HIST)
    ema_fast = close.ewm(span=cfg.macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=cfg.macd_slow, adjust=False).mean()
    out["MACD_dif"] = ema_fast - ema_slow
    out["MACD_dea"] = out["MACD_dif"].ewm(span=cfg.macd_signal, adjust=False).mean()
    out["MACD_hist"] = out["MACD_dif"] - out["MACD_dea"]
    out["MACD_bull"] = out["MACD_dif"] > out["MACD_dea"]
    out["MACD_bear"] = out["MACD_dif"] < out["MACD_dea"]

    return out


def summarize_latest_signals(df: pd.DataFrame, cfg: BacktestConfig) -> Dict[str, str]:
    """Return a compact, LLM-friendly snapshot of the *latest* indicator states."""
    if len(df) == 0:
        return {"error": "empty dataframe"}

    latest = df.iloc[-1]

    # RSI regime
    rsi = float(latest.get("RSI", np.nan))
    if np.isnan(rsi):
        rsi_state = "unknown"
    elif rsi < 30:
        rsi_state = "oversold"
    elif rsi > 70:
        rsi_state = "overbought"
    else:
        rsi_state = "neutral"

    ma_state = "bullish" if bool(latest.get("MA_trend_up", False)) else "bearish"
    macd_state = "bullish" if bool(latest.get("MACD_bull", False)) else "bearish"

    return {
        "RSI_value": f"{rsi:.2f}" if not np.isnan(rsi) else "nan",
        "RSI_state": rsi_state,
        "MA_fast": f"{float(latest.get('MA_fast', np.nan)):.4f}" if not pd.isna(latest.get("MA_fast", np.nan)) else "nan",
        "MA_slow": f"{float(latest.get('MA_slow', np.nan)):.4f}" if not pd.isna(latest.get("MA_slow", np.nan)) else "nan",
        "MA_trend": ma_state,
        "MACD_dif": f"{float(latest.get('MACD_dif', np.nan)):.6f}" if not pd.isna(latest.get("MACD_dif", np.nan)) else "nan",
        "MACD_dea": f"{float(latest.get('MACD_dea', np.nan)):.6f}" if not pd.isna(latest.get("MACD_dea", np.nan)) else "nan",
        "MACD_hist": f"{float(latest.get('MACD_hist', np.nan)):.6f}" if not pd.isna(latest.get("MACD_hist", np.nan)) else "nan",
        "MACD_trend": macd_state,
    }


# =========================
# Trading Signals (RSI + MA + MACD fusion)
# =========================

def generate_signals(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """
    Long-only fused strategy.

    Entry:
      - RSI crosses ABOVE rsi_entry
      - AND MA_fast > MA_slow (trend filter)
      - AND MACD DIF > DEA (momentum confirmation)

    Exit:
      - RSI crosses BELOW rsi_exit (cross-down)
      - OR (optional) MA trend turns bearish
      - OR (optional) MACD turns bearish
    """
    out = df.copy()
    rsi = out["RSI"]

    # RSI triggers
    rsi_entry = (rsi > cfg.rsi_entry) & (rsi.shift(1) <= cfg.rsi_entry)
    rsi_exit  = (rsi < cfg.rsi_exit)  & (rsi.shift(1) >= cfg.rsi_exit)

    # Confirmations (handle missing columns gracefully)
    ma_ok = out["MA_trend_up"] if "MA_trend_up" in out.columns else pd.Series(True, index=out.index)
    macd_ok = out["MACD_bull"] if "MACD_bull" in out.columns else pd.Series(True, index=out.index)

    entry_cond = rsi_entry & ma_ok & macd_ok

    exit_cond = rsi_exit.copy()
    if cfg.exit_on_trend_break:
        if "MA_trend_up" in out.columns:
            exit_cond = exit_cond | (~out["MA_trend_up"])
        if "MACD_bear" in out.columns:
            exit_cond = exit_cond | (out["MACD_bear"])

    pos_raw = np.zeros(len(out), dtype=float)
    in_pos = False

    for i in range(len(out)):
        if in_pos and bool(exit_cond.iloc[i]):
            in_pos = False
        elif (not in_pos) and bool(entry_cond.iloc[i]):
            in_pos = True
        pos_raw[i] = 1.0 if in_pos else 0.0

    out["pos_raw"] = pos_raw
    return out


# =========================
# Position Sizing (Vol Targeting)
# =========================

def add_position_sizing(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    out = df.copy()

    # realized vol proxy using close-to-close returns
    vol_daily = out["ret_cc"].rolling(cfg.vol_window).std()
    vol_annual = vol_daily * np.sqrt(TRADING_DAYS)

    # avoid division by very small vol
    vol_annual = vol_annual.clip(lower=cfg.min_vol_annual)

    target = cfg.target_vol_annual
    leverage = target / vol_annual
    leverage = leverage.clip(upper=cfg.max_leverage)

    out["leverage"] = leverage.fillna(0.0)
    out["position"] = out["pos_raw"] * out["leverage"]
    return out


# =========================
# Backtest
# =========================

def backtest(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    out = df.copy()

    if cfg.use_position_sizing:
        out = add_position_sizing(out, cfg)
    else:
        out["position"] = out["pos_raw"]

    # Next-day open execution:
    # use position decided at t close, applied to t->t+1 open-to-open return
    out["position_lag"] = out["position"].shift(1).fillna(0.0)

    # transaction cost on turnover (absolute change in position)
    turnover = (out["position_lag"] - out["position_lag"].shift(1)).abs().fillna(0.0)
    cost = turnover * (cfg.tx_cost_bps / 1e4)

    out["strategy_ret"] = out["position_lag"] * out["ret_oo"] - cost
    out["equity"] = (1.0 + out["strategy_ret"].fillna(0.0)).cumprod()

    return out


# =========================
# Metrics
# =========================

def compute_metrics(df: pd.DataFrame, cfg: BacktestConfig) -> Dict[str, float]:
    out = {}
    x = df.dropna(subset=["strategy_ret"]).copy()
    if len(x) < 2:
        return {
            "TotalReturn": 0.0,
            "CAGR": 0.0,
            "Sharpe": 0.0,
            "MaxDD": 0.0,
            "HitRate": 0.0,
            "Trades": 0.0,
        }

    total_ret = float(x["equity"].iloc[-1] - 1.0)

    # CAGR
    days = (x["Date"].iloc[-1] - x["Date"].iloc[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = float((x["equity"].iloc[-1]) ** (1 / years) - 1)

    # Sharpe
    rf_daily = cfg.risk_free_rate / TRADING_DAYS
    excess = x["strategy_ret"] - rf_daily
    vol = excess.std()
    sharpe = float(np.sqrt(TRADING_DAYS) * excess.mean() / vol) if vol and vol > 0 else 0.0

    # Max Drawdown
    eq = x["equity"]
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    maxdd = float(dd.min())

    # Hit rate
    hit = float((x["strategy_ret"] > 0).mean())

    # Trades (approx): count turnover events from 0 to >0
    pos = x["position"].fillna(0.0)
    trades = float(((pos > 0) & (pos.shift(1) <= 0)).sum())

    out.update(
        {
            "TotalReturn": total_ret,
            "CAGR": cagr,
            "Sharpe": sharpe,
            "MaxDD": maxdd,
            "HitRate": hit,
            "Trades": trades,
        }
    )
    return out


# =========================
# Plot
# =========================

def plot_equity(df: pd.DataFrame, outdir: Path, ticker: str = "") -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "equity_curve.png"

    fig = plt.figure(figsize=(10, 4))
    plt.plot(df["Date"], df["equity"])
    plt.title(f"Equity Curve {ticker}".strip())
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# =========================
# Reusable base df + splits
# =========================

def prepare_base_df(csv_path: Path, cfg: BacktestConfig) -> pd.DataFrame:
    """Prepare reusable base dataframe (prices + returns + indicators) once."""
    df = load_ohlcv_csv(csv_path)
    df = add_returns(df)
    df = add_indicators(df, cfg)
    return df


def train_test_split(df: pd.DataFrame, train_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split = pd.to_datetime(train_end)
    train = df[df["Date"] <= split]
    test = df[df["Date"] > split]
    return train, test


def compute_split_metrics(df: pd.DataFrame, cfg: BacktestConfig):
    train, test = train_test_split(df, cfg.train_end)
    return {
        "train": compute_metrics(train, cfg),
        "test": compute_metrics(test, cfg),
        "full": compute_metrics(df, cfg),
    }


# =========================
# Parameter Search on TRAIN (searches RSI thresholds)
# =========================

def grid_search_rsi_params(
    base_df: pd.DataFrame,
    cfg: BacktestConfig,
    entry_grid: List[float],
    exit_grid: List[float],
    objective: str = "Sharpe",
    min_trades_train: int = 3,
) -> Tuple[BacktestConfig, pd.DataFrame]:
    """
    Search (rsi_entry, rsi_exit) on TRAIN split only, then return best cfg and result table.

    NOTE (v3): the evaluated strategy is the *fused* strategy (RSI trigger + MA + MACD confirmations).
    """
    objective = objective.strip()
    if objective not in {"Sharpe", "CAGR", "Calmar"}:
        raise ValueError("objective must be one of: Sharpe, CAGR, Calmar")

    rows = []
    best_score = -np.inf
    best_cfg: Optional[BacktestConfig] = None

    for entry in entry_grid:
        for exit_ in exit_grid:
            if exit_ <= entry:
                continue

            tmp_cfg = BacktestConfig(**{**cfg.__dict__, "rsi_entry": float(entry), "rsi_exit": float(exit_)})
            tmp = generate_signals(base_df, tmp_cfg)
            tmp = backtest(tmp, tmp_cfg)

            tmp_train = tmp[tmp["Date"] <= pd.to_datetime(cfg.train_end)]
            m = compute_metrics(tmp_train, tmp_cfg)

            trades = m.get("Trades", 0.0)
            if trades < min_trades_train:
                score = -np.inf
            else:
                if objective == "Sharpe":
                    score = m.get("Sharpe", 0.0)
                elif objective == "CAGR":
                    score = m.get("CAGR", 0.0)
                else:
                    maxdd = abs(m.get("MaxDD", 0.0)) or 1e-9
                    score = m.get("CAGR", 0.0) / maxdd

            rows.append(
                {
                    "rsi_entry": float(entry),
                    "rsi_exit": float(exit_),
                    "train_Sharpe": m.get("Sharpe", 0.0),
                    "train_CAGR": m.get("CAGR", 0.0),
                    "train_MaxDD": m.get("MaxDD", 0.0),
                    "train_Trades": trades,
                    "objective": objective,
                    "objective_score": score,
                }
            )

            if score > best_score:
                best_score = score
                best_cfg = tmp_cfg

    res = pd.DataFrame(rows).sort_values("objective_score", ascending=False).reset_index(drop=True)
    if best_cfg is None:
        best_cfg = cfg
    return best_cfg, res
