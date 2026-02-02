import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Type
from dataclasses import dataclass
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


@dataclass
class RSICfg:
    entry: float  # buy
    exit: float   # sell
    label: str

@dataclass
class BacktestConfig:
    ticker: str
    start_date: str = (datetime.now() - timedelta(days=365*10)).strftime("%Y-%m-%d")
    initial_capital: float = 10000.0
    tx_cost_bps: float = 10.0      
    target_ann_vol: float = 0.40   
    max_leverage: float = 1.5
    rsi_window: int = 14


class RSITechnicalInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., TSLA, MSFT).")

class RSITechnicalTool(BaseTool):
    name: str = "Enhanced RSI Strategy Optimizer"
    description: str = (
        "Automatically finds the best RSI thresholds (e.g., 30/70 vs 20/80) for a stock "
        "by backtesting multiple combinations and selecting the one with the best risk-adjusted return."
    )
    args_schema: Type[BaseModel] = RSITechnicalInput

    def _run(self, ticker: str) -> str:
        try:
            cfg = BacktestConfig(ticker=ticker)
            
            combos = [
                RSICfg(30, 70, "Standard"),
                RSICfg(25, 75, "Deep Reversion"),
                RSICfg(20, 80, "Extreme/Volatile"),
                RSICfg(35, 65, "Sensitive/Tight")
            ]

            raw_df = yf.download(ticker, start=cfg.start_date, progress=False, auto_adjust=True)
            if raw_df.empty: return f"Error: No data for {ticker}."
            if isinstance(raw_df.columns, pd.MultiIndex): raw_df.columns = raw_df.columns.get_level_values(0)
            df_base = raw_df.reset_index()
            
            df_base = self._calculate_base_rsi(df_base, cfg)
            
            best_sharpe = -np.inf
            best_df = None
            best_cfg = None

            for rsi_cfg in combos:
                current_df = self._execute_rsi_backtest(df_base.copy(), cfg, rsi_cfg)
                
                daily_ret = current_df["equity_curve"].pct_change().fillna(0.0)
                sharpe = (daily_ret.mean() / daily_ret.std()) * (252 ** 0.5) if daily_ret.std() > 0 else 0
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_df = current_df
                    best_cfg = rsi_cfg

            s_ret, s_cagr, s_dd, s_sharpe = self._compute_metrics(best_df)
            bh_eq = best_df["Close"] / best_df["Close"].iloc[0]
            b_ret, b_cagr, b_dd, b_sharpe = self._compute_metrics_from_series(bh_eq, best_df)
            win_rate, total_trades = self._calculate_win_rate(best_df)

            self._save_rsi_plots(best_df, ticker, best_cfg)
            
            return f"""
            # Enhanced RSI Analysis Report for {ticker}
            
            ## Optimization Results
            - **Best RSI Thresholds**: Buy at {best_cfg.entry} / Sell at {best_cfg.exit}
            - **Style**: {best_cfg.label}
            
            ## Performance Comparison (Optimized)
            | Metric | Optimized RSI Strategy | Buy & Hold |
            |--------|-----------------------|------------|
            | **Total Return** | {s_ret:.2%} | {b_ret:.2%} |
            | **CAGR** | {s_cagr:.2%} | {b_cagr:.2%} |
            | **Sharpe Ratio** | {s_sharpe:.2f} | {b_sharpe:.2f} |
            | **Max Drawdown** | {s_dd:.2%} | {b_dd:.2%} |
            | **Win Rate** | {win_rate:.2%} | N/A |
            | **Total Trades** | {total_trades} | 1 |
            
            ## Technical Status
            - **Current RSI**: {best_df.iloc[-1]['RSI']:.2f}
            - **Current Stance**: {"LONG (Oversold Recovery)" if best_df.iloc[-1]['signal'] == 1 else "CASH (Wait for Dip)"}
            
            Visualization saved as {ticker}_rsi_optimized.png
            """
        except Exception as e:
            return f"Error in RSI optimization: {str(e)}"

    def _calculate_base_rsi(self, df, cfg):
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/cfg.rsi_window, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/cfg.rsi_window, adjust=False).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        df["Ret"] = df["Close"].pct_change().fillna(0)
        df["Realized_Vol"] = df["Ret"].rolling(20).std() * math.sqrt(252)
        return df

    def _execute_rsi_backtest(self, df, cfg, rsi_cfg):

        signals = []
        curr = 0.0
        for val in df["RSI"]:
            if val < rsi_cfg.entry: curr = 1.0
            elif val > rsi_cfg.exit: curr = 0.0
            signals.append(curr)
        df["signal"] = signals
        
        vol_scalar = (cfg.target_ann_vol / df["Realized_Vol"]).replace([np.inf, -np.inf], 0).fillna(0).clip(upper=cfg.max_leverage)
        
        df["position"] = (df["signal"] * vol_scalar).shift(1).fillna(0)
        cost = df["position"].diff().abs().fillna(0) * (cfg.tx_cost_bps / 10000)
        df["strategy_ret"] = df["position"] * (df["Open"].shift(-1) / df["Open"] - 1).fillna(0) - cost
        df["equity_curve"] = (1 + df["strategy_ret"]).cumprod()
        return df

    def _compute_metrics(self, df):
        return self._compute_metrics_from_series(df["equity_curve"], df)

    def _compute_metrics_from_series(self, series, df):
        final = series.iloc[-1]
        days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days
        cagr = (final ** (365.0 / days)) - 1.0 if days > 0 else 0
        dd = (series / series.cummax() - 1.0).min()
        daily = series.pct_change().fillna(0)
        sharpe = (daily.mean() / daily.std() * math.sqrt(252)) if daily.std() > 0 else 0
        return (final - 1), cagr, dd, sharpe

    def _calculate_win_rate(self, df):
        df['action'] = df['signal'].diff().fillna(0)
        trades = []
        entry = 0
        active = False
        for row in df.itertuples():
            if row.action == 1:
                entry = row.Close
                active = True
            elif row.action == -1 and active:
                trades.append(row.Close / entry - 1)
                active = False
        return (sum(1 for t in trades if t > 0) / len(trades) if trades else 0), len(trades)

    def _save_rsi_plots(self, df, ticker, rsi_cfg):
        output_dir = "analysis_output"
        os.makedirs(output_dir, exist_ok=True)
        plt.switch_backend('Agg')
        sub = df.tail(500)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        ax1.plot(sub["Date"], sub["Close"], color="black", alpha=0.3, label="Price")
        buys = sub[sub["action"] == 1]
        ax1.scatter(buys["Date"], buys["Close"], marker="^", color="green", s=100, label="Optimized Entry")
        ax1.set_title(f"{ticker} Best RSI Fit: {rsi_cfg.entry}/{rsi_cfg.exit} ({rsi_cfg.label})")
        ax1.legend()
        
        ax2.plot(sub["Date"], sub["RSI"], color="purple", label="RSI(14)")
        ax2.axhline(rsi_cfg.exit, color="red", linestyle="--")
        ax2.axhline(rsi_cfg.entry, color="green", linestyle="--")
        ax2.set_ylim(0, 100)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{ticker}_rsi_optimized.png")
        plt.close()