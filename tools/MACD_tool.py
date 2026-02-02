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
class MACDCfg:
    fast: int
    slow: int
    signal: int
    label: str

@dataclass
class BacktestConfig:
    ticker: str
    start_date: str = (datetime.now() - timedelta(days=365*10)).strftime("%Y-%m-%d")
    initial_capital: float = 10000.0
    tx_cost_bps: float = 10.0      
    target_ann_vol: float = 0.40   
    max_leverage: float = 1.5

class MACDTechnicalInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., NVDA, AAPL, SPY).")

class MACDTechnicalTool(BaseTool):
    name: str = "Enhanced MACD Strategy Optimizer"
    description: str = (
        "Automatically identifies the best MACD settings (e.g., 12/26 vs 5/35) for a stock "
        "by backtesting multiple configurations and selecting the one with the highest Sharpe Ratio."
    )
    args_schema: Type[BaseModel] = MACDTechnicalInput

    def _run(self, ticker: str) -> str:
        try:
            cfg = BacktestConfig(ticker=ticker)
            
            combos = [
                MACDCfg(12, 26, 9, "Standard/Classic"),
                MACDCfg(5, 35, 5, "Trend-Focused/Aggressive"),
                MACDCfg(24, 52, 18, "Conservative/Slow-Moving")
            ]
            
            raw_df = yf.download(ticker, start=cfg.start_date, progress=False, auto_adjust=True)
            if raw_df.empty: return f"Error: No data for {ticker}."
            if isinstance(raw_df.columns, pd.MultiIndex): raw_df.columns = raw_df.columns.get_level_values(0)
            df_base = raw_df.reset_index()
            
            best_sharpe = -np.inf
            best_df = None
            best_cfg = None

            for macd_cfg in combos:
                current_df = self._execute_macd_backtest(df_base.copy(), cfg, macd_cfg)
                
                daily_ret = current_df["equity_curve"].pct_change().fillna(0.0)
                sharpe = (daily_ret.mean() / daily_ret.std()) * (252 ** 0.5) if daily_ret.std() > 0 else 0
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_df = current_df
                    best_cfg = macd_cfg

            s_ret, s_cagr, s_dd, s_sharpe = self._compute_metrics(best_df)
            bh_eq = best_df["Close"] / best_df["Close"].iloc[0]
            b_ret, b_cagr, b_dd, b_sharpe = self._compute_metrics_from_series(bh_eq, best_df)
            win_rate, total_trades = self._calculate_win_rate(best_df)

            self._save_macd_plots(best_df, ticker, best_cfg)
            
            return f"""
            # Enhanced MACD Analysis Report for {ticker}
            
            ## Optimization Results
            - **Selected Settings**: Fast={best_cfg.fast}, Slow={best_cfg.slow}, Signal={best_cfg.signal}
            - **Strategy Type**: {best_cfg.label}
            
            ## Performance Comparison (Optimized)
            | Metric | Optimized MACD Strategy | Buy & Hold |
            |--------|-------------------------|------------|
            | **Total Return** | {s_ret:.2%} | {b_ret:.2%} |
            | **CAGR** | {s_cagr:.2%} | {b_cagr:.2%} |
            | **Sharpe Ratio** | {s_sharpe:.2f} | {b_sharpe:.2f} |
            | **Max Drawdown** | {s_dd:.2%} | {b_dd:.2%} |
            | **Win Rate** | {win_rate:.2%} | N/A |
            | **Total Trades** | {total_trades} | 1 |
            
            ## Technical Summary
            - **Current MACD Line**: {best_df.iloc[-1]['MACD_Line']:.4f}
            - **Current Signal Line**: {best_df.iloc[-1]['Signal_Line']:.4f}
            - **Stance**: {"BULLISH (Long)" if best_df.iloc[-1]['signal'] == 1 else "BEARISH (Cash)"}
            
            Charts saved as {ticker}_macd_optimized.png
            """
        except Exception as e:
            return f"Error in MACD optimization: {str(e)}"

    def _execute_macd_backtest(self, df, cfg, m_cfg):
        ema_f = df["Close"].ewm(span=m_cfg.fast, adjust=False).mean()
        ema_s = df["Close"].ewm(span=m_cfg.slow, adjust=False).mean()
        df["MACD_Line"] = ema_f - ema_s
        df["Signal_Line"] = df["MACD_Line"].ewm(span=m_cfg.signal, adjust=False).mean()
        df["MACD_Hist"] = df["MACD_Line"] - df["Signal_Line"]
        
        df["signal"] = np.where(df["MACD_Line"] > df["Signal_Line"], 1.0, 0.0)
        
        df["Ret"] = df["Close"].pct_change().fillna(0)
        df["Realized_Vol"] = df["Ret"].rolling(20).std() * math.sqrt(252)
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

    def _save_macd_plots(self, df: pd.DataFrame, ticker: str, m_cfg: MACDCfg):
        output_dir = "analysis_output"
        os.makedirs(output_dir, exist_ok=True)
        plt.switch_backend('Agg')
        sub = df.tail(500)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        ax1.plot(sub["Date"], sub["Close"], color="black", alpha=0.3, label="Price")
        buys = sub[sub["action"] == 1]
        ax1.scatter(buys["Date"], buys["Close"], marker="^", color="green", s=100, label="Buy Signal")
        ax1.set_title(f"{ticker} Best MACD Fit: {m_cfg.fast}/{m_cfg.slow}/{m_cfg.signal} ({m_cfg.label})")
        ax1.legend()

        ax2.plot(sub["Date"], sub["MACD_Line"], label="MACD", color="blue")
        ax2.plot(sub["Date"], sub["Signal_Line"], label="Signal", color="orange", linestyle="--")
        colors = np.where(sub["MACD_Hist"] >= 0, 'green', 'red')
        ax2.bar(sub["Date"], sub["MACD_Hist"], color=colors, alpha=0.3)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{ticker}_macd_optimized.png")
        plt.close()