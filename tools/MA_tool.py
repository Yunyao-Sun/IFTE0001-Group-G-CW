import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Type, List
from dataclasses import dataclass
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


@dataclass
class MACombo:
    fast: int
    slow: int
    label: str

@dataclass
class BacktestConfig:
    ticker: str
    start_date: str = (datetime.now() - timedelta(days=365*10)).strftime("%Y-%m-%d")
    initial_capital: float = 10000.0
    tx_cost_bps: float = 10.0      
    target_ann_vol: float = 0.40   
    max_leverage: float = 1.5      

class MATechnicalInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., NVDA, AAPL).")

class MATechnicalTool(BaseTool):
    name: str = "Enhanced MA Strategy Optimizer"
    description: str = (
        "Automatically identifies the best Moving Average combination (e.g., 20/50, 50/200) "
        "by backtesting multiple timeframes and selecting the one with the highest Sharpe Ratio."
    )
    args_schema: Type[BaseModel] = MATechnicalInput

    def _run(self, ticker: str) -> str:
        try:
            cfg = BacktestConfig(ticker=ticker)
            combos = [
                MACombo(10, 30, "Aggressive (Short-term)"),
                MACombo(20, 50, "Standard (Medium-term)"),
                MACombo(50, 100, "Robust (Long-term)"),
                MACombo(50, 200, "Conservative (Major Trend)")
            ]
            
            raw_df = yf.download(ticker, start=cfg.start_date, progress=False, auto_adjust=True)
            if raw_df.empty: return f"Error: No data for {ticker}."
            if isinstance(raw_df.columns, pd.MultiIndex): raw_df.columns = raw_df.columns.get_level_values(0)
            df_base = raw_df.reset_index()
            
            best_sharpe = -np.inf
            best_df = None
            best_combo = None

            for combo in combos:
                current_df = self._execute_backtest(df_base.copy(), cfg, combo)
                
                daily_ret = current_df["equity_curve"].pct_change().fillna(0.0)
                sharpe = (daily_ret.mean() / daily_ret.std()) * (252 ** 0.5) if daily_ret.std() > 0 else 0
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_df = current_df
                    best_combo = combo

            s_ret, s_cagr, s_dd, s_sharpe = self._compute_metrics(best_df)
            

            bh_eq = best_df["Close"] / best_df["Close"].iloc[0]
            b_ret, b_cagr, b_dd, b_sharpe = self._compute_metrics_from_series(bh_eq, best_df)

    
            win_rate, total_trades = self._calculate_win_rate(best_df)

       
            self._save_optimized_plots(best_df, ticker, best_combo)
            
            return f"""
            # Enhanced MA Analysis Report for {ticker}
            
            ## Optimization Results
            - **Selected Combination**: {best_combo.fast}/{best_combo.slow} ({best_combo.label})
            - **Selection Criteria**: Highest risk-adjusted return (Sharpe Ratio)
            
            ## Performance Comparison (Optimized)
            | Metric | Optimized MA Strategy | Buy & Hold |
            |--------|-----------------------|------------|
            | **Total Return** | {s_ret:.2%} | {b_ret:.2%} |
            | **CAGR** | {s_cagr:.2%} | {b_cagr:.2%} |
            | **Sharpe Ratio** | {s_sharpe:.2f} | {b_sharpe:.2f} |
            | **Max Drawdown** | {s_dd:.2%} | {b_dd:.2%} |
            | **Win Rate** | {win_rate:.2%} | N/A |
            | **Total Trades** | {total_trades} | 1 |
            
            ## Technical Summary
            - **Current Signal**: {"BULLISH (Long)" if best_df.iloc[-1]['signal'] == 1 else "BEARISH (Cash)"}
            - **Logic**: Fast MA ({best_combo.fast}) is currently {"above" if best_df.iloc[-1]['signal'] == 1 else "below"} Slow MA ({best_combo.slow}).
            
            Charts saved to analysis_output/ as {ticker}_ma_optimized.png
            """
        except Exception as e:
            return f"Error in MA optimization: {str(e)}"

    def _execute_backtest(self, df, cfg, combo):
        df["Fast_MA"] = df["Close"].rolling(window=combo.fast).mean()
        df["Slow_MA"] = df["Close"].rolling(window=combo.slow).mean()
        df["Ret"] = df["Close"].pct_change().fillna(0)
        df["Realized_Vol"] = df["Ret"].rolling(20).std() * math.sqrt(252)
        
        df["signal"] = np.where(df["Fast_MA"] > df["Slow_MA"], 1.0, 0.0)
        vol_scalar = (cfg.target_ann_vol / df["Realized_Vol"]).replace([np.inf, -np.inf], 0).fillna(0).clip(upper=cfg.max_leverage)
        
        df["position"] = (df["signal"] * vol_scalar).shift(1).fillna(0)
        cost = df["position"].diff().abs().fillna(0) * (cfg.tx_cost_bps / 10000)
        df["strategy_ret"] = df["position"] * (df["Open"].shift(-1) / df["Open"] - 1).fillna(0) - cost
        df["equity_curve"] = (1 + df["strategy_ret"]).cumprod()
        return df

    def _compute_metrics(self, df):
        return self._compute_metrics_from_series(df["equity_curve"], df)

    def _compute_metrics_from_series(self, equity_series, df):
        final_eq = equity_series.iloc[-1]
        total_ret = final_eq - 1.0
        days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days
        cagr = (final_eq ** (365.0 / days)) - 1.0 if days > 0 else 0.0
        max_dd = (equity_series / equity_series.cummax() - 1.0).min()
        daily_ret = equity_series.pct_change().fillna(0.0)
        sharpe = (daily_ret.mean() / daily_ret.std()) * (252 ** 0.5) if daily_ret.std() > 0 else 0
        return total_ret, cagr, max_dd, sharpe

    def _calculate_win_rate(self, df):
        df['trade_action'] = df['signal'].diff().fillna(0)
        trades = []
        entry_price = 0.0
        in_trade = False
        for row in df.itertuples():
            if row.trade_action == 1:
                entry_price = row.Close
                in_trade = True
            elif row.trade_action == -1 and in_trade:
                trades.append((row.Close / entry_price) - 1.0)
                in_trade = False
        return (sum(1 for t in trades if t > 0) / len(trades) if trades else 0.0), len(trades)

    def _save_optimized_plots(self, df: pd.DataFrame, ticker: str, combo: MACombo):
        output_dir = "analysis_output"
        os.makedirs(output_dir, exist_ok=True)
        plt.switch_backend('Agg') 
        sub = df.tail(500)
        
        plt.figure(figsize=(12, 6))
        plt.plot(sub["Date"], sub["Close"], label="Price", color="black", alpha=0.3)
        plt.plot(sub["Date"], sub["Fast_MA"], label=f"Fast MA ({combo.fast})", color="blue")
        plt.plot(sub["Date"], sub["Slow_MA"], label=f"Slow MA ({combo.slow})", color="orange")
        
        buys = sub[sub["signal"].diff() == 1]
        plt.scatter(buys["Date"], buys["Close"], marker="^", color="green", s=100, label="Buy Signal")
        
        plt.title(f"{ticker} Optimized MA: {combo.fast}/{combo.slow} ({combo.label})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/{ticker}_ma_optimized.png")
        plt.close()