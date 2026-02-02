Project Title

**Backtesting of Technical Trading Strategies and LLM-Assisted Trade Report Generation (Agent 2)**


1. Project Overview

This project implements a complete quantitative research workflow for evaluating classical technical trading strategies on a single equity asset and generating a professional trade report using a Large Language Model (LLM).

Using **BP.L (BP plc)** daily price data over the past ten years, the project:

* Backtests multiple technical trading strategies
* Optimises strategy parameters via grid search
* Evaluates performance using standard risk–return metrics
* Compares optimised strategies against a Buy & Hold benchmark
* Automatically generates a structured buy-side style trade report using an LLM

The project is designed as a **research-oriented, reproducible experiment**, suitable for academic coursework, dissertation empirical chapters, or internal quantitative research demonstrations.


2. Data

* **Asset**: BP plc (Ticker: BP.L)
* **Frequency**: Daily
* **Time Horizon**: Last 10 years
* **Source**: Yahoo Finance via `yfinance`
* **Price Used**: Adjusted Close (Close price after cleaning)


3. Evaluated Trading Strategies

The following technical strategies are implemented and tested:

3.1 Moving Average (MA) Strategy

* Signal based on short-term and long-term simple moving average crossover
* Long position when MA(short) > MA(long), otherwise short

3.2 MACD Strategy

* Signal based on MACD line crossing its signal line
* Uses exponential moving averages

3.3 RSI Strategy

* Mean-reversion strategy using RSI thresholds
* Long when RSI < lower bound, short when RSI > upper bound


4. Parameter Optimisation

Each strategy is optimised using a **grid search** over commonly used parameter ranges:

* MA: different short/long window combinations
* MACD: short EMA, long EMA, and signal window
* RSI: window length and upper/lower thresholds

The **best-performing parameter set** for each strategy is selected based on **CAGR**.


5. Performance Metrics

Strategies are evaluated using the following metrics:

* **CAGR (Compound Annual Growth Rate)**
* **Sharpe Ratio** (risk-free rate assumed to be zero)
* **Maximum Drawdown**
* **Hit Rate** (percentage of profitable active trading days)

A **Buy & Hold** strategy is used as the benchmark.


6. Trading Assumptions

* Full investment at all times (long or short)
* Fixed transaction cost of **0.2% per position change**
* No leverage
* No slippage or liquidity constraints
* Short selling is allowed

These assumptions are intentionally simplified to focus on strategy comparison rather than execution realism.


7. LLM-Based Trade Report Generation

After backtesting:

1. Strategy performance results are structured into a text-based summary
2. A constrained prompt is constructed
3. An OpenAI LLM is called to generate a **1–2 page professional trade report**
4. The report is saved locally as: LLM_Trade_Report.txt

The LLM is strictly instructed **not to invent or modify any data**, and to maintain a neutral, buy-side analytical tone.


8. Output

* Console output of strategy comparison metrics
* Cumulative return plot comparing all strategies
* Text-based professional trade report (`LLM_Trade_Report.txt`)



