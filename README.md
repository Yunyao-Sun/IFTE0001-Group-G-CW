# IFTE0001-Group-G-CW
# 📈 AI-Driven Financial Strategy & Stock Analysis Crew

This project (Agent 1) is an advanced multi-agent financial analysis system built on the **CrewAI** framework. It automates the process of gathering market data, performing technical backtesting, and synthesizing professional investment reports. Unlike standard AI summaries, this system integrates a custom **Quantitative Backtesting Engine** to provide data-validated insights.
Agent 1 is the chosen best agent that is presented in the main branch. Other agents are involved in other branches. 

---

## 🌟 Key Features

* **Autonomous Technical Backtesting**: Features a custom `TechnicalAnalysisTool` that downloads historical data via `yfinance`, executes a **MACD Crossover Strategy**, and visualizes trading signals.
* **Quantitative Performance Metrics**: Automatically calculates and compares **Total Return, CAGR, Sharpe Ratio, Max Drawdown**, and **Win Rate** against a "Buy & Hold" benchmark.
* **Specialized Multi-Agent Workflow**: Leverages distinct agents—Financial Analyst, Research Analyst, and Investment Advisor—to handle different stages of the investment research lifecycle.
* **Automated Report Synthesis**: A post-processing pipeline in `main.py` merges LLM-generated analysis with locally generated Matplotlib charts into a single, cohesive Markdown report.
* **Environment Resilience**: Implements a startup sequence that clears conflicting API environment variables to ensure stable connectivity with OpenAI.

---

## 🏗️ Project Architecture

* **`main.py`**: The entry point. It handles environment configuration, initializes the Crew process, and compiles the final visual report using the `save_markdown_report` function.
* **`crew.py`**: The orchestration layer. It defines the `StockAnalysisCrew` class, configures the `gpt-4o` LLM, and assigns tools like the `TechnicalAnalysisTool` and `CalculatorTool` to specific agents.
* **`tools/yfinance_tools3.py`**: The core technical engine. It performs vectorized backtesting and generates visual artifacts such as Equity Curves and Signal Charts.
* **`config/`**: Contains `agents.yaml` and `tasks.yaml`, which decouple the agent logic and task descriptions from the Python code.


---

## 🛠️ Installation & Setup

### 1. Prerequisites
* Python 3.10 or higher.
* An OpenAI API Key with access to `gpt-4o`.

### 2. Environment Setup
Create a `.env` file in your project root:
```text
OPENAI_API_KEY=sk-proj-your-api-key-here
OPENAI_MODEL_NAME=gpt-4o
```
### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 How to Run
### 1.Open main.py and set the target_stock (e.g., NVDA, AAPL, TSLA).

### 2. Execute the script:
```base
python main.py

```
 ### 3. Check the generated file: [TICKER]_final_report.md.


## 📊 Visual Analysis Example
The system generates professional visualizations for every analysis:

MACD Signals: Highlights every Buy/Sell trigger point.

Equity Curve: Visualizes the strategy performance over time versus the market.

## ⚠️ Disclaimer
This project is for educational purposes only. It does not constitute financial advice. The backtesting results are based on historical data and do not guarantee future performance.

