# Technical Agent v3 — Coursework Demo

This project implements a Technical Analysis AI Agent that downloads price data, optimizes RSI parameters, generates trading signals, runs backtests, and produces structured output files for coursework evaluation.

--------------------------------------------------

PROJECT STRUCTURE

.
├── run_demo.py
├── llm_report.py
├── make_report_evidence.py
├── src/
│   └── agent.py
└── outputs/

--------------------------------------------------

ENVIRONMENT SETUP

Create environment:

python -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements3.txt

--------------------------------------------------

RUN PIPELINE

Run demo:

python run_demo.py

Or specify ticker and history length:

python run_demo.py --symbol AAPL --years 10

--------------------------------------------------

SUCCESS OUTPUT MESSAGE

If successful, terminal will display:

===== PIPELINE FINISHED (Technical Agent v3) =====
Raw Yahoo data: outputs/raw_prices.csv
Param search table: outputs/param_search_results.csv
Selected params: outputs/selected_params.json
Backtest data: outputs/backtest_full.csv
Metrics: outputs/metrics.json
LLM report: outputs/llm_report.md

--------------------------------------------------

OUTPUT FILES (outputs/ folder)

Main generated files:

outputs/raw_prices.csv
outputs/raw_price_yahoo.csv
outputs/param_search_results.csv
outputs/selected_params.json
outputs/backtest_full.csv
outputs/backtest_output.csv
outputs/metrics.json
outputs/metrics_long.csv
outputs/equity_curve.png
outputs/equity_BP.png
outputs/llm_report.md

--------------------------------------------------

OPTIONAL LLM REPORT

If you want to generate the LLM report:

Create .env file in project root:

OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini

Then run again:

python run_demo.py

--------------------------------------------------

EVIDENCE SCRIPT

Optional evidence generator:

python make_report_evidence.py

--------------------------------------------------

TROUBLESHOOTING

If tree command not found:

Use:

find . -maxdepth 3 -type f

If Yahoo data download fails:

Try another symbol or reduce years parameter.
