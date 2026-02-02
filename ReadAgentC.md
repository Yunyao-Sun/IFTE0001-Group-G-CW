# Fundamental Analyst AI Agent – BP plc

## Project Overview
This project develops a Fundamental Analyst AI Agent for equity valuation and investment decision-making, using BP plc as a case study. The agent automates key tasks in buy-side fundamental research, including financial data processing, ratio analysis, intrinsic and relative valuation, and structured investment decision support.

The system is designed to enhance consistency, transparency, and reproducibility in fundamental analysis by combining deterministic financial computation with large language model (LLM)–based reasoning.

---

## Data Sources
The agent uses publicly available financial and market data for BP plc covering the period 2020–2024.

- Financial statements: Yahoo Finance, Alpha Vantage  
- Market price data: Yahoo Finance (year-end close)

All numerical inputs used by the agent are computed in Python; the LLM does not perform financial calculations.

---

## Agent Design
The agent follows a modular architecture with a clear separation between numerical analysis and reasoning:

1. **Data Collection and Cleaning**  
   Financial statement and market data are standardised, validated, and aligned by fiscal year.

2. **Deterministic Financial Analysis**  
   - Profitability, growth, and solvency ratios  
   - Discounted Cash Flow (DCF) intrinsic valuation  
   - Relative valuation using market and peer multiples  

3. **LLM Decision Layer**  
   A Llama 3 model (accessed via the Groq API) is used as a constrained reasoning layer.  
   The model receives structured, pre-computed financial signals and produces a final BUY / HOLD / SELL recommendation with explicit quantitative evidence.

4. **Explainable Output**  
   The agent outputs a structured decision summary, including rationale and supporting metrics, suitable for investment research reporting.

---


