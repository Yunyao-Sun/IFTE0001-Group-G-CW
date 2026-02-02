# BP Equity Valuation & Investment Memo Generator

This project implements a **fundamental equity valuation pipeline** for **BP p.l.c. (ADR)**, combining
financial statement analysis, ratio computation, discounted cash flow (DCF) valuation,
peer multiple cross-checks, and a structured investment committee (IC) memo generated via a local LLM.

The codebase is designed as a **deterministic financial analysis engine** with a **controlled LLM layer**
used strictly for narrative synthesis, in line with buy-side investment research workflows.

---

## Project Overview

The pipeline performs the following tasks:

1. **Data Acquisition**
   - Annual financial statements (Income Statement, Balance Sheet, Cash Flow) from Alpha Vantage
   - Market prices and peer multiples from Yahoo Finance (via `yfinance`)

2. **Data Cleaning & Validation**
   - Year filtering (2020–2024)
   - Numeric coercion and accounting consistency checks
   - Guardrails for missing fields and API rate limits

3. **Fundamental Analysis**
   - Profitability, leverage, liquidity, efficiency, and cash flow ratios
   - Historical free cash flow (FCF) reconstruction

4. **Valuation**
   - DCF valuation using historical FCF growth
   - Terminal value estimation
   - Equity value and implied share price
   - Market price comparison and recommendation logic (BUY / HOLD / SELL)

5. **Multiples Cross-Check**
   - Peer comparison using:
     - P/E
     - EV / EBITDA
     - EV / Operating Cash Flow
   - Implied equity values and share prices

6. **Risk Assessment**
   - Fundamental risk proxies (leverage, liquidity, volatility)
   - ESG proxy assessment (environmental, social, governance)

7. **LLM-Generated Investment Memo**
   - Strictly controlled Markdown IC memo
   - LLM used **only** for narrative synthesis
   - No invented assumptions or data leakage


