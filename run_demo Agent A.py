#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_demo.py — short demo script for reproducing key outputs

What it reproduces (key outputs):
- Financial ratios (ROE, Net Margin, EBITDA Margin, Leverage, Current Ratio, etc.)
- DCF intrinsic price + market comparison + recommendation
- Peer multiples + implied share prices (P/E, EV/EBITDA, EV/Operating CF)
- Risk dashboard (ESG proxy + fundamental risks)
- Exports CSV/PNG into ./outputs/

Prereqs:
- pip install -r requirements.txt
- (optional) export ALPHAVANTAGE_API_KEY="your_key"
"""

import os
import sys
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import yfinance as yf


# =========================
# Config
# =========================
BASE_AV = "https://www.alphavantage.co/query"
SYMBOL_DEFAULT = "BP"  # BP ADR ticker used in your project
START_YEAR_DEFAULT = 2020
END_YEAR_DEFAULT = 2024

OUTPUT_DIR = "outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

PEERS_DEFAULT = {
    "BP": "BP",
    "Shell": "SHEL",
    "ExxonMobil": "XOM",
}

# DCF parameters (kept identical spirit to your notebook/script)
RISK_FREE_RATE = 0.04
MARKET_RISK_PREMIUM = 0.06
TERMINAL_GROWTH = 0.02
FORECAST_YEARS = 5
UPSIDE_THRESHOLD = 0.15  # BUY/SELL threshold


# =========================
# Utilities
# =========================
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def save_df(df: pd.DataFrame, name: str):
    """Save both CSV and Excel to outputs/"""
    csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    xlsx_path = os.path.join(OUTPUT_DIR, f"{name}.xlsx")
    df.to_csv(csv_path, index=True, encoding="utf-8-sig")
    try:
        df.to_excel(xlsx_path, index=True)
    except Exception:
        # Excel export is nice-to-have; don't fail the demo if openpyxl is missing
        pass
    return csv_path, xlsx_path


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =========================
# Alpha Vantage fetch + clean
# =========================
def fetch_alpha_vantage(function_name: str, symbol: str, start_year: int, end_year: int) -> pd.DataFrame:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ALPHAVANTAGE_API_KEY not found. "
            "Set it via: export ALPHAVANTAGE_API_KEY='your_key' (mac/linux) "
            "or setx ALPHAVANTAGE_API_KEY \"your_key\" (windows)."
        )

    params = {"function": function_name, "symbol": symbol, "apikey": api_key}
    r = requests.get(BASE_AV, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Alpha Vantage may return a rate-limit "Note"
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage rate limit triggered: {data['Note']}")

    reports = data.get("annualReports", [])
    df = pd.DataFrame(reports)

    if "fiscalDateEnding" in df.columns:
        df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"], errors="coerce")
        df = df[df["fiscalDateEnding"].dt.year.between(start_year, end_year)]
        df = df.sort_values("fiscalDateEnding", ascending=True)

    return df


def clean_financial_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"{name}: empty or invalid DataFrame.")

    if "fiscalDateEnding" not in df.columns:
        raise ValueError(f"{name}: missing fiscalDateEnding.")

    out = df.copy()
    out["fiscalDateEnding"] = pd.to_datetime(out["fiscalDateEnding"], errors="coerce")
    out = out.dropna(subset=["fiscalDateEnding"]).set_index("fiscalDateEnding").sort_index()

    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


# =========================
# Ratios
# =========================
def compute_ratios(bs: pd.DataFrame, is_: pd.DataFrame, cf: pd.DataFrame) -> pd.DataFrame:
    def safe_get(df, col):
        return df[col] if col in df.columns else None

    net_income = safe_get(is_, "netIncome")
    total_assets = safe_get(bs, "totalAssets")
    total_liabilities = safe_get(bs, "totalLiabilities")

    equity = None
    if total_assets is not None and total_liabilities is not None:
        equity = total_assets - total_liabilities

    revenue = safe_get(is_, "totalRevenue")
    ebitda = safe_get(is_, "ebitda")

    ratios = pd.DataFrame(index=bs.index)

    if net_income is not None and equity is not None:
        ratios["ROE"] = net_income / equity
    if net_income is not None and total_assets is not None:
        ratios["ROA"] = net_income / total_assets
    if net_income is not None and revenue is not None:
        ratios["Net_Margin"] = net_income / revenue
    if ebitda is not None and revenue is not None:
        ratios["EBITDA_Margin"] = ebitda / revenue
    if total_liabilities is not None and equity is not None:
        ratios["Leverage"] = total_liabilities / equity

    current_assets = safe_get(bs, "totalCurrentAssets")
    current_liabilities = safe_get(bs, "totalCurrentLiabilities")
    if current_assets is not None and current_liabilities is not None:
        ratios["Current_Ratio"] = current_assets / current_liabilities

    interest_expense = safe_get(is_, "interestExpense")
    ebit = safe_get(is_, "ebit")
    if ebit is not None and interest_expense is not None:
        ratios["Interest_Coverage"] = ebit / interest_expense.abs()

    if revenue is not None and total_assets is not None:
        ratios["Asset_Turnover"] = revenue / total_assets

    op_cf = safe_get(cf, "operatingCashflow")
    capex = safe_get(cf, "capitalExpenditures")
    if op_cf is not None and capex is not None:
        ratios["FCF"] = op_cf - capex
        ratios["FCF_Margin"] = ratios["FCF"] / revenue if revenue is not None else np.nan

    return ratios


def compute_historical_fcf(cf: pd.DataFrame) -> pd.Series:
    op = pd.to_numeric(cf.get("operatingCashflow"), errors="coerce")
    capex = pd.to_numeric(cf.get("capitalExpenditures"), errors="coerce")
    fcf = op - capex
    return fcf.rename("FCF")


def forecast_fcf_4yr_cagr(hist_fcf: pd.Series, years: int = 5):
    hist = hist_fcf.dropna()
    recent = hist.tail(4)
    if len(recent) < 2:
        last = float(hist.iloc[-1])
        idx = pd.date_range(start=hist.index[-1] + pd.DateOffset(years=1), periods=years, freq="YE")
        return pd.Series([last] * years, index=idx, name="FCF"), 0.0

    start = float(recent.iloc[0])
    end = float(recent.iloc[-1])
    n = len(recent) - 1

    cagr = 0.0 if (start <= 0 or end <= 0) else (end / start) ** (1 / n) - 1

    last = float(recent.iloc[-1])
    fcfs = []
    for _ in range(years):
        last *= (1 + cagr)
        fcfs.append(last)

    idx = pd.date_range(start=hist.index[-1] + pd.DateOffset(years=1), periods=years, freq="YE")
    return pd.Series(fcfs, index=idx, name="FCF"), float(cagr)


# =========================
# Overview + WACC (using Alpha Vantage OVERVIEW)
# =========================
def fetch_overview(symbol: str) -> dict:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not found (required for OVERVIEW).")

    params = {"function": "OVERVIEW", "symbol": symbol, "apikey": api_key}
    r = requests.get(BASE_AV, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage rate limit triggered: {data['Note']}")
    return data


def estimate_wacc(bs: pd.DataFrame, is_: pd.DataFrame, overview: dict):
    market_cap = float(overview.get("MarketCapitalization", np.nan))
    if np.isnan(market_cap) or market_cap <= 0:
        raise ValueError("MarketCapitalization missing/invalid from OVERVIEW.")

    # Debt
    if "totalDebt" in bs.columns and not pd.isna(bs["totalDebt"].iloc[-1]):
        total_debt = float(bs["totalDebt"].iloc[-1])
    else:
        lt = float(bs["longTermDebt"].iloc[-1]) if "longTermDebt" in bs.columns else 0.0
        st = float(bs["shortTermDebt"].iloc[-1]) if "shortTermDebt" in bs.columns else 0.0
        total_debt = lt + st

    cash = float(bs["cashAndCashEquivalentsAtCarryingValue"].iloc[-1]) if "cashAndCashEquivalentsAtCarryingValue" in bs.columns else 0.0
    net_debt = total_debt - cash

    ev = market_cap + net_debt
    w_e = market_cap / ev if ev != 0 else 0.7
    w_d = net_debt / ev if ev != 0 else 0.3

    beta_raw = float(overview.get("Beta", np.nan))
    beta_min, beta_max = 0.6, 1.2
    beta = 0.9 if (np.isnan(beta_raw) or beta_raw < beta_min or beta_raw > beta_max) else beta_raw

    cost_of_equity = RISK_FREE_RATE + beta * MARKET_RISK_PREMIUM

    interest_expense = abs(float(is_["interestExpense"].iloc[-1])) if "interestExpense" in is_.columns else 0.0
    cost_of_debt = interest_expense / total_debt if total_debt else 0.05

    tax_rate = 0.25
    wacc = w_e * cost_of_equity + w_d * cost_of_debt * (1 - tax_rate)

    return {
        "market_cap": market_cap,
        "total_debt": total_debt,
        "cash": cash,
        "net_debt": net_debt,
        "ev": ev,
        "w_e": w_e,
        "w_d": w_d,
        "beta": beta,
        "cost_of_equity": cost_of_equity,
        "cost_of_debt": cost_of_debt,
        "tax_rate": tax_rate,
        "wacc": wacc,
    }


# =========================
# DCF + market comparison
# =========================
def terminal_value(last_fcf: float, wacc: float, g: float) -> float:
    return last_fcf * (1 + g) / (wacc - g)


def discount_cashflows(cashflows: pd.Series, rate: float) -> pd.Series:
    years = np.arange(1, len(cashflows) + 1)
    return pd.Series(cashflows.values / ((1 + rate) ** years), index=cashflows.index, name="Discounted_FCF")


def dcf_vs_market(ticker_symbol: str, intrinsic_price: float, threshold: float = UPSIDE_THRESHOLD):
    px = yf.Ticker(ticker_symbol).history(period="10d")["Close"].dropna()
    if px.empty:
        raise ValueError("No market price data retrieved via yfinance.")
    market_price = float(px.iloc[-1])

    upside = intrinsic_price / market_price - 1
    if upside > threshold:
        rec = "BUY"
    elif upside < -threshold:
        rec = "SELL"
    else:
        rec = "HOLD"

    summary = pd.DataFrame(
        {
            "Metric": ["DCF Intrinsic Price (USD)", "Market Price (latest, USD)", "Upside/Downside (%)", "Recommendation"],
            "Value": [round(intrinsic_price, 2), round(market_price, 2), round(upside * 100, 2), rec],
        }
    )
    return summary, upside, rec, market_price


# =========================
# Multiples
# =========================
def fetch_multiples_extended(ticker: str):
    info = yf.Ticker(ticker).info
    market_cap = info.get("marketCap")
    net_income = info.get("netIncomeToCommon")
    operating_cf = info.get("operatingCashflow")
    ebitda = info.get("ebitda")

    total_debt = info.get("totalDebt")
    cash = info.get("totalCash")

    ev = (market_cap + total_debt - cash) if (market_cap and total_debt is not None and cash is not None) else None

    pe = (market_cap / net_income) if (market_cap and net_income and net_income > 0) else np.nan
    ev_ebitda = (ev / ebitda) if (ev and ebitda and ebitda > 0) else np.nan
    ev_ocf = (ev / operating_cf) if (ev and operating_cf and operating_cf > 0) else np.nan

    return {
        "Market Cap": market_cap,
        "EV": ev,
        "Net Income (TTM)": net_income,
        "EBITDA (TTM)": ebitda,
        "Operating CF (TTM)": operating_cf,
        "P/E": pe,
        "EV/EBITDA": ev_ebitda,
        "EV/Operating CF": ev_ocf,
        "Shares Outstanding": info.get("sharesOutstanding"),
    }


def implied_prices_from_multiples(bp_row: pd.Series, industry_median: pd.Series):
    rows = []
    bp_net_income = bp_row["Net Income (TTM)"]
    bp_ebitda = bp_row["EBITDA (TTM)"]
    bp_ocf = bp_row["Operating CF (TTM)"]
    bp_net_debt = bp_row["EV"] - bp_row["Market Cap"]
    shares = bp_row["Shares Outstanding"]

    if shares is None or shares == 0:
        return pd.DataFrame(columns=["Method", "Implied Equity Value", "Implied Share Price"])

    if not np.isnan(industry_median.get("P/E", np.nan)) and bp_net_income and bp_net_income > 0:
        implied_equity = float(industry_median["P/E"]) * float(bp_net_income)
        rows.append({"Method": "P/E", "Implied Equity Value": implied_equity, "Implied Share Price": implied_equity / shares})

    if not np.isnan(industry_median.get("EV/EBITDA", np.nan)) and bp_ebitda and bp_ebitda > 0:
        implied_ev = float(industry_median["EV/EBITDA"]) * float(bp_ebitda)
        implied_equity = implied_ev - float(bp_net_debt)
        rows.append({"Method": "EV/EBITDA", "Implied Equity Value": implied_equity, "Implied Share Price": implied_equity / shares})

    if not np.isnan(industry_median.get("EV/Operating CF", np.nan)) and bp_ocf and bp_ocf > 0:
        implied_ev = float(industry_median["EV/Operating CF"]) * float(bp_ocf)
        implied_equity = implied_ev - float(bp_net_debt)
        rows.append({"Method": "EV/Operating CF", "Implied Equity Value": implied_equity, "Implied Share Price": implied_equity / shares})

    return pd.DataFrame(rows)


# =========================
# ESG proxy + fundamental risks
# =========================
def esg_proxy_assessment(info: dict):
    result = {}
    result["Environmental Risk"] = "High" if info.get("sector") == "Energy" else "Medium"

    employees = info.get("fullTimeEmployees", 0) or 0
    result["Social Risk"] = "Medium" if employees > 50000 else "Low"

    roe = info.get("returnOnEquity", 0) or 0
    debt_to_equity = info.get("debtToEquity", 0) or 0
    result["Governance Risk"] = "Weak" if (roe < 0.05 or debt_to_equity > 150) else "Acceptable"

    result["Overall ESG Risk"] = "Elevated" if result["Environmental Risk"] == "High" else "Moderate"
    return result


def fundamental_risk_from_data(ratio_df: pd.DataFrame, hist_fcf: pd.Series, is_clean: pd.DataFrame):
    risks = {}

    if "Leverage" in ratio_df.columns:
        avg_leverage = float(ratio_df["Leverage"].mean())
        risks["Leverage"] = "High" if avg_leverage > 2.0 else "Medium" if avg_leverage > 1.0 else "Low"
    else:
        risks["Leverage"] = "Unknown"

    if "Current_Ratio" in ratio_df.columns:
        avg_cr = float(ratio_df["Current_Ratio"].mean())
        risks["Liquidity"] = "High" if avg_cr < 1.0 else "Medium" if avg_cr < 1.5 else "Low"
    else:
        risks["Liquidity"] = "Unknown"

    fcf_vol = float(hist_fcf.pct_change().std())
    risks["Cash Flow Stability"] = "High" if fcf_vol > 0.4 else "Medium" if fcf_vol > 0.2 else "Low"

    if "netIncome" in is_clean.columns:
        ni_vol = float(is_clean["netIncome"].pct_change().std())
        risks["Earnings Volatility"] = "High" if ni_vol > 0.4 else "Medium" if ni_vol > 0.2 else "Low"
    else:
        risks["Earnings Volatility"] = "Unknown"

    return risks


# =========================
# Plot helpers
# =========================
def plot_ratio_trends(ratio_df: pd.DataFrame, symbol: str):
    # keep it minimal: profitability + leverage
    cols = [c for c in ["ROE", "Net_Margin", "EBITDA_Margin", "Leverage"] if c in ratio_df.columns]
    if not cols:
        return None

    ax = ratio_df[cols].copy()
    # plot in percentage for margins/returns
    for c in ["ROE", "Net_Margin", "EBITDA_Margin"]:
        if c in ax.columns:
            ax[c] = ax[c] * 100.0

    plt.figure()
    ax.plot(marker="o")
    plt.title(f"{symbol} — Key Ratio Trends")
    plt.xlabel("Fiscal Year End")
    plt.ylabel("Value (%, except Leverage)")
    plt.tight_layout()

    out = os.path.join(FIG_DIR, f"{symbol}_ratio_trends.png")
    plt.savefig(out, dpi=200)
    plt.close()
    return out


# =========================
# Main demo
# =========================
def main():
    ensure_dirs()

    symbol = os.getenv("DEMO_SYMBOL", SYMBOL_DEFAULT)
    start_year = int(os.getenv("DEMO_START_YEAR", START_YEAR_DEFAULT))
    end_year = int(os.getenv("DEMO_END_YEAR", END_YEAR_DEFAULT))

    print_header(f"Demo run — {symbol} ({start_year}-{end_year})")
    print(f"Time: {datetime.utcnow().isoformat()}Z")
    print(f"Output folder: ./{OUTPUT_DIR}/")

    # ---------- Step 1: Data fetch ----------
    av_ok = bool(os.getenv("ALPHAVANTAGE_API_KEY"))
    if not av_ok:
        print("\n[INFO] ALPHAVANTAGE_API_KEY not set → skipping financial statements + DCF.")
        print("       You can still reproduce multiples + ESG proxy outputs.\n")

    bs_clean = is_clean = cf_clean = None
    ratio_df = None
    hist_fcf = None
    dcf_summary = None
    dcf_intrinsic_price = None
    recommendation = None

    if av_ok:
        print_header("1) Fetch & clean Alpha Vantage financial statements")
        try:
            bs = fetch_alpha_vantage("BALANCE_SHEET", symbol, start_year, end_year)
            time.sleep(12)
            is_ = fetch_alpha_vantage("INCOME_STATEMENT", symbol, start_year, end_year)
            time.sleep(12)
            cf = fetch_alpha_vantage("CASH_FLOW", symbol, start_year, end_year)

            bs_clean = clean_financial_df(bs, "Balance Sheet")
            is_clean = clean_financial_df(is_, "Income Statement")
            cf_clean = clean_financial_df(cf, "Cash Flow")

            save_df(bs_clean, f"{symbol}_balance_sheet_{start_year}_{end_year}")
            save_df(is_clean, f"{symbol}_income_statement_{start_year}_{end_year}")
            save_df(cf_clean, f"{symbol}_cash_flow_{start_year}_{end_year}")

            print("[OK] Saved cleaned statements to outputs/ (csv/xlsx).")
        except Exception as e:
            print(f"[WARN] Alpha Vantage part failed: {e}")
            print("       Continuing with yfinance-only outputs.\n")
            bs_clean = is_clean = cf_clean = None

    # ---------- Step 2: Ratios ----------
    if bs_clean is not None and is_clean is not None and cf_clean is not None:
        print_header("2) Compute key ratios + FCF")
        ratio_df = compute_ratios(bs_clean, is_clean, cf_clean)
        hist_fcf = compute_historical_fcf(cf_clean)

        # print a compact view
        show_cols = [c for c in ["ROE", "Net_Margin", "EBITDA_Margin", "Leverage", "Current_Ratio", "Interest_Coverage", "FCF"] if c in ratio_df.columns]
        print(ratio_df[show_cols].round(4))

        save_df(ratio_df, f"{symbol}_ratios_{start_year}_{end_year}")
        save_df(hist_fcf.to_frame(), f"{symbol}_historical_fcf_{start_year}_{end_year}")

        fig = plot_ratio_trends(ratio_df, symbol)
        if fig:
            print(f"[OK] Saved chart: {fig}")

    # ---------- Step 3: DCF ----------
    if bs_clean is not None and is_clean is not None and cf_clean is not None:
        print_header("3) DCF valuation (primary)")
        try:
            overview = fetch_overview(symbol)
            wacc_info = estimate_wacc(bs_clean, is_clean, overview)
            wacc = float(wacc_info["wacc"])

            fcf_forecast, cagr = forecast_fcf_4yr_cagr(hist_fcf, years=FORECAST_YEARS)
            tv = terminal_value(float(fcf_forecast.iloc[-1]), wacc, TERMINAL_GROWTH)
            tv_pv = tv / ((1 + wacc) ** FORECAST_YEARS)
            discounted_fcf = discount_cashflows(fcf_forecast, wacc)

            ev_dcf = float(discounted_fcf.sum() + tv_pv)
            equity_value = ev_dcf - float(wacc_info["net_debt"])

            shares_out = float(overview.get("SharesOutstanding", 1) or 1)
            dcf_intrinsic_price = equity_value / shares_out

            dcf_summary, upside, recommendation, market_price = dcf_vs_market(symbol, dcf_intrinsic_price)
            print(dcf_summary)

            # Save DCF details
            dcf_details = pd.DataFrame(
                {
                    "Item": ["WACC", "Terminal growth", "Forecast CAGR (4yr)", "EV (DCF)", "Equity Value", "Shares Outstanding", "DCF Intrinsic Price", "Market Price", "Upside"],
                    "Value": [wacc, TERMINAL_GROWTH, cagr, ev_dcf, equity_value, shares_out, dcf_intrinsic_price, market_price, upside],
                }
            )
            save_df(dcf_details.set_index("Item"), f"{symbol}_dcf_details")
        except Exception as e:
            print(f"[WARN] DCF failed: {e}")
            dcf_intrinsic_price = None
            recommendation = None

    # ---------- Step 4: Multiples ----------
    print_header("4) Peer multiples + implied prices (cross-check)")
    rows = []
    for name, ticker in PEERS_DEFAULT.items():
        d = fetch_multiples_extended(ticker)
        d["Company"] = name
        rows.append(d)

    df_multiples = pd.DataFrame(rows).set_index("Company")
    core = df_multiples[["P/E", "EV/EBITDA", "EV/Operating CF"]]
    summary_stats = core.agg(["median", "mean", "min", "max"])

    print("\nPeer multiples (core):")
    print(core.round(3))
    print("\nPeer multiples summary:")
    print(summary_stats.round(3))

    save_df(df_multiples, f"{symbol}_peer_multiples_full")
    save_df(summary_stats, f"{symbol}_peer_multiples_summary")

    # implied prices for BP using industry median
    bp_row = df_multiples.loc["BP"]
    industry_median = core.median()
    implied = implied_prices_from_multiples(bp_row, industry_median)

    if dcf_intrinsic_price is not None:
        # include DCF row to match your valuation_summary idea
        implied = pd.concat(
            [
                implied,
                pd.DataFrame(
                    [{"Method": "DCF", "Implied Equity Value": np.nan, "Implied Share Price": float(dcf_intrinsic_price)}]
                ),
            ],
            ignore_index=True,
        )

    print("\nImplied share prices (median multiples):")
    print(implied.round(2))

    save_df(implied.set_index("Method"), f"{symbol}_implied_share_prices")

    # ---------- Step 5: Risks ----------
    print_header("5) Risk dashboard (ESG proxy + fundamentals)")
    bp_info = yf.Ticker(symbol).info
    esg = esg_proxy_assessment(bp_info)

    if ratio_df is not None and hist_fcf is not None and is_clean is not None:
        fund = fundamental_risk_from_data(ratio_df, hist_fcf, is_clean)
    else:
        # if no AV data, provide ESG proxy only
        fund = {
            "Leverage": "Unknown",
            "Liquidity": "Unknown",
            "Cash Flow Stability": "Unknown",
            "Earnings Volatility": "Unknown",
        }

    risk_dashboard = pd.DataFrame(
        {
            "Risk Dimension": [
                "Environmental (ESG)",
                "Social (ESG)",
                "Governance (ESG)",
                "Leverage",
                "Liquidity",
                "Cash Flow Stability",
                "Earnings Volatility",
            ],
            "Risk Level": [
                esg["Environmental Risk"],
                esg["Social Risk"],
                esg["Governance Risk"],
                fund["Leverage"],
                fund["Liquidity"],
                fund["Cash Flow Stability"],
                fund["Earnings Volatility"],
            ],
        }
    )

    print(risk_dashboard)
    save_df(risk_dashboard.set_index("Risk Dimension"), f"{symbol}_risk_dashboard")

    # ---------- Done ----------
    print_header("Done")
    if recommendation:
        print(f"DCF-based Recommendation: {recommendation}")
    print(f"Files saved in: ./{OUTPUT_DIR}/")
    print("[INFO] See IC_memo.md for the full investment memo.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
