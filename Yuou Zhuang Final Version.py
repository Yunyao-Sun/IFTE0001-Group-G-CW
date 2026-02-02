#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from datetime import datetime


# In[2]:


BASE = "https://www.alphavantage.co/query"
AV_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

def fetch_alpha_vantage(function_name: str, symbol: str, start_year=2020, end_year=2024) -> pd.DataFrame:
   
    if not AV_API_KEY:
        raise RuntimeError("No detction of ALPHAVANTAGE_API_KEY. Please export ALPHAVANTAGE_API_KEY='your key'")

    params = {
        "function": function_name,
        "symbol": symbol,
        "apikey": AV_API_KEY,
    }

    r = requests.get(BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    reports = data.get("annualReports", [])
    df = pd.DataFrame(reports)

    if "fiscalDateEnding" in df.columns:
        df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"], errors="coerce")
        df = df[df["fiscalDateEnding"].dt.year.between(start_year, end_year)]
        df = df.sort_values("fiscalDateEnding", ascending=True)

    return df

def save_outputs(df: pd.DataFrame, prefix: str):
    df.to_csv(f"{prefix}.csv", index=False, encoding="utf-8-sig")
    df.to_excel(f"{prefix}.xlsx", index=False)

def run(symbol="BP"):
    bs = fetch_alpha_vantage("BALANCE_SHEET", symbol)
    is_ = fetch_alpha_vantage("INCOME_STATEMENT", symbol)
    cf = fetch_alpha_vantage("CASH_FLOW", symbol)

    save_outputs(bs, f"{symbol}_balance_sheet_2020_2024")
    save_outputs(is_, f"{symbol}_income_statement_2020_2024")
    save_outputs(cf, f"{symbol}_cash_flow_2020_2024")

    print("Finished：CSV/XLSX files have been exported.")

if __name__ == "__main__":
    run(symbol="BP")


# In[3]:


AV_API_KEY = "Paste your key here"
BASE = "https://www.alphavantage.co/query"

def fetch_raw(function_name: str, symbol: str):
    print(f"\n=== Initiate Request: {function_name} / {symbol} ===")
    params = {"function": function_name, "symbol": symbol, "apikey": AV_API_KEY}

    try:
        r = requests.get(BASE, params=params, timeout=30)
        print("HTTP Status Codes:", r.status_code)
        r.raise_for_status()
        text_preview = r.text[:300].replace("\n", " ")
        print("Response Preview (First 300 Characters):", text_preview)

        data = r.json()
        print("Return to top-level keys:", list(data.keys())[:10])
        return data
    except Exception as e:
        print("Request/Parsing Failed:", repr(e))
        return None

def to_annual_df(data):
    if not data:
        return None
    if "Error Message" in data:
        print("Interface error(Error Message):", data["Error Message"])
        return None
    if "Note" in data:
        print("Trigger Frequency Limit(Note):", data["Note"])
        return None

    reports = data.get("annualReports")
    if not reports:
        print("The annualReports field is either missing or empty.")
        return None

    df = pd.DataFrame(reports)
    if "fiscalDateEnding" in df.columns:
        df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"], errors="coerce")
        df = df[df["fiscalDateEnding"].dt.year.between(2020, 2024)].sort_values("fiscalDateEnding")
    return df

symbol = "BP"

data_bs = fetch_raw("BALANCE_SHEET", symbol)
time.sleep(15)

data_is = fetch_raw("INCOME_STATEMENT", symbol)
time.sleep(15)

data_cf = fetch_raw("CASH_FLOW", symbol)

bs = to_annual_df(data_bs)
is_ = to_annual_df(data_is)
cf = to_annual_df(data_cf)

print("\n=== Begin displaying the DataFrame ===")
print("Balance Sheet df:", None if bs is None else bs.shape)
print("Income Statement df:", None if is_ is None else is_.shape)
print("Cash Flow Statement df:", None if cf is None else cf.shape)

if bs is not None:
    print("\n📘 Balance Sheet")
    display(bs)
if is_ is not None:
    print("\n📗 Income Statement")
    display(is_)
if cf is not None:
    print("\n📙 Cash Flow Statement")
    display(cf)

print("\n=== End of script: If you can see this line, it means the cell's output mechanism is working fine. ===")


# In[4]:


def clean_financial_df(df, name="Unknown"):
 
    if df is None:
        print(f"❌ {name}: Input is None (Possible API rate limiting)")
        return None

    if not isinstance(df, pd.DataFrame):
        print(f"❌ {name}: The input is not a DataFrame.")
        return None

    if df.empty:
        print(f"❌ {name}: The DataFrame is empty.")
        return None

    df = df.copy()

    if "fiscalDateEnding" not in df.columns:
        print(f"❌ {name}: The fiscalDateEnding column is missing.")
        return None

    df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"], errors="coerce")
    df = df.dropna(subset=["fiscalDateEnding"])
    df = df.set_index("fiscalDateEnding").sort_index()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"✅ {name}: Cleaning finished，shape = {df.shape}")
    return df


# In[5]:


bs_clean = clean_financial_df(bs, "Balance Sheet")
is_clean = clean_financial_df(is_, "Income Statement")
cf_clean = clean_financial_df(cf, "Cash Flow")


# In[6]:


def validate_financials_final(bs, is_, cf, start_year=2020, end_year=2024):
    issues = []

    if bs is None:
        issues.append("[Existence] Balance Sheet not available ")
    if is_ is None:
        issues.append("[Existence] Income Statement not available ")
    if cf is None:
        issues.append("[Existence] Cash Flow not available ")
    if issues:
        return issues

    expected_years = set(range(start_year, end_year + 1))
    for df, name in [(bs, "Balance Sheet"), (is_, "Income Statement"), (cf, "Cash Flow")]:
        years = set(df.index.year)
        missing = expected_years - years
        if missing:
            issues.append(f"[Completeness] {name} Missing years: {sorted(missing)}")

    required_fields = {
        "Cash Flow": ["operatingCashflow", "capitalExpenditures"],
        "Income Statement": ["netIncome"],
        "Balance Sheet": ["totalAssets", "totalLiabilities"]
    }
    for name, fields in required_fields.items():
        df = {"Cash Flow": cf, "Income Statement": is_, "Balance Sheet": bs}[name]
        for field in fields:
            if field not in df.columns:
                issues.append(f"[Readiness] {name} Missing field: {field}")

    if "totalAssets" in bs.columns and "totalLiabilities" in bs.columns:
        bs["impliedEquity"] = bs["totalAssets"] - bs["totalLiabilities"]
    else:
        issues.append("[Accounting] The absence of totalAssets or totalLiabilities prevents the generation of impliedEquity.")

    shares_candidates = [
        "weightedAverageShsOutDiluted",
        "weightedAverageShsOut",
        "commonStockSharesOutstanding"
    ]
    shares_ok = any(field in is_.columns for field in shares_candidates) or any(field in bs.columns for field in shares_candidates)
    if not shares_ok:
        issues.append("[Shares] Shares Outstanding Missing: Unable to calculate EPS/P/E")

    return issues


# In[7]:


issues = validate_financials_final(bs_clean, is_clean, cf_clean)

if not issues:
    print("✅ Data validation passed: Proceed with DCF/Multiple analysis.")
else:
    print("⚠️ Data validation identified issues：")
    for i in issues:
        print("-", i)


# In[8]:


pd.set_option("display.float_format", "{:.6f}".format)

def compute_ratios(bs, is_, cf):
   
    def safe_get(df, col):
        return df[col] if col in df.columns else None


    net_income = safe_get(is_, "netIncome")
    total_assets = safe_get(bs, "totalAssets")
    equity = safe_get(bs, "impliedEquity")  
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

    total_liabilities = safe_get(bs, "totalLiabilities")
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

    inventory = safe_get(bs, "inventory")
    cogs = safe_get(is_, "costofGoodsAndServicesSold")
    if inventory is not None and cogs is not None:
        ratios["Inventory_Turnover"] = cogs / inventory

    op_cf = safe_get(cf, "operatingCashflow")
    capex = safe_get(cf, "capitalExpenditures")
    if op_cf is not None and capex is not None:
        ratios["FCF"] = op_cf - capex
        ratios["FCF_Margin"] = ratios["FCF"] / revenue

    return ratios

ratio_df = compute_ratios(bs_clean, is_clean, cf_clean)

ratio_df.round(6)


# In[9]:


def compute_historical_fcf(cf):
    cf = cf.copy()
    cf["operatingCashflow"] = pd.to_numeric(cf["operatingCashflow"], errors="coerce")
    cf["capitalExpenditures"] = pd.to_numeric(cf["capitalExpenditures"], errors="coerce")
    cf["FCF"] = cf["operatingCashflow"] - cf["capitalExpenditures"]
    return cf[["FCF"]]

hist_fcf = compute_historical_fcf(cf_clean)
hist_fcf


# In[10]:


def forecast_fcf_4yr_cagr(hist_fcf, years=5):
    hist_fcf = hist_fcf.dropna()

    recent = hist_fcf.tail(4)

    if len(recent) < 2:
        last_value = hist_fcf["FCF"].iloc[-1]
        future_index = pd.date_range(
            start=hist_fcf.index[-1] + pd.DateOffset(years=1),
            periods=years,
            freq="YE"
        )
        return pd.Series([last_value]*years, index=future_index, name="FCF"), 0.0

    start = recent["FCF"].iloc[0]
    end = recent["FCF"].iloc[-1]
    n = len(recent) - 1

    if start <= 0 or end <= 0:
        cagr = 0.0
    else:
        cagr = (end / start) ** (1/n) - 1

    last_value = recent["FCF"].iloc[-1]
    forecast = []
    for i in range(1, years + 1):
        last_value = last_value * (1 + cagr)
        forecast.append(last_value)

    future_index = pd.date_range(
        start=hist_fcf.index[-1] + pd.DateOffset(years=1),
        periods=years,
        freq="YE"
    )
    fcf_forecast = pd.Series(forecast, index=future_index, name="FCF")
    return fcf_forecast, cagr

fcf_forecast, cagr = forecast_fcf_4yr_cagr(hist_fcf, years=5)

print("CAGR over the past 4 years:", cagr)
fcf_forecast


# In[11]:


API_KEY = "Your Alpha Vantage API Key"

def fetch_overview(symbol: str) -> dict:
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": API_KEY
    }
    r = requests.get(url, params=params)
    data = r.json()
    return data


# In[12]:


overview = fetch_overview("BP")
overview


# In[13]:


def estimate_wacc(bs, is_, overview, risk_free_rate=0.04, market_risk_premium=0.06):
   
    if "MarketCapitalization" not in overview:
        raise ValueError("Unable to retrieve market cap. Please check the Overview interface.")
    market_cap = float(overview["MarketCapitalization"])

    if "totalDebt" in bs.columns:
        total_debt = float(bs["totalDebt"].iloc[-1])
    else:
        lt = float(bs["longTermDebt"].iloc[-1]) if "longTermDebt" in bs.columns else 0
        st = float(bs["shortTermDebt"].iloc[-1]) if "shortTermDebt" in bs.columns else 0
        total_debt = lt + st

    cash = float(bs["cashAndCashEquivalentsAtCarryingValue"].iloc[-1])
    net_debt = total_debt - cash

    ev = market_cap + net_debt
    w_e = market_cap / ev
    w_d = net_debt / ev

    beta_raw = float(overview.get("Beta", np.nan))

    beta_min, beta_max = 0.6, 1.2

    if np.isnan(beta_raw) or beta_raw < beta_min or beta_raw > beta_max:
        beta = 0.9  
        beta_used = "Industry median (0.9) replacement"
    else:
        beta = beta_raw
        beta_used = f"API beta({beta_raw})"

    cost_of_equity = risk_free_rate + beta * market_risk_premium

    interest_expense = abs(float(is_["interestExpense"].iloc[-1])) if "interestExpense" in is_.columns else 0
    cost_of_debt = interest_expense / total_debt if total_debt != 0 else 0.05

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
        "beta_raw": beta_raw,
        "beta": beta,
        "beta_used": beta_used,
        "cost_of_equity": cost_of_equity,
        "cost_of_debt": cost_of_debt,
        "tax_rate": tax_rate,
        "wacc": wacc
    }

wacc_info = estimate_wacc(bs_clean, is_clean, overview)
wacc_info


# In[14]:


forecast_years = 5            
terminal_growth_rate = 0.02   
wacc = wacc_info["wacc"]      


# In[20]:


future_fcf = fcf_forecast

future_fcf


# In[21]:


def terminal_value(last_fcf, wacc, terminal_growth):
    return last_fcf * (1 + terminal_growth) / (wacc - terminal_growth)

tv = terminal_value(future_fcf.iloc[-1], wacc, terminal_growth_rate)

tv_pv = tv / ((1 + wacc) ** forecast_years)
tv, tv_pv


# In[22]:


def discount_cashflows(cashflows, discount_rate):
    years = range(1, len(cashflows) + 1)
    discount_factors = [(1 + discount_rate) ** t for t in years]
    discounted = cashflows.values / discount_factors
    return pd.Series(discounted, index=cashflows.index)

discounted_fcf = discount_cashflows(future_fcf, wacc)
discounted_fcf


# In[23]:


ev_dcf = discounted_fcf.sum() + tv_pv

net_debt = wacc_info["net_debt"]
equity_value = ev_dcf - net_debt

shares_outstanding = float(overview.get("SharesOutstanding", 1))
price_per_share = equity_value / shares_outstanding

ev_dcf, equity_value, price_per_share


# In[24]:


def dcf_vs_market(ticker_symbol, intrinsic_price, 
                  start_date="2024-12-30", 
                  end_date="2025-01-05",
                  threshold=0.15):

    ticker = yf.Ticker("BP")
    price_series = ticker.history(start=start_date, end=end_date)["Close"]

    if price_series.empty:
        raise ValueError("No market price data retrieved. Please check date range.")

    market_price = float(price_series.iloc[0])

    upside = intrinsic_price / market_price - 1

    if upside > threshold:
        recommendation = "BUY"
    elif upside < -threshold:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    summary = pd.DataFrame({
        "Metric": [
            "DCF Intrinsic Price (USD)",
            "Market Price (End 2024, USD)",
            "Upside / Downside (%)",
            "Investment Recommendation"
        ],
        "Value": [
            round(intrinsic_price, 2),
            round(market_price, 2),
            round(upside * 100, 2),
            recommendation
        ]
    })

    return summary, upside, recommendation


# In[25]:


dcf_summary, upside, recommendation = dcf_vs_market(
    ticker_symbol="BP",
    intrinsic_price=price_per_share
)

dcf_summary


# In[26]:


peers = {
    "BP": "BP",
    "Shell": "SHEL",
    "ExxonMobil": "XOM"
}


# In[27]:


def fetch_multiples_extended(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    market_cap = info.get("marketCap", None)
    net_income = info.get("netIncomeToCommon", None)
    operating_cf = info.get("operatingCashflow", None)
    ebitda = info.get("ebitda", None)

    total_debt = info.get("totalDebt", None)
    cash = info.get("totalCash", None)

    if market_cap is not None and total_debt is not None and cash is not None:
        ev = market_cap + total_debt - cash
    else:
        ev = None

    pe = market_cap / net_income if market_cap and net_income and net_income > 0 else np.nan
    ev_ebitda = ev / ebitda if ev and ebitda and ebitda > 0 else np.nan
    ev_ocf = ev / operating_cf if ev and operating_cf and operating_cf > 0 else np.nan

    return {
        "Market Cap": market_cap,
        "EV": ev,
        "Net Income (TTM)": net_income,
        "EBITDA (TTM)": ebitda,
        "Operating CF (TTM)": operating_cf,
        "P/E": pe,
        "EV/EBITDA": ev_ebitda,
        "EV/Operating CF": ev_ocf
    }


# In[28]:


rows = []

for name, ticker in peers.items():
    data = fetch_multiples_extended(ticker)
    data["Company"] = name
    rows.append(data)

df_multiples = pd.DataFrame(rows).set_index("Company")
df_multiples


# In[29]:


multiples_table = df_multiples[[
    "EV",
    "EBITDA (TTM)",
    "EV/EBITDA"
]]

multiples_table


# In[30]:


multiples_core = df_multiples[[
    "P/E",
    "EV/EBITDA",
    "EV/Operating CF"
]]

multiples_core


# In[31]:


summary = multiples_core.agg(["mean", "median", "min", "max"])
summary


# In[32]:


bp_data = df_multiples.loc["BP"]

bp_net_income = bp_data["Net Income (TTM)"]
bp_ebitda = bp_data["EBITDA (TTM)"]
bp_ocf = bp_data["Operating CF (TTM)"]
bp_net_debt = bp_data["EV"] - bp_data["Market Cap"]

bp_data


# In[33]:


industry_median = multiples_core.median()
industry_median


# In[34]:


valuation_rows = []

if not np.isnan(industry_median["P/E"]) and bp_net_income > 0:
    implied_equity_pe = industry_median["P/E"] * bp_net_income
    valuation_rows.append({
        "Method": "P/E",
        "Implied Enterprise Value": np.nan,
        "Implied Equity Value": implied_equity_pe
    })

if not np.isnan(industry_median["EV/EBITDA"]) and bp_ebitda > 0:
    implied_ev_ebitda = industry_median["EV/EBITDA"] * bp_ebitda
    implied_equity_ebitda = implied_ev_ebitda - bp_net_debt
    valuation_rows.append({
        "Method": "EV/EBITDA",
        "Implied Enterprise Value": implied_ev_ebitda,
        "Implied Equity Value": implied_equity_ebitda
    })

if not np.isnan(industry_median["EV/Operating CF"]) and bp_ocf > 0:
    implied_ev_ocf = industry_median["EV/Operating CF"] * bp_ocf
    implied_equity_ocf = implied_ev_ocf - bp_net_debt
    valuation_rows.append({
        "Method": "EV/Operating CF",
        "Implied Enterprise Value": implied_ev_ocf,
        "Implied Equity Value": implied_equity_ocf
    })

df_implied_value = pd.DataFrame(valuation_rows)
df_implied_value


# In[35]:


shares_outstanding = yf.Ticker("BP").info.get("sharesOutstanding")

df_implied_value["Implied Share Price"] = (
    df_implied_value["Implied Equity Value"] / shares_outstanding
)

df_implied_value


# In[36]:


dcf_share_price = price_per_share


# In[37]:


valuation_summary = df_implied_value[[
    "Method",
    "Implied Equity Value",
    "Implied Share Price"
]].copy()

valuation_summary = pd.concat([
    valuation_summary,
    pd.DataFrame([{
        "Method": "DCF",
        "Implied Equity Value": equity_value,
        "Implied Share Price": dcf_share_price
    }])
], ignore_index=True)

valuation_summary


# In[38]:


valuation_summary_formatted = valuation_summary.copy()

valuation_summary_formatted["Implied Equity Value (USD bn)"] = (
    valuation_summary_formatted["Implied Equity Value"] / 1e9
).round(2)

valuation_summary_formatted["Implied Share Price (USD)"] = (
    valuation_summary_formatted["Implied Share Price"]
).round(2)

valuation_summary_formatted = valuation_summary_formatted[[
    "Method",
    "Implied Equity Value (USD bn)",
    "Implied Share Price (USD)"
]]

valuation_summary_formatted


# In[39]:


bp_info = yf.Ticker("BP").info

def esg_proxy_assessment(info):
    result = {}
  
    if info.get("sector") == "Energy":
        result["Environmental Risk"] = "High"
    else:
        result["Environmental Risk"] = "Medium"
 
    employees = info.get("fullTimeEmployees", 0)
    if employees > 50000:
        result["Social Risk"] = "Medium"
    else:
        result["Social Risk"] = "Low"
    
    roe = info.get("returnOnEquity", 0)
    debt_to_equity = info.get("debtToEquity", 0)
    
    if roe < 0.05 or debt_to_equity > 150:
        result["Governance Risk"] = "Weak"
    else:
        result["Governance Risk"] = "Acceptable"
    
    if result["Environmental Risk"] == "High":
        result["Overall ESG Risk"] = "Elevated"
    else:
        result["Overall ESG Risk"] = "Moderate"
    
    return result

bp_esg_proxy = esg_proxy_assessment(bp_info)
bp_esg_proxy


# In[41]:


fundamental_risks = {}

if "Leverage" in ratio_df.columns:
    avg_leverage = ratio_df["Leverage"].mean()
    fundamental_risks["Leverage"] = (
        "High" if avg_leverage > 2.0 else
        "Medium" if avg_leverage > 1.0 else
        "Low"
    )
else:
    fundamental_risks["Leverage"] = "Unknown"


if "Current_Ratio" in ratio_df.columns:
    avg_cr = ratio_df["Current_Ratio"].mean()
    fundamental_risks["Liquidity"] = (
        "High" if avg_cr < 1.0 else
        "Medium" if avg_cr < 1.5 else
        "Low"
    )
else:
    fundamental_risks["Liquidity"] = "Unknown"

fcf_vol = hist_fcf["FCF"].pct_change().std()
fundamental_risks["Cash Flow Stability"] = (
    "High" if fcf_vol > 0.4 else
    "Medium" if fcf_vol > 0.2 else
    "Low"
)

if "netIncome" in is_clean.columns:
    ni_vol = is_clean["netIncome"].pct_change().std()
    fundamental_risks["Earnings Volatility"] = (
        "High" if ni_vol > 0.4 else
        "Medium" if ni_vol > 0.2 else
        "Low"
    )
else:
    fundamental_risks["Earnings Volatility"] = "Unknown"

fundamental_risks


# In[42]:


risk_dashboard = pd.DataFrame({
    "Risk Dimension": [
        "Environmental (ESG)",
        "Social (ESG)",
        "Governance (ESG)",
        "Leverage",
        "Liquidity",
        "Cash Flow Stability",
        "Earnings Volatility"
    ],
    "Risk Level": [
        bp_esg_proxy["Environmental Risk"],
        bp_esg_proxy["Social Risk"],
        bp_esg_proxy["Governance Risk"],
        fundamental_risks["Leverage"],
        fundamental_risks["Liquidity"],
        fundamental_risks["Cash Flow Stability"],
        fundamental_risks["Earnings Volatility"]
    ]
})

risk_dashboard


# In[208]:


SOURCES = {
    "price": "Yahoo Finance (yfinance) – market price / trading multiples",
    "multiples": "Yahoo Finance (yfinance) – peer trading multiples",
    "statements": "Alpha Vantage – annual financial statements (Income/BS/CF)",
    "calc": "Author calculations based on stated data sources"
}


# In[209]:


memo_input = {
    "company": {
        "name": "BP p.l.c. (ADR)",
        "ticker": "BP",
        "analysis_date": "End-2024",
        "citation": SOURCES["statements"]
    },

    "valuation_summary": {
    "market_price_usd": 27.44,
    "implied_valuation": valuation_summary.to_dict(orient="records"),
    "primary_anchor": "DCF",
    "secondary_cross_check": [
        "EV/EBITDA",
        "EV/Operating CF",
        "P/E"
    ],
    "final_recommendation": recommendation,

    "citation": f"{SOURCES['price']} + {SOURCES['multiples']} + {SOURCES['calc']}"
        },


    "financial_performance": {
        "profitability": {
            "roe": ratio_df["ROE"].dropna().to_dict(),
            "net_margin": ratio_df["Net_Margin"].dropna().to_dict(),
            "ebitda_margin": ratio_df["EBITDA_Margin"].dropna().to_dict(),
            "citation": f"{SOURCES['statements']} + {SOURCES['calc']}"
        },
        "leverage": {
            "debt_to_equity": ratio_df["Debt_to_Equity"].dropna().to_dict() if "Debt_to_Equity" in ratio_df.columns else None,
            "interest_coverage": ratio_df["Interest_Coverage"].dropna().to_dict() if "Interest_Coverage" in ratio_df.columns else None,
            "citation": f"{SOURCES['statements']} + {SOURCES['calc']}"
        }
    },

    "cash_flow": {
        "historical_fcf_usd": hist_fcf["FCF"].dropna().to_dict(),
        "citation": f"{SOURCES['statements']} + {SOURCES['calc']}"
    },

    "peer_comparison": {
    "rule": (
        "Peer comparison MUST be strictly multiples-based. "
        "Use EV/EBITDA as primary; P/E and EV/Operating CF as secondary. "
        "Use summary stats (mean/median/min/max) across selected peers."
    ),
    "explicit_findings": {
        "pe_positioning": "BP P/E is above both peer mean and median",
        "ev_ebitda_positioning": "BP EV/EBITDA is broadly in line with peer median",
        "ev_ocf_positioning": "BP EV/Operating CF is broadly in line with peers"
    },
    "multiples_summary_table": peer_multiples_summary_records,
    "citation": SOURCES["multiples"]
},


    "risk_overview": {
        "table": risk_dashboard.to_dict(orient="records"),
        "citation": f"{SOURCES['statements']} + {SOURCES['calc']}"
    },

    "data_sources": [
        SOURCES["price"],
        SOURCES["multiples"],
        SOURCES["statements"],
        SOURCES["calc"]
    ],

    "guardrails": [
        "Do NOT invent WACC, terminal growth, or any assumptions not provided in memo_input.",
        "If WACC/g are not provided, omit them from the memo entirely.",
        "Do NOT display EV and EBITDA levels in peer comparison; only show multiples summary table.",
        "If a financial metric is unavailable, do NOT explicitly state 'not available'; instead, omit the metric and discuss leverage or financial risk using available indicators only.",
        "Do NOT soften or normalize extreme values.",
        "If a multiple is materially above or below peer mean/median, state this explicitly."
    ]
}

memo_input


# In[214]:


import json

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"   

def ollama_chat(prompt: str, system: str = "", temperature: float = 0.0) -> str:
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": (
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": prompt}]
        ),
        "options": {
            "temperature": temperature,
        },
        "stream": False
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


# In[215]:


def build_ic_memo_prompt(memo_input: dict, analyst_view: str = "Neutral") -> str:
    """
    IC memo prompt with controlled visual hierarchy:
    - Title > Section headers > Subsection headers
    - Tables included
    - Consistent font sizing in Markdown render
    """
    return f"""
You are preparing an INTERNAL Investment Committee (IC) memo for an equity investment.

Analyst stance (tone only, not to change the conclusion): {analyst_view}

========================
STRICT FORMATTING RULES (NON-NEGOTIABLE)
========================
Use Markdown ONLY. Follow this hierarchy exactly:

1) '# ' → Document title (use ONCE only)
2) '## ' → Main section headers (ALL same level and size)
3) '### ' → Subsection headers (ALL same level and size)
4) Bold text (**) → Allowed ONLY for key figures or conclusions, NEVER for headers

Additional rules:
- Do NOT use '####' or deeper levels.
- Do NOT use bold text as a substitute for headers.
- Do NOT use HTML, colors, emojis, or special symbols.
- Tables MUST be standard Markdown tables (pipes and dashes only).
- Tables should appear immediately after the subsection they relate to.
- Do NOT repeat headings.

========================
CONTENT RULES
========================
1) Use ONLY the data provided in DATA INPUT.
2) If a financial metric is unavailable, do NOT explicitly state 'not available'; instead, omit the metric and discuss leverage or financial risk using available indicators only.
3) Discounted Cash Flow (DCF) is the PRIMARY valuation anchor.
   - Multiples (P/E, EV/EBITDA, EV/Operating Cash Flow) are SECONDARY cross-checks.
   - Multiples must NOT override the DCF-based conclusion.
   - In the "Multiples Cross-Check" section, you MUST use the *multiples implied share prices*
  (e.g., rows like P/E, EV/EBITDA, EV/Operating CF with "Implied Share Price").
   - Do NOT discuss raw multiple levels (e.g., "P/E is high/low", "EV/EBITDA is above peers").
  Raw multiples may ONLY appear in the Peer Comparison section if explicitly provided there.
   - Multiples cross-check MUST compare implied prices vs DCF intrinsic price and market price.
4) The FINAL recommendation MUST match the quantitative recommendation
   provided in DATA INPUT.
5) Target length: approximately 900–1200 words (1–2 A4 pages).
6) All numerical claims, tables, and comparisons MUST be explicitly attributable
   to a data source listed in the Data Sources section.
7）The model MUST NOT introduce any valuation assumptions (including WACC, terminal growth rate, forecast horizon)
  unless they are explicitly present in DATA INPUT. If such assumptions are not provided, DO NOT mention them.
8）When describing trends:
- Use the FIRST and LAST available years only.
- Do NOT substitute peak or median values.
- Do NOT ignore negative observations.

========================
NUMERICAL ACCURACY RULES (MANDATORY)
========================
- When describing trends (increase/decrease/improve/deteriorate), you MUST verify against the actual numeric series.
- If a metric moves from negative to positive, you MUST describe it as "improved" (or "recovered"), not "declined".
- If metrics worsen, explicitly state deterioration.
- Do NOT soften or reframe negative trends.


========================
PEER COMPARISON INTERPRETATION (MANDATORY)
========================
- In the Peer Comparison section, you MUST include 2–4 explanatory sentences BEFORE or AFTER the table.
- You MUST interpret BP's position relative to the peer range/median using the provided summary stats.
- Use at least one sentence anchored on EV/EBITDA (primary multiple). If available, mention P/E or EV/Operating CF briefly.
- Do NOT include EV or EBITDA absolute levels unless explicitly provided.

========================
REQUIRED STRUCTURE (FOLLOW EXACTLY)
========================

# BP p.l.c. (ADR) – Equity Investment Memorandum

## Investment Thesis
- 3–5 concise bullet points.
- Explicitly state DCF intrinsic value, market price, and upside/downside.
- Make the implied recommendation unmistakably clear.

## Company & Business Overview
- Describe the business model, integration, and geographic scope.
- Keep this section factual and concise.

## Financial Performance & Key Ratios

### Profitability and Returns

| Year | ROE (%) | Net Margin (%) | EBITDA Margin (%) |
|------|---------|----------------|-------------------|
| Use provided data only |

## Valuation

### Discounted Cash Flow (Primary)
- Present intrinsic value, market price, and upside/downside clearly.
- Explain why DCF is the anchor valuation.

### Multiples Cross-Check
- Present P/E, EV/EBITDA, and EV/Operating Cash Flow implications.
- Explicitly explain any divergence versus DCF without overturning it.

## Peer Comparison
- Compare BP against peers strictly using valuation multiples.
- Include a peer multiples table if provided.
- If peer metrics are missing, explicitly state "Not available".

## Management Quality & Capital Allocation
- Assess management discipline, capital allocation, and balance sheet strategy.
- Avoid speculation; base assessment on provided data.

## Catalysts
- List 3–6 plausible, non-speculative catalysts.
- Keep each catalyst to one concise line.

## Risks
- Cover fundamental, financial, and ESG risks.
- Tie risks directly to margins, leverage, and cash flow volatility.

## Investment Recommendation
- Clearly restate the final recommendation (BUY / HOLD / SELL).
- Summarize:
  - 3 key reasons supporting the recommendation
  - 3 key risks that could challenge the thesis

## Data Sources
- Explicitly list and cite all external data providers used.
- All financial statements, ratios, prices, and valuation multiples must be traceable
  to these sources.
- "Author calculations" MUST be described as derived from the listed external sources, not as an independent data source.

========================
DATA INPUT (TREAT AS GROUND TRUTH)
========================
{memo_input}
""".strip()


# In[216]:


system_msg = "You are a disciplined senior buy-side equity analyst. Write like an investment professional."

ic_prompt = build_ic_memo_prompt(memo_input=memo_input)

ic_memo_md = ollama_chat(
    prompt=ic_prompt,
    system=system_msg,
    temperature=0.0
)


# In[217]:


from IPython.display import display, Markdown

display(Markdown(ic_memo_md))


# In[218]:


with open("IC_memo.md", "w", encoding="utf-8") as f:
    f.write(ic_memo_md)

from markdown import markdown

html = markdown(ic_memo_md, extensions=["tables", "fenced_code"])

with open("IC_memo.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Saved: IC_memo.md and IC_memo.html")

