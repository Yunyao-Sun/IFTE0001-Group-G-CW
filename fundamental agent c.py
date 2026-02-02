#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
import time
from IPython.display import display

years = ["2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"]

urls = {
    "income": "https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol=BP&apikey=your key",
    "balance": "https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol=BP&apikey=your key",
    "cashflow": "https://www.alphavantage.co/query?function=CASH_FLOW&symbol=BP&apikey=your key"
}

results = {}

for name, url in urls.items():
    data = requests.get(url).json()
    time.sleep(2)

    if "annualReports" in data:
        df = pd.DataFrame(data["annualReports"])

        df = df[df["fiscalDateEnding"].isin(years)].sort_values("fiscalDateEnding")


        cols = df.columns.tolist()

        df = df[cols]  
        df.rename(columns={cols[0]: "year"}, inplace=True)


        df["year"] = df["year"].str[:4]

        results[name] = df
    else:
        print(f"{name} error:", data)


income = results.get("income")
balance = results.get("balance")
cashflow = results.get("cashflow")

display(income)
display(balance)
display(cashflow)







# In[2]:


income_clean = income.copy()
for col in income_clean.columns[1:]:
    income_clean[col] = pd.to_numeric(income_clean[col], errors='coerce')
income_clean.fillna(0, inplace=True)
income_clean.reset_index(drop=True, inplace=True)
# Balance
balance_clean = balance.copy()
for col in balance_clean.columns[1:]:
    balance_clean[col] = pd.to_numeric(balance_clean[col], errors='coerce')
balance_clean.fillna(0, inplace=True)
balance_clean.reset_index(drop=True, inplace=True)

# Cashflow
cashflow_clean = cashflow.copy()
for col in cashflow_clean.columns[1:]:
    cashflow_clean[col] = pd.to_numeric(cashflow_clean[col], errors='coerce')
cashflow_clean.fillna(0, inplace=True)
cashflow_clean.reset_index(drop=True, inplace=True)









# In[3]:


cashflow_clean.columns


# In[4]:


cashflow_clean['FCF'] = cashflow_clean['operatingCashflow'] - cashflow_clean['capitalExpenditures']
fcf = cashflow_clean[["year","FCF"]]
fcf


# In[5]:


import pandas as pd
import numpy as np


fcf = pd.DataFrame({
    "year": [2020, 2021, 2022, 2023, 2024],
    "FCF": [-144000000, 12725000000, 28863000000, 17754000000, 12000000000]
})


fcf_positive = fcf[fcf["FCF"] > 0].copy()
first_positive_fcf = fcf_positive["FCF"].iloc[0]
last_positive_fcf = fcf_positive["FCF"].iloc[-1]
n_years = fcf_positive.shape[0] - 1


cagr = (last_positive_fcf / first_positive_fcf) ** (1 / n_years) - 1
print(f" CAGR: {cagr:.2%}")


forecast_years = 5
last_year = fcf["year"].iloc[-1]
fcf_forecast = [last_positive_fcf * (1 + cagr) ** (i+1) for i in range(forecast_years)]
forecast_years_list = [last_year + i + 1 for i in range(forecast_years)]

fcf_forecast_df = pd.DataFrame({
    "year": forecast_years_list,
    "FCF_Forecast": fcf_forecast
})

from IPython.display import display
display(fcf_forecast_df)


# In[6]:


import pandas as pd
wacc=0.1
g=0.02




discount_factors = [(1 / (1 + wacc) ** i) for i in range(1, forecast_years + 1)]


dcf_values = [fcf_forecast[i] * discount_factors[i] for i in range(forecast_years)]


dcf_df = pd.DataFrame({
    "year": forecast_years_list,
    "DCF": dcf_values
})

from IPython.display import display
display(dcf_df)


# In[7]:


fcf_terminal = fcf_forecast[-1] * (1 + g)

terminal_value = fcf_terminal / (wacc - g)
pv_terminal = terminal_value / (1 + wacc) ** forecast_years
enterprise_value = sum(dcf_values) + pv_terminal
print(f"BP enterprice value: {enterprise_value:.2f} USD")


# In[8]:


balance_clean.columns
netdebt_2024 = (
    balance_clean.loc[balance_clean["year"] == "2024", "shortTermDebt"].values[0]
    + balance_clean.loc[balance_clean["year"] == "2024", "longTermDebt"].values[0]
    - balance_clean.loc[balance_clean["year"] == "2024", "cashAndCashEquivalentsAtCarryingValue"].values[0]
)
shares = balance_clean.loc[balance_clean["year"] == "2024", "commonStock"].values[0]
equity_value=enterprise_value-netdebt_2024
intrinsic_value_per_share = equity_value / shares
print(f"BP intrinsic_value_per_share: {intrinsic_value_per_share :.2f} USD")


# In[9]:


import yfinance as yf

bp = yf.Ticker("BP")


price_2024 = bp.history(start="2024-01-01", end="2025-01-01")


close_2024 = price_2024["Close"].iloc[-1]

print(f"BP 2024 year-end close price: {close_2024:.2f} USD")



# In[10]:


valuation_diff = intrinsic_value_per_share - close_2024
status = "Undervalued" if valuation_diff > 0 else "Overvalued"

print(f"intrinsic value: {intrinsic_value_per_share:.2f} USD")
print(f"stock price: {close_2024:.2f} USD")
print(f"gap: {valuation_diff:.2f} USD → {status}")


# In[11]:


income_clean.columns


# In[12]:


Revenue = (income_clean["totalRevenue"])
Net_Income = (income_clean["netIncome"])  
Total_Assets = (balance_clean["totalAssets"])

EBITDA = (income_clean["ebitda"])  



Net_Profit_Margin = Net_Income / Revenue       
ROA = Net_Income / Total_Assets                
EBITDA_Margin = EBITDA / Revenue               

profitability = pd.DataFrame({
    "year": income_clean["year"],       
    "Net Profit Margin": Net_Profit_Margin,
    "ROA": ROA,
    "EBITDA_Margin": EBITDA_Margin
})

from IPython.display import display
display(profitability)



# In[13]:


import matplotlib.pyplot as plt
import pandas as pd

# 确保 year 是数值并排序
profitability["year"] = pd.to_numeric(profitability["year"], errors="coerce")
profitability = profitability.sort_values("year")

# 转成百分比（更符合财务展示习惯）
profitability_plot = profitability.copy()
profitability_plot["Net Profit Margin"] *= 100
profitability_plot["ROA"] *= 100
profitability_plot["EBITDA_Margin"] *= 100
plt.figure()
plt.plot(
    profitability_plot["year"],
    profitability_plot["Net Profit Margin"],
    marker="o",
    label="Net Profit Margin"
)
plt.plot(
    profitability_plot["year"],
    profitability_plot["EBITDA_Margin"],
    marker="o",
    label="EBITDA Margin"
)

plt.title("Profitability Margins Trend (2020–2024)")
plt.xlabel("Year")
plt.ylabel("Percentage (%)")
plt.legend()
plt.grid(True)
plt.show()


# In[14]:


plt.figure()
plt.plot(
    profitability_plot["year"],
    profitability_plot["ROA"],
    marker="o"
)

plt.title("Return on Assets (ROA) Trend (2020–2024)")
plt.xlabel("Year")
plt.ylabel("Percentage (%)")
plt.grid(True)
plt.show()


# In[15]:


import pandas as pd


total_debt = pd.to_numeric(balance_clean["shortTermDebt"], errors="coerce") + \
             pd.to_numeric(balance_clean["longTermDebt"], errors="coerce")


total_assets = pd.to_numeric(balance_clean["totalAssets"], errors="coerce")


EBIT = pd.to_numeric(income_clean["ebit"], errors="coerce")
interest_expense = pd.to_numeric(income_clean["interestExpense"], errors="coerce")


EBITDA = pd.to_numeric(income_clean["ebitda"], errors="coerce")


Debt_Ratio = total_debt / total_assets
Interest_Coverage = EBIT / interest_expense
Debt_to_EBITDA = total_debt / EBITDA 


solvency = pd.DataFrame({
    "year": balance_clean["year"],  
    "Debt_Ratio": Debt_Ratio,
    "Interest_Coverage": Interest_Coverage,
    "Debt_to_EBITDA": Debt_to_EBITDA
})


solvency.replace([float('inf'), -float('inf')], pd.NA, inplace=True)

from IPython.display import display
display(solvency)


# In[16]:


revenue = (income_clean["totalRevenue"])
revenue_growth = revenue.pct_change()  


net_income = (income_clean["netIncome"])
net_income_growth = net_income.pct_change()


fcf=cashflow_clean['FCF']
fcf_growth = fcf.pct_change()



growth=pd.DataFrame({"Revenue Growth":revenue_growth,"Net Income Growth":net_income_growth,"FCF Growth":fcf_growth})
growth


# In[17]:


import pandas as pd
import matplotlib.pyplot as plt


growth['year'] = income_clean['year'] 


x = growth['year']

plt.figure(figsize=(10,6))


plt.plot(x, growth['Revenue Growth'], marker='o', label='Revenue Growth')
plt.plot(x, growth['Net Income Growth'], marker='o', label='Net Income Growth')
plt.plot(x, growth['FCF Growth'], marker='o', label='FCF Growth')


plt.title('Financial Growth Rates')
plt.xlabel('Year')
plt.ylabel('Growth Rate')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()


plt.show()


# In[18]:


import pandas as pd
import numpy as np


total_debt = pd.to_numeric(balance_clean["shortTermDebt"], errors="coerce") + \
             pd.to_numeric(balance_clean["longTermDebt"], errors="coerce")


cash = pd.to_numeric(balance_clean["cashAndCashEquivalentsAtCarryingValue"], errors="coerce")



market_cap = shares * close_2024


enterprise_value = market_cap + total_debt - cash

print("Enterprise Value:", enterprise_value)



# In[19]:


import pandas as pd
import numpy as np


net_income = pd.to_numeric(income_clean["netIncome"], errors="coerce")
EBITDA = pd.to_numeric(income_clean["ebitda"], errors="coerce")
FCF = pd.to_numeric(cashflow_clean["FCF"], errors="coerce")
total_equity = pd.to_numeric(balance_clean["totalShareholderEquity"], errors="coerce")


market_cap = shares * close_2024


PE_bp = market_cap / net_income             
PB_bp = market_cap / total_equity            
EV_EBITDA_bp = enterprise_value / EBITDA    
EV_FCF_bp = enterprise_value / FCF          


bp_multiples = pd.DataFrame({
    "year": income_clean["year"],
    "P/E": PE_bp,
    "P/B": PB_bp,
    "EV/EBITDA": EV_EBITDA_bp,
    "EV/FCF": EV_FCF_bp
})


bp_multiples.replace([np.inf, -np.inf], pd.NA, inplace=True)

from IPython.display import display
display(bp_multiples)


# In[20]:


import matplotlib.pyplot as plt


plt.figure(figsize=(10,6))

plt.plot(bp_multiples["year"], bp_multiples["P/E"], marker='o', label="P/E")
plt.plot(bp_multiples["year"], bp_multiples["P/B"], marker='o', label="P/B")
plt.plot(bp_multiples["year"], bp_multiples["EV/EBITDA"], marker='o', label="EV/EBITDA")
plt.plot(bp_multiples["year"], bp_multiples["EV/FCF"], marker='o', label="EV/FCF")

plt.title("Valuation Multiples Over Years")
plt.xlabel("Year")
plt.ylabel("Multiple")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)  
plt.tight_layout()
plt.show()


# In[21]:


pip install yfinance pandas


# In[22]:


import yfinance as yf

bp = yf.Ticker("SHEL")

income = bp.financials
income


# In[23]:


cashflow = bp.cashflow
cashflow


# In[24]:


balance = bp.balancesheet
balance


# In[25]:


income.head()
income.info()
income.index

income_clean1 = income.T
balance_clean1 = balance.T
cashflow_clean1 = cashflow.T



income_clean1.index = income_clean1.index.astype(str)
balance_clean1.index = balance_clean1.index.astype(str)
cashflow_clean1.index = cashflow_clean1.index.astype(str)


cashflow_clean1 = cashflow_clean1.fillna(0)


print("Income missing values:\n", income_clean1.isna().sum().sort_values(ascending=False))
print("Balance missing values:\n", balance_clean1.isna().sum().sort_values(ascending=False))
print("Cashflow missing values:\n", cashflow_clean1.isna().sum().sort_values(ascending=False))


# In[26]:


import yfinance as yf

bp = yf.Ticker("SHEL")


price_2024 = bp.history(start="2024-01-01", end="2025-01-01")


close_price = price_2024["Close"].iloc[-1]

print(f"SHEL 2024 year-end close price: {close_price:.2f} USD")



# In[27]:


shares_shell = (balance_clean1["Share Issued"].values[0])  
current_price = 35.38  

market_cap = shares_shell * close_price
print(f"Market Capitalization: {market_cap:.2f} USD")


# In[28]:


balance_clean1.columns
income_clean1.columns


# In[29]:


import pandas as pd
import numpy as np

total_equity = (balance_clean1["Stockholders Equity"].values[0])  
pb = market_cap / total_equity


total_debt_2024 = (balance_clean1["Total Debt"].values[0]) 
cash_2024 = (balance_clean1["Cash And Cash Equivalents"].values[0]) 

EV_2024 = market_cap + total_debt_2024 - cash_2024
net_income_shell = (income_clean1["Net Income"].values[0]) 
EBITDA = (income_clean1["EBITDA"].values[0]) 
pe_ratio_shell = market_cap / net_income_shell
FCF = (cashflow_clean1["Free Cash Flow"].values[0])
EV_EBITDA = EV_2024 / EBITDA    # EV/EBITDA
EV_FCF = EV_2024 / FCF
multiples = pd.DataFrame({
    "Multiple": ["P/E", "P/B", "EV/EBITDA", "EV/FCF"],
    "Value": [pe_ratio_shell, pb, EV_EBITDA, EV_FCF]
})


multiples["Value"] = pd.to_numeric(multiples["Value"], errors="coerce")

multiples["Value"] = multiples["Value"].replace([np.inf, -np.inf], pd.NA)



multiples["Value"] = multiples["Value"].round(2)

from IPython.display import display
display(multiples)



# In[30]:


net_income_bp2024=(income_clean["netIncome"].values[0]) 
equity_value_pe = pe_ratio_shell * net_income_bp2024
price_peer = total_equity / shares                             #equity_value_pe = pe_peer_2024 * net_income_2024  price_pe = equity_value_pe / shares_2024                                                            #price_pe = equity_value_pe / shares_2024
price_peer


# In[39]:


import pandas as pd
import numpy as np
from IPython.display import Markdown, display

def _to_pct(x, digits=1):
    if pd.isna(x):
        return "N/A"
    return f"{x*100:.{digits}f}%"

def _to_num(x, digits=2):
    if pd.isna(x):
        return "N/A"
    return f"{x:.{digits}f}"

def _trend_arrow(first, last):
    if pd.isna(first) or pd.isna(last):
        return "→"
    if last > first:
        return "↑"
    if last < first:
        return "↓"
    return "→"

def _safe_year_value(df, year, col, year_col="year"):
    sub = df[df[year_col] == year]
    if len(sub) == 0:
        return np.nan
    return pd.to_numeric(sub[col], errors="coerce").iloc[0]

def _range_str(df, col):
    s = pd.to_numeric(df[col], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return "N/A"
    return f"{s.min():.2f} to {s.max():.2f}"

def _flag_risks(profitability, solvency, growth, year=2024):
    flags = []

    # Interest coverage risk
    ic = _safe_year_value(solvency, year, "Interest_Coverage")
    if pd.notna(ic) and ic < 3:
        flags.append(f"⚠️ Interest coverage is weak in {year} (≈{ic:.2f}x).")

    # Leverage risk
    d2e = _safe_year_value(solvency, year, "Debt_to_EBITDA")
    if pd.notna(d2e) and d2e > 2.5:
        flags.append(f"⚠️ Leverage elevated in {year} (Debt/EBITDA ≈{d2e:.2f}x).")

    # Net margin extremely low / negative
    npm = _safe_year_value(profitability, year, "Net Profit Margin")
    if pd.notna(npm) and npm < 0.01:
        flags.append(f"⚠️ Net profit margin is very low in {year} ({npm*100:.2f}%).")

    # FCF growth volatility (use abs)
    if "FCF Growth" in growth.columns:
        gfcf = pd.to_numeric(growth["FCF Growth"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if gfcf.dropna().empty is False and (gfcf.dropna().abs().max() > 1.0):
            flags.append("⚠️ FCF growth is highly volatile across the sample (large swings YoY).")

    if not flags:
        flags.append("No major quantitative red flags triggered by the rule-based checks.")
    return flags

def llm_final_action(close_2024, intrinsic_per_share, peer_price,
                     interest_coverage_2024=np.nan,
                     net_profit_margin_2024=np.nan,
                     fcf_volatility_flag=False):

    dcf_gap = (intrinsic_per_share - close_2024) / close_2024
    peer_gap = (peer_price - close_2024) / close_2024

    risk_score = 0
    if pd.notna(interest_coverage_2024) and interest_coverage_2024 < 3:
        risk_score += 1
    if pd.notna(net_profit_margin_2024) and net_profit_margin_2024 < 0.01:
        risk_score += 1
    if fcf_volatility_flag:
        risk_score += 1

    # Decision logic
    if dcf_gap > 0.10 and risk_score <= 1:
        action = "BUY"
        confidence = "Medium"
        reason = "DCF implies meaningful upside and risk flags are limited."
    elif dcf_gap < -0.15 and risk_score >= 3:
        action = "SELL"
        confidence = "Medium"
        reason = "DCF implies meaningful downside and multiple risk flags are triggered."
    else:
        action = "HOLD"
        confidence = "Medium"
        reason = "Mixed intrinsic vs peer signals; cyclicality warrants a neutral stance."


    return action, confidence, reason, dcf_gap, peer_gap, risk_score


def llm_agent_output_md_rich(
    company: str,
    close_2024: float,
    intrinsic_per_share: float,
    peer_price: float,
    wacc: float,
    g: float,
    profitability: pd.DataFrame,
    solvency: pd.DataFrame,
    growth: pd.DataFrame,
    bp_multiples: pd.DataFrame,
    year_col: str = "year",
    year: int = 2024
):
    # ---- Ensure year type
    for df in [profitability, solvency, bp_multiples]:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype(int)

    y0 = int(profitability[year_col].min())
    yN = year

    # ---- Pull key metrics
    npm_0 = _safe_year_value(profitability, y0, "Net Profit Margin", year_col)
    npm_N = _safe_year_value(profitability, yN, "Net Profit Margin", year_col)
    ebitda_m_0 = _safe_year_value(profitability, y0, "EBITDA_Margin", year_col)
    ebitda_m_N = _safe_year_value(profitability, yN, "EBITDA_Margin", year_col)
    roa_0 = _safe_year_value(profitability, y0, "ROA", year_col)
    roa_N = _safe_year_value(profitability, yN, "ROA", year_col)

    debt_ratio_0 = _safe_year_value(solvency, y0, "Debt_Ratio", year_col)
    debt_ratio_N = _safe_year_value(solvency, yN, "Debt_Ratio", year_col)
    ic_0 = _safe_year_value(solvency, y0, "Interest_Coverage", year_col)
    ic_N = _safe_year_value(solvency, yN, "Interest_Coverage", year_col)
    d2e_0 = _safe_year_value(solvency, y0, "Debt_to_EBITDA", year_col)
    d2e_N = _safe_year_value(solvency, yN, "Debt_to_EBITDA", year_col)

    # Multiples 2024
    pe_N = _safe_year_value(bp_multiples, yN, "P/E", year_col)
    pb_N = _safe_year_value(bp_multiples, yN, "P/B", year_col)
    ev_e_N = _safe_year_value(bp_multiples, yN, "EV/EBITDA", year_col)
    ev_f_N = _safe_year_value(bp_multiples, yN, "EV/FCF", year_col)

    # Growth summary (ranges)
    revg_rng = _range_str(growth, "Revenue Growth") if "Revenue Growth" in growth.columns else "N/A"
    nig_rng  = _range_str(growth, "Net Income Growth") if "Net Income Growth" in growth.columns else "N/A"
    fcfg_rng = _range_str(growth, "FCF Growth") if "FCF Growth" in growth.columns else "N/A"

    # Valuation math
    gap = intrinsic_per_share - close_2024
    gap_pct = gap / close_2024 * 100
    avg_fair = (intrinsic_per_share + peer_price) / 2
    avg_gap_pct = (avg_fair - close_2024) / close_2024 * 100

    def label(x):
        if x > 0.15:
            return "MATERIALLY UNDERVALUED"
        if x > 0.05:
            return "MODERATELY UNDERVALUED"
        if x > -0.05:
            return "FAIRLY VALUED"
        return "POTENTIALLY OVERVALUED"

    decision_dcf = label(gap_pct/100)
    decision_blend = label(avg_gap_pct/100)

    # Risk flags (reuse your existing)
    flags = _flag_risks(profitability, solvency, growth, year=yN)
    flags_md = "\n".join([f"- {f}" for f in flags])

    # ---- Decision module: BUY/HOLD/SELL
    fcf_vol_flag = any(("FCF growth" in f or "FCF Growth" in f) for f in flags)
    action, conf, reason, dcf_gap, peer_gap, risk_score = llm_final_action(
        close_2024=close_2024,
        intrinsic_per_share=intrinsic_per_share,
        peer_price=peer_price,
        interest_coverage_2024=ic_N,
        net_profit_margin_2024=npm_N,
        fcf_volatility_flag=fcf_vol_flag
    )

    md = f"""
## 🧠 Fundamental Analyst Agent – Rich Output ({company}, {year})

### 1) Inputs
- **Market Price (Year-End):** `{close_2024:.2f}` USD  
- **DCF Intrinsic Value:** `{intrinsic_per_share:.2f}` USD  *(WACC `{wacc*100:.0f}%`, terminal g `{g*100:.0f}%`)*  
- **Peer-Implied Price:** `{peer_price:.2f}` USD  
- **Blended Fair Value (DCF + Peer avg):** `{avg_fair:.2f}` USD  

### 2) Profitability Signals ({y0} → {year})
- Net Profit Margin: `{_to_pct(npm_0)}` → `{_to_pct(npm_N)}` `{_trend_arrow(npm_0, npm_N)}`
- EBITDA Margin: `{_to_pct(ebitda_m_0)}` → `{_to_pct(ebitda_m_N)}` `{_trend_arrow(ebitda_m_0, ebitda_m_N)}`
- ROA: `{_to_pct(roa_0)}` → `{_to_pct(roa_N)}` `{_trend_arrow(roa_0, roa_N)}`

**LLM interpretation:** EBITDA-based profitability is more stable than net income metrics, implying core operations are relatively resilient while bottom-line results are more cyclical/non-operating-item sensitive.

### 3) Solvency & Leverage Signals ({y0} → {year})
- Debt Ratio: `{_to_num(debt_ratio_0,3)}` → `{_to_num(debt_ratio_N,3)}` `{_trend_arrow(debt_ratio_0, debt_ratio_N)}`
- Interest Coverage: `{_to_num(ic_0,2)}x` → `{_to_num(ic_N,2)}x` `{_trend_arrow(ic_0, ic_N)}`
- Debt / EBITDA: `{_to_num(d2e_0,2)}x` → `{_to_num(d2e_N,2)}x` `{_trend_arrow(d2e_0, d2e_N)}`

**LLM interpretation:** Balance-sheet risk appears manageable overall, but interest coverage weakening in {year} indicates higher sensitivity to earnings downturns.

### 4) Growth Signals (YoY ranges over sample)
- Revenue Growth (range): `{revg_rng}`
- Net Income Growth (range): `{nig_rng}`
- FCF Growth (range): `{fcfg_rng}`

**LLM interpretation:** Growth signals are volatile and non-linear, consistent with a cyclical energy business and supporting conservative long-term cash flow assumptions.

### 5) Market Multiples Snapshot ({year})
- P/E: `{_to_num(pe_N,2)}`
- P/B: `{_to_num(pb_N,2)}`
- EV/EBITDA: `{_to_num(ev_e_N,2)}`
- EV/FCF: `{_to_num(ev_f_N,2)}`

**LLM interpretation:** P/E can be distorted when net income is compressed; EV-based multiples typically provide a more reliable valuation read for capital-intensive cyclicals.

### 6) Valuation Read & Decision
- **DCF vs Market:** `{gap:+.2f}` USD (**{gap_pct:+.1f}%**) → **{decision_dcf}**
- **Blended (DCF+Peer) vs Market:** **{avg_gap_pct:+.1f}%** → **{decision_blend}**

### 7) Risk Flags (rule-based)
{flags_md}

### 8) Final Action (LLM Decision Layer)
- **Action:** **{action}**
- **Confidence:** {conf}
- **Risk Score:** {risk_score}
- **Why:** {reason}

---

### ✅ Final Agent Decision
> **{action}** — driven by mixed intrinsic vs peer signals and cyclical FCF uncertainty (not immediate solvency stress).
"""
    return md.strip()

from IPython.display import Markdown, display

md = llm_agent_output_md_rich(
    company="BP",
    close_2024=close_2024,
    intrinsic_per_share=intrinsic_value_per_share,
    peer_price=price_peer,
    wacc=0.10,
    g=0.02,
    profitability=profitability,
    solvency=solvency,
    growth=growth,
    bp_multiples=bp_multiples,
    year=2024
)

display(Markdown(md))


# In[40]:


import json
import requests

GROQ_API_KEY = "your key"  

def groq_llama3_decide(summary, model="llama-3.1-8b-instant"):
    system = (
    "You are a buy-side Fundamental Analyst Agent. "
    "Return ONLY valid JSON with the following schema: "
    '{ "action":"BUY|HOLD|SELL", '
    '"confidence":"Low|Medium|High", '
    '"rationale":"2-4 sentences", '
    '"key_evidence":["...","...","..."] }. '
    "Rules: "
    "1) Use ONLY the numbers provided in INPUT_JSON. Do not invent facts. "
    "2) Cite at least 3 exact quantitative evidence points. "
    "3) If valuation signals are mixed or conflicting (e.g., DCF downside but strong peer support), "
    "**default to HOLD rather than SELL.** "
    "4) SELL should be chosen only when both intrinsic valuation downside is severe "
    "and financial risk is extreme. "
    "Maintain a professional, conservative investment tone."
)


    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "INPUT_JSON:\n" + json.dumps(summary)}
            ],
            "temperature": 0.2
        },
        timeout=60
    )

    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return json.loads(content)



# In[41]:


summary = {
    "company": "BP",
    "year": 2024,
    "valuation": {
        "market_price_year_end": 27.89,
        "dcf_intrinsic_value_per_share": 24.34,
        "peer_implied_price_per_share": 42.81,
        "wacc": 0.10,
        "terminal_growth": 0.02
    },
    "profitability": {
        "net_profit_margin_2024": 0.002,
        "ebitda_margin_2024": 0.148,
        "roa_2024": 0.001
    },
    "solvency": {
        "interest_coverage_2024": 2.48,
        "debt_to_ebitda_2024": 2.22
    },
    "growth": {
        "fcf_growth_volatility": "high"
    },
    "multiples_2024": {
        "pe": 304.84,
        "pb": 1.96,
        "ev_ebitda": 5.15,
        "ev_fcf": 12.00
    },
    "risk_flags": [
        "Weak interest coverage (~2.48x)",
        "Very low net profit margin (~0.2%)",
        "Highly volatile free cash flow growth"
    ]
}
decision = groq_llama3_decide(summary)
decision
from IPython.display import Markdown, display

display(Markdown(f"""
## 🧠 LLM Decision Output (Llama 3 – BP 2024)

**Final Action:** **{decision["action"]}**  
**Confidence:** {decision["confidence"]}

**Rationale:**  
{decision["rationale"]}

**Key Quantitative Evidence**
- {decision["key_evidence"][0]}
- {decision["key_evidence"][1]}
- {decision["key_evidence"][2]}
"""))


# In[ ]:




