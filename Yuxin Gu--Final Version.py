#!/usr/bin/env python
# coding: utf-8

# In[3]:


from IPython.display import HTML, display

display(HTML("""
<div style="max-width:980px;">
  <h2 style="font-size:36px;font-weight:900;margin:12px 0 12px 0;">📌 Table of Contents</h2>
  <ul style="font-size:20px; line-height:1.9; margin:0 0 12px 18px;">
    <li><a href="#sec-fin" style="color:#7c3aed;text-decoration:none;">1. Financial Statements</a></li>
    <li><a href="#sec-clean" style="color:#7c3aed;text-decoration:none;">2. Data & Cleaning</a></li>
    <li><a href="#sec-ratios" style="color:#7c3aed;text-decoration:none;">3. Ratio Analysis</a></li>
    <li><a href="#sec-multiples" style="color:#7c3aed;text-decoration:none;">4. Multiples</a></li>
    <li><a href="#sec-dcf" style="color:#7c3aed;text-decoration:none;">5. FCF Construction & DCF Valuation</a></li>
    <li><a href="#sec-ddm" style="color:#7c3aed;text-decoration:none;">6. DDM Valuation</a></li>
    <li><a href="#sec-compare" style="color:#7c3aed;text-decoration:none;">7. Valuation Comparison</a></li>
  </ul>
  <hr style="border:none;border-top:1px solid #e5e7eb;margin:16px 0 22px 0;"/>
</div>
"""))


# In[4]:


display(HTML("""
<div id="sec-fin" style="scroll-margin-top:80px;"></div>
<h1 style="
  font-size:40px;font-weight:900;margin:26px 0 6px 0;
  border-left:10px solid #111827;padding-left:14px;">
1) Financial Statements
</h1>
<div style="color:#6b7280;font-size:16px;margin:0 0 14px 24px;">
Income Statement • Balance Sheet • Cash Flow (2020–2024)
</div>
"""))


# In[5]:


import time
import requests
import pandas as pd

API_KEY = os.getenv ("ALPHAVANTAGE_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "ALPHAVANTAGE_API_KEY not found.
        "Please export it in your environment before running the notebook.
    )
SYMBOL = "BP"
YEARS = {"2020", "2021", "2022", "2023", "2024"}

def fetch(function_name: str, max_retries: int = 8):
    url = "https://www.alphavantage.co/query"
    params = {"function": function_name, "symbol": SYMBOL, "apikey": API_KEY}

    for attempt in range(1, max_retries + 1):
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if "annualReports" in data:
            return data

        msg = data.get("Note") or data.get("Information") or data.get("Error Message")
        if msg:
            wait = 1.2 * attempt
            print(f"No annualReports returned, retrying: {data}")
            time.sleep(wait)
            continue

        raise RuntimeError(f"[{function_name}] Unexpected response: {data}")

    raise RuntimeError(f"[{function_name}] Failed after {max_retries} retries (daily quota may be exhausted)")

def to_df_2020_2024(annual_reports):
    df = pd.DataFrame(annual_reports)
    if "fiscalDateEnding" in df.columns:
        df = df[df["fiscalDateEnding"].astype(str).str[:4].isin(YEARS)].copy()
        df = df.sort_values("fiscalDateEnding")
    return df


income = fetch("INCOME_STATEMENT")
time.sleep(1.2)

cash = fetch("CASH_FLOW")
time.sleep(1.2)

balance = fetch("BALANCE_SHEET")


df_income  = to_df_2020_2024(income["annualReports"])
df_cash    = to_df_2020_2024(cash["annualReports"])
df_balance = to_df_2020_2024(balance["annualReports"])

print("Years (Income): ", df_income["fiscalDateEnding"].tolist() if len(df_income) else "EMPTY")
print("Years (Cash):   ", df_cash["fiscalDateEnding"].tolist() if len(df_cash) else "EMPTY")
print("Years (Balance):", df_balance["fiscalDateEnding"].tolist() if len(df_balance) else "EMPTY")

df_income.to_csv("BP_income_2020_2024_all_fields.csv", index=False, encoding="utf-8-sig")
df_cash.to_csv("BP_cashflow_2020_2024_all_fields.csv", index=False, encoding="utf-8-sig")
df_balance.to_csv("BP_balance_2020_2024_all_fields.csv", index=False, encoding="utf-8-sig")

with pd.ExcelWriter("BP_financials_2020_2024_full.xlsx", engine="openpyxl") as writer:
    df_income.to_excel(writer, sheet_name="Income_Statement", index=False)
    df_cash.to_excel(writer, sheet_name="Cash_Flow", index=False)
    df_balance.to_excel(writer, sheet_name="Balance_Sheet", index=False)

print("Saved: BP_financials_2020_2024_full.xlsx and 3 CSV files")



# In[6]:


import pandas as pd

pd.set_option("display.max_rows", None)     
pd.set_option("display.max_columns", None)   
pd.set_option("display.width", None)         


# In[7]:


df_income


# In[8]:


df_balance


# In[9]:


df_cash


# In[10]:


display(HTML("""
<div id="sec-clean" style="scroll-margin-top:80px;"></div>
<h1 style="
  font-size:40px;font-weight:900;margin:26px 0 6px 0;
  border-left:10px solid #111827;padding-left:14px;">
2) Data & Cleaning
</h1>
<div style="color:#6b7280;font-size:16px;margin:0 0 14px 24px;">
Fetch, normalize, align fiscal years, convert to numeric, handle missing values
</div>
"""))


# In[11]:


import pandas as pd
import numpy as np
import re

MISSING_TOKENS = {
    "", " ", "\t", "\n",
    "NA", "N/A", "na", "n/a",
    "NULL", "null",
    "None", "none",
    "-", "--", "—", "_", "__"

}

def normalize_missing(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        v = x.strip()
        return np.nan if v in MISSING_TOKENS else v
    return x

_num_cleanup_re = re.compile(r"[,\s￥$]")

def to_number_maybe(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)

    s = str(x).strip()
    if s in MISSING_TOKENS:
        return np.nan

    neg = False
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        neg = True
        s = s[1:-1].strip()

    s = _num_cleanup_re.sub("", s)

    try:
        val = float(s)
        return -val if neg else val
    except Exception:
        return np.nan

def clean_alpha_vantage_financial_df(df: pd.DataFrame, date_col: str = "fiscalDateEnding") -> pd.DataFrame:
    out = df.copy()


    out = out.map(normalize_missing)

    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    else:
        out.index = pd.Index([str(i).strip() for i in out.index])

    for c in out.columns:
        if out[c].dtype == "datetime64[ns]":
            continue
        out[c] = out[c].map(to_number_maybe)

    out.attrs["source"] = "alphavantage"
    out.attrs["unit"] = "raw_number"
    return out

balance_clean  = clean_alpha_vantage_financial_df(df_balance)
income_clean   = clean_alpha_vantage_financial_df(df_income)
cashflow_clean = clean_alpha_vantage_financial_df(df_cash)


# In[12]:


display(HTML('<div id="sec-ratios"></div>'))


# In[13]:


import pandas as pd
import numpy as np


def pick_first(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def get_series(df, candidates, name=None):
    col = pick_first(df, candidates)
    s = df[col] if col else pd.Series(np.nan, index=df.index)
    if name:
        s = s.rename(name)
    return s

def safe_div(a, b):
    b0 = b.replace(0, np.nan)
    return a / b0


idx = balance_clean.index.intersection(income_clean.index).intersection(cashflow_clean.index)
bal = balance_clean.loc[idx].sort_index()
inc = income_clean.loc[idx].sort_index()
cf  = cashflow_clean.loc[idx].sort_index()

revenue   = get_series(inc, ["totalRevenue"], "revenue")
grossp    = get_series(inc, ["grossProfit"], "gross_profit")
opinc     = get_series(inc, ["operatingIncome"], "operating_income")
ebitda    = get_series(inc, ["ebitda"], "ebitda")
netinc    = get_series(inc, ["netIncome"], "net_income")
pretax    = get_series(inc, ["incomeBeforeTax"], "pretax_income")
interest  = get_series(inc, ["interestExpense"], "interest_expense")

assets    = get_series(bal, ["totalAssets"], "total_assets")
liab      = get_series(bal, ["totalLiabilities"], "total_liabilities")
equity    = get_series(bal, ["totalShareholderEquity", "totalShareholderEquityGrossMinorityInterest"], "total_equity")
cur_assets= get_series(bal, ["totalCurrentAssets"], "current_assets")
cur_liab  = get_series(bal, ["totalCurrentLiabilities"], "current_liabilities")
cash_bal  = get_series(bal, ["cashAndCashEquivalentsAtCarryingValue", "cashAndShortTermInvestments"], "cash_and_equiv")
inv       = get_series(bal, ["inventory"], "inventory")
st_debt   = get_series(bal, ["shortTermDebt"], "short_term_debt")
lt_debt   = get_series(bal, ["longTermDebt"], "long_term_debt")

ocf       = get_series(cf, ["operatingCashflow"], "operating_cashflow")
capex     = get_series(cf, ["capitalExpenditures"], "capex")
dividend  = get_series(cf, ["dividendPayout"], "dividend_payout")


avg_assets = (assets + assets.shift(1)) / 2
avg_equity = (equity + equity.shift(1)) / 2

capex_out = capex.abs()

fcf = (ocf - capex_out).rename("free_cash_flow")

ratios = pd.DataFrame(index=idx)

ratios["gross_margin"]      = safe_div(grossp, revenue)
ratios["operating_margin"]  = safe_div(opinc, revenue)
ratios["net_margin"]        = safe_div(netinc, revenue)
ratios["roa"]               = safe_div(netinc, avg_assets)
ratios["roe"]               = safe_div(netinc, avg_equity)
ratios["ebitda_margin"]     = safe_div(ebitda, revenue)

ratios["current_ratio"]     = safe_div(cur_assets, cur_liab)
ratios["quick_ratio"]       = safe_div(cur_assets - inv, cur_liab)

ratios["debt_to_equity"]    = safe_div((st_debt.fillna(0) + lt_debt.fillna(0)), equity)
ratios["liab_to_assets"]    = safe_div(liab, assets)

ratios["interest_coverage_operating"] = safe_div(opinc, interest.abs())
ratios["interest_coverage_ebitda"]    = safe_div(ebitda, interest.abs())

ratios["asset_turnover"]    = safe_div(revenue, avg_assets)

ratios = ratios.sort_index()

ratios


# In[14]:


avg_assets = (assets + assets.shift(1)) / 2
avg_equity = (equity + equity.shift(1)) / 2


# In[15]:


display(avg_assets)
display(avg_equity)


# In[16]:


display(HTML('<div id="sec-multiples"></div>'))


# In[17]:


display(HTML("""
<div id="sec-ratios" style="scroll-margin-top:80px;"></div>
<h1 style="
  font-size:40px;font-weight:900;margin:26px 0 6px 0;
  border-left:10px solid #111827;padding-left:14px;">
3) Ratio Analysis
</h1>
<div style="color:#6b7280;font-size:16px;margin:0 0 14px 24px;">
Profitability • Returns • Liquidity • Leverage • Coverage • Efficiency
</div>
"""))


# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def pick(df, cols):
    for c in cols:
        if c in df.columns:
            return df[c]
    return pd.Series(np.nan, index=df.index)

def safe_div(a, b):
    return a / b.replace(0, np.nan)

def _existing_cols(df, cols):
    return [c for c in cols if c in df.columns]

def plot_lines(df, cols, title, percent=False):
    cols = _existing_cols(df, cols)
    if not cols:
        return
    d = df[cols].sort_index()
    plt.figure(figsize=(10, 5))
    for c in cols:
        plt.plot(d.index, d[c], marker="o", label=c)
    plt.title(title)
    if percent:
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.xticks(rotation=45)
    plt.grid(True, linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


idx = balance_clean.index.intersection(income_clean.index).intersection(cashflow_clean.index)
bal = balance_clean.loc[idx].sort_index()
inc = income_clean.loc[idx].sort_index()
cf  = cashflow_clean.loc[idx].sort_index()


revenue = pick(inc, ["totalRevenue"]).rename("Revenue")
ebitda  = pick(inc, ["ebitda"]).rename("EBITDA")

ocf = pick(cf, ["operatingCashflow"]).rename("OperatingCashFlow")
capex = pick(cf, ["capitalExpenditures"]).abs().rename("CapEx")


fcf = (ocf - capex).rename("FreeCashFlow")
ebitda_margin = safe_div(ebitda, revenue).rename("EBITDA_Margin")

fcf_components = pd.concat(
    [revenue, ebitda, ebitda_margin, ocf, capex, fcf],
    axis=1
).sort_index()


categories = {
    "Profitability (Margins)": {
        "df": ratios,
        "cols": ["gross_margin", "operating_margin", "net_margin"],
        "percent": True,
    },
    "Profitability (Returns)": {
        "df": ratios,
        "cols": ["roa", "roe"],
        "percent": True,
    },
    "Liquidity": {
        "df": ratios,
        "cols": ["current_ratio", "quick_ratio"],
    },
    "Leverage": {
        "df": ratios,
        "cols": ["debt_to_equity", "liab_to_assets"],
    },
    "Coverage": {
        "df": ratios,
        "cols": ["interest_coverage_operating", "interest_coverage_ebitda"],
    },
    "Cash Flow (Margins & Payout)": {
        "df": ratios,
        "cols": ["ocf_margin", "fcf_margin", "dividend_to_fcf"],
        "percent": True,
    },
    "Cash Flow (Conversion)": {
        "df": ratios,
        "cols": ["fcf_to_ocf"],
    },
    "Efficiency": {
        "df": ratios,
        "cols": ["asset_turnover"],
    },
    "Free Cash Flow (OCF - CapEx)": {
        "df": fcf_components,
        "cols": ["FreeCashFlow"],
    },
    "EBITDA Margin": {
        "df": fcf_components,
        "cols": ["EBITDA_Margin"],
        "percent": True,
    },
}

for title, cfg in categories.items():
    plot_lines(cfg["df"], cfg["cols"], title, percent=cfg.get("percent", False))


# In[19]:


display(HTML("""
<div id="sec-multiples" style="scroll-margin-top:80px;"></div>
<h1 style="
  font-size:40px;font-weight:900;margin:26px 0 6px 0;
  border-left:10px solid #111827;padding-left:14px;">
4) Multiples
</h1>
<div style="color:#6b7280;font-size:16px;margin:0 0 14px 24px;">
P/E • P/S • EV/EBITDA and implied valuation using peer averages
</div>
"""))


# In[20]:


pip install yfinance pandas


# In[21]:


import yfinance as yf

BP = yf.Ticker("BP")

income_BP = BP.financials
income_BP


# In[22]:


balancesheet_BP = BP.balancesheet
balancesheet_BP


# In[23]:


cashflow_BP = BP.cashflow
cashflow_BP


# In[24]:


import yfinance as yf
import pandas as pd
import numpy as np

MISSING_TOKENS = {"", " ", "\t", "\n","NA","N/A","na","n/a","NULL","null","None","none","-","--","—","_","__"}

def normalize_missing(x):
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        v = x.strip()
        return np.nan if v in MISSING_TOKENS else v
    return x

def clean_to_raw_unit(df):
    df = (df.map(normalize_missing) if hasattr(df, "map") else df.applymap(normalize_missing)).copy()
    df = df.drop(columns=[c for c in df.columns if isinstance(c, str) and "__is_missing" in c], errors="ignore")
    df.index = pd.Index([str(i).strip() for i in df.index])

    date_cols = [c for c in df.columns if isinstance(c, (pd.Timestamp, np.datetime64))]
    if not date_cols:
        parsed = pd.to_datetime(df.columns, errors="coerce")
        if pd.notna(parsed).any():
            df.columns = parsed
            date_cols = [c for c in df.columns if isinstance(c, (pd.Timestamp, np.datetime64))]

    for c in date_cols:
        s = df[c]
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            s = s.astype("string").str.replace(r"[,\s￥$]", "", regex=True)
        df[c] = pd.to_numeric(s, errors="coerce")

    df.attrs["unit"] = "raw"
    return df

balancesheet_clean_BP = clean_to_raw_unit(balancesheet_BP)
income_clean_BP       = clean_to_raw_unit(income_BP)
cashflow_clean_BP     = clean_to_raw_unit(cashflow_BP)


# In[25]:


import yfinance as yf

shell = yf.Ticker("Shel")

income_shell = shell.financials
income_shell


# In[26]:


balancesheet_shell = shell.balancesheet
balancesheet_shell


# In[27]:


cashflow_shell = shell.cashflow
cashflow_shell


# In[28]:


import yfinance as yf
import pandas as pd
import numpy as np

MISSING_TOKENS = {"", " ", "\t", "\n","NA","N/A","na","n/a","NULL","null","None","none","-","--","—","_","__"}

def normalize_missing(x):
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        v = x.strip()
        return np.nan if v in MISSING_TOKENS else v
    return x

def clean_to_raw_unit(df):
    df = (df.map(normalize_missing) if hasattr(df, "map") else df.applymap(normalize_missing)).copy()
    df = df.drop(columns=[c for c in df.columns if isinstance(c, str) and "__is_missing" in c], errors="ignore")
    df.index = pd.Index([str(i).strip() for i in df.index])

    date_cols = [c for c in df.columns if isinstance(c, (pd.Timestamp, np.datetime64))]
    if not date_cols:
        parsed = pd.to_datetime(df.columns, errors="coerce")
        if pd.notna(parsed).any():
            df.columns = parsed
            date_cols = [c for c in df.columns if isinstance(c, (pd.Timestamp, np.datetime64))]

    for c in date_cols:
        s = df[c]
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            s = s.astype("string").str.replace(r"[,\s￥$]", "", regex=True)
        df[c] = pd.to_numeric(s, errors="coerce")

    df.attrs["unit"] = "raw"
    return df

balancesheet_clean_shell = clean_to_raw_unit(balancesheet_shell)
income_clean_shell       = clean_to_raw_unit(income_shell)
cashflow_clean_shell     = clean_to_raw_unit(cashflow_shell)


# In[29]:


import yfinance as yf

Exxon_Mobil = yf.Ticker("XOM")

income_xom = Exxon_Mobil.financials
income_xom


# In[30]:


balancesheet_xom = Exxon_Mobil.balancesheet
balancesheet_xom


# In[31]:


cashflow_xom = Exxon_Mobil.cashflow
cashflow_xom


# In[32]:


import yfinance as yf
import pandas as pd
import numpy as np

MISSING_TOKENS = {"", " ", "\t", "\n","NA","N/A","na","n/a","NULL","null","None","none","-","--","—","_","__"}

def normalize_missing(x):
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        v = x.strip()
        return np.nan if v in MISSING_TOKENS else v
    return x

def clean_to_raw_unit(df):
    df = (df.map(normalize_missing) if hasattr(df, "map") else df.applymap(normalize_missing)).copy()
    df = df.drop(columns=[c for c in df.columns if isinstance(c, str) and "__is_missing" in c], errors="ignore")
    df.index = pd.Index([str(i).strip() for i in df.index])

    date_cols = [c for c in df.columns if isinstance(c, (pd.Timestamp, np.datetime64))]
    if not date_cols:
        parsed = pd.to_datetime(df.columns, errors="coerce")
        if pd.notna(parsed).any():
            df.columns = parsed
            date_cols = [c for c in df.columns if isinstance(c, (pd.Timestamp, np.datetime64))]

    for c in date_cols:
        s = df[c]
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            s = s.astype("string").str.replace(r"[,\s￥$]", "", regex=True)
        df[c] = pd.to_numeric(s, errors="coerce")

    df.attrs["unit"] = "raw"
    return df

balancesheet_clean_xom = clean_to_raw_unit(balancesheet_xom)
income_clean_xom       = clean_to_raw_unit(income_xom)
cashflow_clean_xom     = clean_to_raw_unit(cashflow_xom)


# In[33]:


import numpy as np
import pandas as pd

def pick(df, cols):
    for c in cols:
        if c in df.columns:
            return df[c]
    return pd.Series(np.nan, index=df.index)

def get_last(df):
    return df.sort_index().iloc[-1]


# In[34]:


import numpy as np
import pandas as pd

def get_last(df):
    return df.sort_index().iloc[-1]

def pick_val(row, cols, default=np.nan):
    for c in cols:
        if c in row.index and pd.notna(row[c]):
            return float(row[c])
    return default

def calc_multiples_from_financials(
    income_clean,
    balance_clean,
    share_price_2024,
    shares_2024
):
    inc_2024 = get_last(income_clean)
    bal_2024 = get_last(balance_clean)

    revenue = pick_val(inc_2024, ["totalRevenue"])
    net_profit = pick_val(inc_2024, ["netIncome"])
    ebitda = pick_val(inc_2024, ["ebitda"])

    cash = pick_val(
        bal_2024,
        ["cashAndCashEquivalentsAtCarryingValue", "cashAndShortTermInvestments"],
        0.0
    )
    debt = (
        pick_val(bal_2024, ["shortTermDebt"], 0.0)
        + pick_val(bal_2024, ["longTermDebt"], 0.0)
    )

    market_cap = share_price_2024 * shares_2024
    pe = market_cap / net_profit
    ps = market_cap / revenue
    ev = market_cap + debt - cash
    ev_ebitda = ev / ebitda

    return pd.Series({
        "Share Price (2024)": share_price_2024,
        "Shares Outstanding (2024)": shares_2024,
        "Market Cap": market_cap,
        "Revenue": revenue,
        "Net Profit": net_profit,
        "EBITDA": ebitda,
        "Total Debt": debt,
        "Cash": cash,
        "P/E": pe,
        "P/S": ps,
        "EV": ev,
        "EV / EBITDA": ev_ebitda
    })


# In[35]:


import yfinance as yf

tickers = {
    "BP": "BP",
    "Shell": "SHEL",
    "XOM": "XOM"
}

def get_close_price_2024_12_31(ticker):
    df = yf.download(
        ticker,
        start="2024-12-30",
        end="2025-01-02",
        progress=False
    )
    return float(df.loc["2024-12-31", "Close"])

share_price_2024_BP = get_close_price_2024_12_31(tickers["BP"])
share_price_2024_shell = get_close_price_2024_12_31(tickers["Shell"])
share_price_2024_xom = get_close_price_2024_12_31(tickers["XOM"])

share_price_2024_BP, share_price_2024_shell, share_price_2024_xom


# In[36]:


import numpy as np
import pandas as pd
from IPython.display import display

def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def pick_from_fs(df: pd.DataFrame, keys, year_col=None, default=np.nan):

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return default

    if year_col is None:
        year_col = df.columns.max() if len(df.columns) else None

    if year_col in df.columns:
        idx_norm = {_norm(i): i for i in df.index}

        for k in keys:
            kn = _norm(k)
            if kn in idx_norm:
                v = df.loc[idx_norm[kn], year_col]
                return float(v) if pd.notna(v) else default
        for i in df.index:
            inorm = _norm(i)
            if any(_norm(k) in inorm or inorm in _norm(k) for k in keys):
                v = df.loc[i, year_col]
                return float(v) if pd.notna(v) else default

    if isinstance(df.index, pd.DatetimeIndex):
        last_row = df.sort_index().iloc[-1]
        col_norm = {_norm(c): c for c in df.columns}
        for k in keys:
            kn = _norm(k)
            if kn in col_norm:
                v = last_row[col_norm[kn]]
                return float(v) if pd.notna(v) else default
        for c in df.columns:
            cn = _norm(c)
            if any(_norm(k) in cn or cn in _norm(k) for k in keys):
                v = last_row[c]
                return float(v) if pd.notna(v) else default

    return default

def calc_multiples_fs(df_income, df_balance, price_2024, shares_2024):
    year_col = df_income.columns.max()

    revenue = pick_from_fs(df_income, ["totalRevenue", "revenue", "Revenue"], year_col)
    net_profit = pick_from_fs(df_income, ["netIncome", "netIncomeLoss", "netprofit", "Net Profit"], year_col)
    ebitda = pick_from_fs(df_income, ["ebitda", "EBITDA"], year_col)

    cash = pick_from_fs(df_balance, [
        "cashAndCashEquivalentsAtCarryingValue",
        "cashAndShortTermInvestments",
        "cashAndCashEquivalents",
        "cash"
    ], year_col, default=0.0)

    debt = (
        pick_from_fs(df_balance, ["shortTermDebt", "debtCurrent"], year_col, default=0.0) +
        pick_from_fs(df_balance, ["longTermDebt", "debtNonCurrent"], year_col, default=0.0)
    )

    price = float(price_2024)
    shares = float(shares_2024)

    market_cap = price * shares
    pe = market_cap / net_profit if pd.notna(net_profit) and net_profit != 0 else np.nan
    ps = market_cap / revenue if pd.notna(revenue) and revenue != 0 else np.nan
    ev = market_cap + debt - cash
    ev_ebitda = ev / ebitda if pd.notna(ebitda) and ebitda != 0 else np.nan

    return pd.Series({
        "Share Price (2024)": price,
        "Shares Outstanding (2024)": shares,
        "Market Cap": market_cap,
        "Revenue": revenue,
        "Net Profit": net_profit,
        "EBITDA": ebitda,
        "Total Debt": debt,
        "Cash": cash,
        "P/E": pe,
        "P/S": ps,
        "EV": ev,
        "EV / EBITDA": ev_ebitda
    })

shares_2024_BP    = 15_000_000_000  
shares_2024_shell = 7_800_000_000   
shares_2024_xom   = 4_200_000_000

multiples_bp = calc_multiples_fs(
    income_clean_BP, balancesheet_clean_BP, share_price_2024_BP, shares_2024_BP if "shares_2024_BP" in globals() else shares_2024
)

multiples_shell = calc_multiples_fs(
    income_clean_shell,balancesheet_clean_shell,share_price_2024_shell,shares_2024_shell
)


multiples_xom = calc_multiples_fs(
    income_clean_xom, balancesheet_clean_xom, share_price_2024_xom, shares_2024_xom
)

multiples_all = pd.DataFrame({"BP": multiples_bp, "Shell": multiples_shell, "XOM": multiples_xom}).T
display(multiples_all)


# In[37]:


import numpy as np
import pandas as pd
from IPython.display import display, Markdown

def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / b.replace(0, np.nan)

def pick_row_by_keys(fs_df: pd.DataFrame, keys):
    def _norm(s): return "".join(ch.lower() for ch in str(s) if ch.isalnum())
    idx_norm = {_norm(i): i for i in fs_df.index}

    for k in keys:
        kn = _norm(k)
        if kn in idx_norm:
            return pd.to_numeric(fs_df.loc[idx_norm[kn]], errors="coerce")

    key_norms = [_norm(k) for k in keys]
    for i in fs_df.index:
        inorm = _norm(i)
        if any(kn in inorm or inorm in kn for kn in key_norms):
            return pd.to_numeric(fs_df.loc[i], errors="coerce")

    return pd.Series(np.nan, index=fs_df.columns)

def avg_two_years(series: pd.Series):
    s = series.sort_index()
    return (s + s.shift(1)) / 2

def build_ratios(income_df, balance_df, cashflow_df):
    # Income
    revenue = pick_row_by_keys(income_df, ["totalRevenue", "Revenue"]).sort_index()
    grossp  = pick_row_by_keys(income_df, ["grossProfit", "grossProfitLoss"]).sort_index()
    opinc   = pick_row_by_keys(income_df, ["operatingIncome", "operatingIncomeLoss"]).sort_index()
    netinc  = pick_row_by_keys(income_df, ["netIncome", "netIncomeLoss"]).sort_index()
    ebitda  = pick_row_by_keys(income_df, ["ebitda", "EBITDA"]).sort_index()

    # Balance
    assets  = pick_row_by_keys(balance_df, ["totalAssets"]).sort_index()
    equity  = pick_row_by_keys(balance_df, ["totalShareholderEquity", "totalStockholdersEquity"]).sort_index()
    liab    = pick_row_by_keys(balance_df, ["totalLiabilities"]).sort_index()

    cur_assets = pick_row_by_keys(balance_df, ["totalCurrentAssets"]).sort_index()
    cur_liab   = pick_row_by_keys(balance_df, ["totalCurrentLiabilities"]).sort_index()
    inv        = pick_row_by_keys(balance_df, ["inventory"]).sort_index()

    st_debt = pick_row_by_keys(balance_df, ["shortTermDebt", "debtCurrent"]).sort_index()
    lt_debt = pick_row_by_keys(balance_df, ["longTermDebt", "debtNonCurrent"]).sort_index()

    # Cashflow
    ocf = pick_row_by_keys(cashflow_df, ["operatingCashflow", "cashflowFromOperatingActivities", "NetCashProvidedByOperatingActivities"]).sort_index()
    capex = pick_row_by_keys(cashflow_df, ["capitalExpenditures", "capitalExpenditure", "paymentsForCapitalExpenditures"]).sort_index().abs()
    fcf = (ocf - capex).sort_index()

    dividends = pick_row_by_keys(cashflow_df, ["dividendsPaid", "paymentsOfDividends", "dividendPayments", "dividendPayout"]).sort_index().abs()
    interest = pick_row_by_keys(cashflow_df, ["interestPaid", "interestPayments"]).sort_index()

    # Averages (for ROA/ROE/turnover)
    avg_assets = avg_two_years(assets)
    avg_equity = avg_two_years(equity)

    ratios = pd.DataFrame(index=revenue.index)

    ratios["gross_margin"] = safe_div(grossp, revenue)
    ratios["operating_margin"] = safe_div(opinc, revenue)
    ratios["net_margin"] = safe_div(netinc, revenue)
    ratios["ebitda_margin"] = safe_div(ebitda, revenue)

    ratios["roa"] = safe_div(netinc, avg_assets)
    ratios["roe"] = safe_div(netinc, avg_equity)

    ratios["ocf_margin"] = safe_div(ocf, revenue)
    ratios["fcf_margin"] = safe_div(fcf, revenue)
    ratios["fcf_to_ocf"] = safe_div(fcf, ocf)
    ratios["dividend_to_fcf"] = safe_div(dividends, fcf)

    ratios["current_ratio"] = safe_div(cur_assets, cur_liab)
    ratios["quick_ratio"] = safe_div(cur_assets - inv, cur_liab)

    ratios["debt_to_equity"] = safe_div(st_debt.fillna(0) + lt_debt.fillna(0), equity)
    ratios["liab_to_assets"] = safe_div(liab, assets)

    ratios["interest_coverage_operating"] = safe_div(opinc, interest.abs())
    ratios["interest_coverage_ebitda"] = safe_div(ebitda, interest.abs())

    ratios["asset_turnover"] = safe_div(revenue, avg_assets)

    return ratios

ratios_bp = build_ratios(income_clean_BP, balancesheet_clean_BP, cashflow_clean_BP)
ratios_shell = build_ratios(income_clean_shell, balancesheet_clean_shell, cashflow_clean_shell)
ratios_xom = build_ratios(income_clean_xom, balancesheet_clean_xom, cashflow_clean_xom)

YEAR = 2024
ratio_cols = [
    "gross_margin","operating_margin","net_margin","ebitda_margin",
    "roa","roe",
    "ocf_margin","fcf_margin","fcf_to_ocf","dividend_to_fcf",
    "current_ratio","quick_ratio",
    "debt_to_equity","liab_to_assets",
    "interest_coverage_operating","interest_coverage_ebitda",
    "asset_turnover"
]

def row_for_year(df, year):
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index.year
        df = df.copy()
        df.index = idx
    if year in df.index:
        return df.loc[year]
    return df.iloc[-1]

bp_yr = row_for_year(ratios_bp, YEAR)
sh_yr = row_for_year(ratios_shell, YEAR)
xo_yr = row_for_year(ratios_xom, YEAR)

common = [c for c in ratio_cols if c in ratios_bp.columns and c in ratios_shell.columns and c in ratios_xom.columns]
comp = pd.DataFrame({"BP": bp_yr[common], "Shell": sh_yr[common], "XOM": xo_yr[common]}).T

def is_pct(c): return any(k in c.lower() for k in ["margin","roa","roe"])
pct_cols = [c for c in comp.columns if is_pct(c)]
num_cols = [c for c in comp.columns if c not in pct_cols]

display(
    comp.style
    .format({c:"{:.2%}" for c in pct_cols} | {c:"{:,.2f}" for c in num_cols})
    .apply(lambda r: ["background-color:#fff3cd" if r.name=="BP" else "" for _ in r], axis=1)
    .set_properties(**{"text-align":"right","font-family":"Arial","font-size":"12pt"})
    .set_table_styles([{"selector":"th","props":[("text-align","center"),("font-size","13pt")]}])
)

peer_mean = comp.loc[["Shell","XOM"]].mean(axis=0)

def prefer_lower(c): return any(k in c.lower() for k in ["debt","liab"])
def fmt(c, v): return f"{v*100:.2f}%" if is_pct(c) else f"{v:,.2f}"

sections = {
    "Margins & Profitability": ["gross_margin","operating_margin","net_margin","ebitda_margin"],
    "Returns": ["roa","roe"],
    "Cash Flow Quality": ["ocf_margin","fcf_margin","fcf_to_ocf","dividend_to_fcf"],
    "Liquidity": ["current_ratio","quick_ratio"],
    "Leverage": ["debt_to_equity","liab_to_assets"],
    "Coverage": ["interest_coverage_operating","interest_coverage_ebitda"],
    "Efficiency": ["asset_turnover"],
} 
display(Markdown)


# In[38]:


import matplotlib.pyplot as plt

ev_ebitda = multiples_all["EV / EBITDA"]

plt.figure(figsize=(6,4))
ev_ebitda.plot(kind="bar")
plt.ylabel("EV / EBITDA")
plt.title("EV / EBITDA Comparison (2024)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# In[39]:


shell_ev_ebitda = 16.00
xom_ev_ebitda = 42.00

bp_ebitda = 154.76
bp_shares = 150.0

avg_ev_ebitda = (shell_ev_ebitda + xom_ev_ebitda) / 2
implied_ev = bp_ebitda * avg_ev_ebitda
value_per_share = implied_ev / bp_shares


# In[40]:


print(avg_ev_ebitda, implied_ev, value_per_share)


# In[41]:


display(HTML("""
<div id="sec-dcf" style="scroll-margin-top:80px;"></div>
<h1 style="
  font-size:40px;font-weight:900;margin:26px 0 6px 0;
  border-left:10px solid #111827;padding-left:14px;">
5) FCF Construction & DCF Valuation
</h1>
<div style="color:#6b7280;font-size:16px;margin:0 0 14px 24px;">
FCF (OCF − Capex) • Forecast • Discount (WACC) • Terminal Value • Equity Value / Share
</div>
"""))


# In[42]:


import numpy as np
import pandas as pd

def pick(df, cols):
    for c in cols:
        if c in df.columns:
            return df[c]
    return pd.Series(np.nan, index=df.index)

def safe_div(a, b):
    return a / b.replace(0, np.nan)

idx = balance_clean.index.intersection(income_clean.index).intersection(cashflow_clean.index)
bal = balance_clean.loc[idx].sort_index()
inc = income_clean.loc[idx].sort_index()

capex_out = capex.abs()

fcf = (ocf - capex_out).rename("free_cash_flow")


# In[43]:


display (fcf)


# In[44]:


ratios["ocf_margin"]        = safe_div(ocf, revenue)
ratios["fcf_margin"]        = safe_div(fcf, revenue)
ratios["fcf_to_ocf"]        = safe_div(fcf, ocf)
ratios["dividend_to_fcf"]   = safe_div(dividend.abs(), fcf)


# In[45]:


fcf_cols = [
    "fcf_margin",
    "fcf_to_ocf",
    "dividend_to_fcf"
]

fcf_cols_exist = [c for c in fcf_cols if c in ratios.columns]

ratios_fcf = ratios[fcf_cols_exist]
ratios_fcf


# In[46]:


import numpy as np
import pandas as pd

def pick(df, cols):
    for c in cols:
        if c in df.columns:
            return df[c]
    return pd.Series(np.nan, index=df.index)

idx = income_clean.index.intersection(cashflow_clean.index)
inc = income_clean.loc[idx].sort_index()
cf  = cashflow_clean.loc[idx].sort_index()

revenue = pick(inc, ["totalRevenue"]).rename("Revenue")
ocf     = pick(cf, ["operatingCashflow"]).rename("OperatingCashFlow")
capex   = pick(cf, ["capitalExpenditures"]).abs().rename("CapEx")

fcf_hist = (ocf - capex).rename("FCF")

fcf_hist

use_years = 3
min_g, max_g = -0.05, 0.08

fcf_recent = fcf_hist.dropna().iloc[-use_years:]
start, end = fcf_recent.iloc[0], fcf_recent.iloc[-1]

if start > 0:
    g = (end / start) ** (1 / (len(fcf_recent) - 1)) - 1
else:
    g = fcf_recent.pct_change().mean()

growth_rate = float(np.clip(g, min_g, max_g))
growth_rate


# In[47]:


import numpy as np
import pandas as pd

forecast_years = 5
g_stage1 = -0.02
wacc = 0.09
g_terminal = 0.03       


# In[48]:


last_fcf = fcf_hist.dropna().iloc[-1]
last_fcf


# In[49]:


years = np.arange(1, forecast_years + 1)

fcf_forecast = pd.Series(
    [last_fcf * (1 + g_stage1) ** t for t in years],
    index=years,
    name="Forecast_FCF"
)

display(fcf_forecast)


# In[50]:


discount_factors = 1 / (1 + wacc) ** years

pv_fcf = pd.Series(
    fcf_forecast.values * discount_factors,
    index=years,
    name="PV_FCF"
)

display(pd.DataFrame({
    "FCF": fcf_forecast,
    "Discount_Factor": discount_factors,
    "PV_FCF": pv_fcf
}))


# In[51]:


fcf_year6 = fcf_forecast.iloc[-1] * (1 + g_terminal)

terminal_value = fcf_year6 / (wacc - g_terminal)
pv_terminal = terminal_value / (1 + wacc) ** forecast_years

terminal_value, pv_terminal


# In[52]:


enterprise_value = pv_fcf.sum() + pv_terminal
enterprise_value


# In[53]:


bal_last = balance_clean.sort_index().iloc[-1:]

cash = pick(
    bal_last,
    ["cashAndCashEquivalentsAtCarryingValue", "cashAndShortTermInvestments"]
).iloc[0]

total_debt = (
    pick(bal_last, ["shortTermDebt"]).fillna(0).iloc[0]
    + pick(bal_last, ["longTermDebt"]).fillna(0).iloc[0]
)
net_debt = total_debt - cash


# In[54]:


equity_value = enterprise_value - net_debt
equity_value


# In[55]:


import numpy as np
import pandas as pd

def pick(df, cols):
    for c in cols:
        if c in df.columns:
            return df[c]
    return pd.Series(np.nan, index=df.index)

bal_last = balance_clean.sort_index().iloc[-1:]

shares_2024 = pick(
    bal_last,
    [
        "commonStockSharesOutstanding",
        "commonStockSharesIssued",
        "weightedAverageShsOut",
    ]
).iloc[0]

value_per_share_dcf = equity_value / shares_2024

shares_2024, value_per_share_dcf


# In[56]:


display(HTML("""
<div id="sec-ddm" style="scroll-margin-top:80px;"></div>
<h1 style="
  font-size:40px;font-weight:900;margin:26px 0 6px 0;
  border-left:10px solid #111827;padding-left:14px;">
6) DDM Valuation
</h1>
<div style="color:#6b7280;font-size:16px;margin:0 0 14px 24px;">
Dividend growth estimation • Cost of equity (Ke) • Gordon Growth Model (GGM)
</div>
"""))


# In[57]:


import numpy as np
import pandas as pd

def pick(df, cols):
    for c in cols:
        if c in df.columns:
            return df[c]
    return pd.Series(np.nan, index=df.index)

def ggm_value(D1, ke, g):
    return D1 / (ke - g)


# In[58]:


div_raw = pick(cashflow_clean, ["dividendPayout"]).dropna().sort_index()
div_hist =div_raw.abs().rename("Dividends")


# In[59]:


div_growth = div_hist.pct_change().rename("Dividend_Growth")


# In[60]:


use_years = 3
min_g, max_g = -0.05, 0.08
g_est = div_growth.dropna().iloc[-use_years:].mean()
g_future = float(np.clip(g_est, min_g, max_g))  


# In[61]:


D0 = div_hist.iloc[-1]
D1 = D0 * (1 + g_future)


# In[62]:


def ggm_value(D1, ke, g):
    return D1 / (ke - g)

ke = 0.10 
equity_value_ggm = ggm_value(D1, ke, g_future)

equity_value_ggm


# In[63]:


ddm_table = pd.concat([div_hist, div_growth], axis=1)
ggm_summary = pd.Series({
    "D0 (Last Dividends)": D0,
    "Estimated g (future)": g_future,
    "D1 (Next Dividends)": D1,
    "Ke (Cost of Equity)": ke,
    "Equity Value (GGM)": equity_value_ggm
})


# In[64]:


ddm_table, ggm_summary


# In[65]:


value_per_share_ddm = equity_value_ggm / shares_2024

value_per_share_ddm


# In[66]:


display(HTML('<div id="sec-compare-conclusion"></div>'))


# In[67]:


display(HTML("""
<div id="sec-compare" style="scroll-margin-top:80px;"></div>
<h1 style="
  font-size:40px;font-weight:900;margin:26px 0 6px 0;
  border-left:10px solid #111827;padding-left:14px;">
7) Valuation Comparison
</h1>
<div style="color:#6b7280;font-size:16px;margin:0 0 14px 24px;">
DCF vs DDM vs Multiples vs 2024 year-end market price
</div>
"""))


# In[68]:


from IPython.display import display, Markdown
import numpy as np

def pct_diff(intrinsic, market):
    return (intrinsic / market - 1) * 100

results = {
    "DCF": value_per_share_dcf,
    "DDM": value_per_share_ddm,
    "Multiples": value_per_share
}

market_price = share_price_2024_BP

lines = []
lines.append("## Valuation Summary vs Market Price (2024 YE)\n")
lines.append(f"**Market Price (2024 YE):** {market_price:,.2f}\n")

for method, value in results.items():
    diff = pct_diff(value, market_price)
    status = "undervalued" if diff > 0 else "overvalued"
    lines.append(
        f"- **{method} Value:** {value:,.2f} "
        f"({diff:+.1f}% vs market → *{status}*)"
    )

avg_value = np.mean(list(results.values()))
avg_diff = pct_diff(avg_value, market_price)
overall_view = "undervalued" if avg_diff > 0 else "overvalued"

lines.append("\n---\n")
lines.append(
    f"### Conclusion\n"
    f"Across DCF, DDM, and trading multiples, the **average intrinsic value** is "
    f"**{avg_value:,.2f}**, which implies the stock is **{overall_view}** by "
    f"**{avg_diff:.1f}%** relative to the 2024 year-end market price."
)

display(Markdown("\n".join(lines)))


# In[69]:


import numpy as np
import pandas as pd
from IPython.display import display, Markdown


MARKET_PRICE_VAR = "share_price_2024_BP"


DCF_VAR = "value_per_share_dcf"
DDM_VAR = "value_per_share_ddm"      
GGM_FALLBACK_VAR = "value_per_share_ggm"
MULT_VAR = "value_per_share"


def getv(name, default=np.nan):
    return globals().get(name, default)

market_price = getv(MARKET_PRICE_VAR)

dcf = getv(DCF_VAR)
ddm = getv(DDM_VAR, getv(GGM_FALLBACK_VAR))
mult = getv(MULT_VAR)

def pct_upside(iv, px):
    if pd.isna(iv) or pd.isna(px) or px == 0:
        return np.nan
    return iv / px - 1

rows = [
    ("DCF", dcf),
    ("DDM / GGM", ddm),
    ("Multiples", mult),
]

summary = pd.DataFrame(rows, columns=["Method", "Intrinsic Value / Share"])
summary["Market Price (2024 YE)"] = market_price
summary["Upside / Downside"] = summary.apply(lambda r: pct_upside(r["Intrinsic Value / Share"], market_price), axis=1)

available = summary["Intrinsic Value / Share"].dropna()
blended = available.mean() if len(available) else np.nan
blended_upside = pct_upside(blended, market_price)

conclusion = "N/A"
if pd.notna(blended_upside):
    if blended_upside > 0.10:
        conclusion = "Undervalued (blended > +10%)"
    elif blended_upside > 0:
        conclusion = "Slightly undervalued (0%~+10%)"
    elif blended_upside > -0.10:
        conclusion = "Fair value (-10%~0%)"
    else:
        conclusion = "Overvalued (< -10%)"


display(Markdown("### Summary Table (Intrinsic Value vs Market Price)"))

styled = (
    summary.set_index("Method")
    .style
    .format({
        "Intrinsic Value / Share": "{:,.2f}",
        "Market Price (2024 YE)": "{:,.2f}",
        "Upside / Downside": "{:.1%}",
    })
    .set_properties(**{"text-align":"right","font-family":"Arial","font-size":"12pt"})
    .set_table_styles([{"selector":"th","props":[("text-align","center"),("font-size","13pt"),
                                                 ("background-color","#111827"),("color","white")]}])
)

display(styled)

display(Markdown(
    f"**Blended intrinsic value (average of available methods):** `{blended:,.2f}`  \n"
    f"**Blended upside/downside vs market:** `{blended_upside:+.1%}`  \n"
    f"**Conclusion:** **{conclusion}**"
))


# In[70]:


import numpy as np
import pandas as pd
from IPython.display import display, Markdown

bp_market_price = share_price_2024_BP

bp_dcf = value_per_share_dcf
bp_ddm = globals().get("value_per_share_ddm", globals().get("value_per_share_ggm", np.nan))
bp_multiples = value_per_share

def pct_upside(iv, px):
    return (iv / px - 1) * 100

def is_pct(col):
    return any(k in str(col).lower() for k in ["margin", "roa", "roe", "growth", "yield", "rate"])

def fmt(col, v):
    if pd.isna(v): return "NaN"
    if is_pct(col): return f"{v*100:.2f}%"
    return f"{v:,.2f}"

def prefer_lower(col):
    c = str(col).lower()
    return any(k in c for k in ["debt", "liab", "net_debt"])

def bullet_compare_ratio(col):
    bp = comp.loc["BP", col] if col in comp.columns else np.nan
    sh = comp.loc["Shell", col] if col in comp.columns else np.nan
    xo = comp.loc["XOM", col] if col in comp.columns else np.nan
    peer_mean = np.nanmean([sh, xo])

    if pd.isna(bp) or np.isnan(peer_mean):
        return None

    if prefer_lower(col):
        better = bp < peer_mean
        pref = "lower is better"
    else:
        better = bp > peer_mean
        pref = "higher is better"

    verdict = "stronger" if better else "weaker"
    return (
        f"- **{col}**: BP **{fmt(col,bp)}** vs peer mean **{fmt(col,peer_mean)}** "
        f"(Shell {fmt(col,sh)}, XOM {fmt(col,xo)}) → BP is **{verdict}** ({pref})."
    )

def bullet_compare_multiple(col):
    if col not in multiples_all.columns:
        return None
    bp = multiples_all.loc["BP", col]
    sh = multiples_all.loc["Shell", col]
    xo = multiples_all.loc["XOM", col]
    peer_mean = np.nanmean([sh, xo])
    if pd.isna(bp) or np.isnan(peer_mean):
        return None
    cheaper = bp < peer_mean
    tag = "cheaper vs peers" if cheaper else "richer vs peers"
    return f"- **{col}**: BP **{bp:,.2f}** vs peer mean **{peer_mean:,.2f}** (Shell {sh:,.2f}, XOM {xo:,.2f}) → BP is **{tag}**."

lines = []
lines.append("# BP Integrated Analysis: Valuation + Peer Ratios (Shell & XOM)\n")

lines.append("## 1) Valuation Summary vs 2024 Year-End Price\n")
lines.append(f"- **Market Price (2024 YE):** {bp_market_price:,.2f}\n")

vals = {
    "DCF (FCF-based)": bp_dcf,
    "DDM / GGM (Dividend-based)": bp_ddm,
    "Trading Multiples (Peer EV/EBITDA)": bp_multiples
}

for k, v in vals.items():
    if pd.isna(v):
        lines.append(f"- **{k}:** NaN (not available)")
        continue
    u = pct_upside(v, bp_market_price)
    view = "upside" if u > 0 else "downside"
    lines.append(f"- **{k} Intrinsic Value:** {v:,.2f} (**{u:+.1f}%** {view} vs market)")

avg_iv = np.nanmean([v for v in vals.values() if pd.notna(v)])
avg_up = pct_upside(avg_iv, bp_market_price) if pd.notna(avg_iv) else np.nan
lines.append("\n**Blended View (simple average of available methods):** "
             f"{avg_iv:,.2f} (**{avg_up:+.1f}%** vs market)\n")

lines.append("## 2) Why the Methods May Differ\n")
lines.append(
    "- **DCF** is driven by assumptions on future FCF growth, WACC, and terminal growth; it is the most sensitive to long-run cash generation.\n"
    "- **DDM/GGM** depends on dividend level and sustainable dividend growth; it tends to be conservative if payout policy is volatile.\n"
    "- **Multiples** reflects how the market prices comparable firms today (sentiment, cycle, risk premia), so it can diverge from fundamentals.\n"
)

lines.append("## 3) Peer Multiples Snapshot (BP vs Shell vs XOM)\n")
for c in ["EV / EBITDA", "P/E", "P/S"]:
    b = bullet_compare_multiple(c)
    if b: lines.append(b)

lines.append(
    "\nInterpretation tip: **Lower EV/EBITDA** can mean BP is **cheaper**, but also can reflect **higher perceived risk**, "
    "weaker profitability, or lower growth expectations.\n"
)

lines.append("## 4) Fundamental Ratio Comparison (BP vs Peers)\n")

sections = {
    "Margins & Profitability": ["gross_margin","operating_margin","net_margin","ebitda_margin"],
    "Returns": ["roa","roe"],
    "Cash Flow Quality": ["ocf_margin","fcf_margin","fcf_to_ocf","dividend_to_fcf"],
    "Liquidity": ["current_ratio","quick_ratio"],
    "Leverage": ["debt_to_equity","liab_to_assets"],
    "Coverage": ["interest_coverage_operating","interest_coverage_ebitda"],
    "Efficiency": ["asset_turnover"],
}

score = 0
count = 0
key_strengths = []
key_weaknesses = []

for sec, cols in sections.items():
    cols = [c for c in cols if c in comp.columns]
    if not cols:
        continue

    lines.append(f"### {sec}")
    for col in cols:
        b = bullet_compare_ratio(col)
        if not b:
            continue
        lines.append(b)

        # score for summary
        bp = comp.loc["BP", col]
        sh = comp.loc["Shell", col]
        xo = comp.loc["XOM", col]
        pm = np.nanmean([sh, xo])

        if pd.isna(bp) or np.isnan(pm):
            continue

        if prefer_lower(col):
            better = bp < pm
        else:
            better = bp > pm

        score += 1 if better else -1
        count += 1


        diff = bp - pm
        if prefer_lower(col):
            diff = -diff  # so "positive" means better
        if better:
            key_strengths.append((abs(diff), col))
        else:
            key_weaknesses.append((abs(diff), col))

    lines.append("")

key_strengths = [c for _, c in sorted(key_strengths, reverse=True)[:3]]
key_weaknesses = [c for _, c in sorted(key_weaknesses, reverse=True)[:3]]

overall_fund = "overall stronger" if score > 0 else ("overall weaker" if score < 0 else "broadly in-line")

lines.append("---")
lines.append("## 5) Integrated Conclusion (Valuation + Fundamentals)\n")

if pd.notna(avg_up):
    if avg_up > 10:
        val_view = "materially undervalued"
    elif avg_up > 0:
        val_view = "modestly undervalued"
    elif avg_up > -10:
        val_view = "roughly fairly valued"
    else:
        val_view = "overvalued"
else:
    val_view = "inconclusive (missing valuation inputs)"

lines.append(
    f"### Bottom Line\n"
    f"Based on the three valuation approaches, BP appears **{val_view}** relative to the 2024 year-end market price "
    f"(blended implied value **{avg_iv:,.2f}**, **{avg_up:+.1f}%** vs market).\n"
)

lines.append(
    f"### What the Peer Ratios Suggest\n"
    f"Fundamentally, BP looks **{overall_fund}** versus Shell and XOM across the available ratio set. "
    f"This matters because **multiples are often justified by fundamentals**: stronger margins/returns/cash conversion and "
    f"lower leverage tend to support higher valuation multiples.\n"
)

if key_strengths:
    lines.append("**BP relative strengths (largest favorable gaps vs peers):** " + ", ".join([f"`{c}`" for c in key_strengths]) + ".")
if key_weaknesses:
    lines.append("**BP relative weaknesses (largest unfavorable gaps vs peers):** " + ", ".join([f"`{c}`" for c in key_weaknesses]) + ".")

lines.append(
    "\n### Reconciling Valuation With Fundamentals\n"
    "- If BP screens **cheaper on EV/EBITDA** while also showing **weaker margins/returns or higher leverage**, the discount may be justified.\n"
    "- If BP screens **cheaper** but fundamentals are **in-line or stronger**, the discount could represent a potential opportunity.\n"
    "- Differences between DCF and DDM are common for oil majors because **dividend policy** and **cycle-driven earnings** can be volatile; "
    "DCF tends to be the better anchor if you trust the FCF forecasts.\n"
)

lines.append(
    "### Practical Takeaway\n"
    "Use the blended valuation as a starting point, then stress-test the result:\n"
    "- **DCF sensitivity:** WACC ±1% and terminal growth ±1% can move intrinsic value materially.\n"
    "- **FCF quality:** confirm whether FCF is sustainable (OCF stability, capex discipline).\n"
    "- **Balance sheet risk:** leverage and coverage ratios affect downside protection in a commodity downturn.\n"
)

display(Markdown("\n".join(lines)))


# In[71]:


import requests, json

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

def ask_ollama(prompt: str, model: str = "llama3", temperature: float = 0.2) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["response"]

print(ask_ollama("Say OK in one word."))


# In[73]:


import numpy as np
import pandas as pd
from IPython.display import Markdown, display

market = float(share_price_2024_BP)
dcf    = float(value_per_share_dcf)
ddm    = float(value_per_share_ddm)
mult   = float(value_per_share)

vals = pd.DataFrame({
    "Method": ["Market", "DCF", "DDM", "Multiples"],
    "Value":  [market, dcf, ddm, mult]
})
vals["Upside vs Market"] = vals["Value"]/market - 1
display(vals)

import requests, json, textwrap

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

def ask_ollama(prompt: str, model: str = "llama3", temperature: float = 0.2) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json().get("response", "")

intrinsic_vals = np.array([dcf, ddm, mult], dtype=float)
avg_intrinsic  = float(np.mean(intrinsic_vals))
median_intr    = float(np.median(intrinsic_vals))
min_intr       = float(np.min(intrinsic_vals))
max_intr       = float(np.max(intrinsic_vals))

up_avg    = avg_intrinsic/market - 1
up_median = median_intr/market - 1

dispersion = (max_intr - min_intr) / avg_intrinsic if avg_intrinsic != 0 else np.nan

if up_median >= 0.20:
    reco = "BUY"
elif up_median <= -0.10:
    reco = "SELL"
else:
    reco = "HOLD"

prompt = f"""
You are an equity research analyst. Write an investment recommendation for BP based ONLY on the numbers below.

Market price (2024 YE): {market:.6f}
DCF intrinsic value per share: {dcf:.6f}
DDM intrinsic value per share: {ddm:.6f}
Multiples intrinsic value per share: {mult:.6f}

Average intrinsic (3 methods): {avg_intrinsic:.6f}  (Upside vs market: {up_avg*100:.2f}%)
Median intrinsic (3 methods):  {median_intr:.6f}   (Upside vs market: {up_median*100:.2f}%)
Range (min~max): {min_intr:.6f} ~ {max_intr:.6f}
Method disagreement (range/avg): {dispersion:.2f}

Decision rule suggestion (do not mention thresholds):
- If most methods show meaningful upside, recommend BUY; if mixed, HOLD; if mostly below market, SELL.

Task:
1) Output a clear final rating: BUY / HOLD / SELL (one word).
2) Provide a detailed rationale in 6-10 bullet points.
   - Must explicitly compare each valuation method vs market (DCF, DDM, Multiples).
   - Explain why DCF is higher than Multiples (generic finance reasoning is fine: growth/discount rate assumptions vs market sentiment).
   - Comment on uncertainty using the method disagreement metric.
3) Provide a short "Key Risks" section (3-5 bullets) that could invalidate the valuation.
4) Provide a short "Conclusion" paragraph (2-4 sentences).
Write in English. Use headings. Keep it concise but substantive.
"""

analysis = ask_ollama(prompt, model="llama3", temperature=0.2).strip()

if not analysis:
    analysis = f"""# {reco}

## Rationale
- Market: {market:.2f}
- DCF: {dcf:.2f} ({(dcf/market-1)*100:.1f}% vs market) suggests stronger long-term cash-flow value.
- DDM: {ddm:.2f} ({(ddm/market-1)*100:.1f}% vs market) implies dividend capacity supports upside.
- Multiples: {mult:.2f} ({(mult/market-1)*100:.1f}% vs market) is close to market, reflecting current sentiment/peer pricing.
- Average intrinsic: {avg_intrinsic:.2f} ({up_avg*100:.1f}% upside) / Median intrinsic: {median_intr:.2f} ({up_median*100:.1f}% upside).
- Methods disagree (range/avg ≈ {dispersion:.2f}), so confidence depends on assumptions (growth, WACC, payouts).

## Key Risks
- Commodity price volatility impacts cash flow and dividends.
- Higher discount rates can compress DCF value materially.
- Payout policy changes can weaken DDM.
- Market may continue to price BP at lower multiples vs peers.

## Conclusion
Overall, intrinsic estimates (especially DCF/DDM) point above market, while multiples are nearer current price. Recommendation reflects upside potential balanced by uncertainty from assumption sensitivity.
"""

display(Markdown(analysis))

