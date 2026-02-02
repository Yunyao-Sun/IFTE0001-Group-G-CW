import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta

transaction_cost = 0.002
trading_days = 252

end_date = datetime.today()
start_date = end_date - relativedelta(years=10)
start = start_date.strftime('%Y-%m-%d')
today = datetime.today().strftime('%Y-%m-%d')
data = yf.download("BP.L", start=start, end=today)
data = data[['Close']].dropna()
data['Return'] = data['Close'].pct_change()

def calculate_cagr(cum_return, periods=252):
    years = len(cum_return) / periods
    return cum_return.iloc[-1] ** (1 / years) - 1

def calculate_sharpe(returns, periods=252):
    return np.sqrt(periods) * returns.mean() / returns.std()

def calculate_max_drawdown(cum_return):
    rolling_max = cum_return.cummax()
    drawdown = cum_return / rolling_max - 1
    return drawdown.min()

def calculate_hit_rate(net_return, signal):
    active_day = signal != 0
    return (net_return[active_day] > 0).mean()

def backtest_ma(data, short_window, long_window, cost):
    df = data.copy()

    df['MA_S'] = df['Close'].rolling(short_window).mean()
    df['MA_L'] = df['Close'].rolling(long_window).mean()

    df['Signal'] = np.where(df['MA_S'] > df['MA_L'], 1, -1)
    df['Position_Change'] = df['Signal'].diff().abs()

    df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']
    df['Cost'] = df['Position_Change'] * cost
    df['Net_Return'] = df['Strategy_Return'] - df['Cost']

    df['Cum_Strategy'] = (1 + df['Net_Return']).cumprod()

    return df.dropna()

def backtest_macd(data, short, long, signal_window, cost):
    df = data.copy()

    ema_s = df['Close'].ewm(span=short, adjust=False).mean()
    ema_l = df['Close'].ewm(span=long, adjust=False).mean()

    df['MACD'] = ema_s - ema_l
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()

    df['Signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
    df['Position_Change'] = df['Signal'].diff().abs()

    df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']
    df['Cost'] = df['Position_Change'] * cost
    df['Net_Return'] = df['Strategy_Return'] - df['Cost']

    df['Cum_Strategy'] = (1 + df['Net_Return']).cumprod()

    return df.dropna()

def backtest_rsi(data, window, lower, upper, cost):
    df = data.copy()

    #calculate delta
    close = df['Close'].squeeze()

    delta = close.diff()

    # gain and loss
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Trading signal
    df['Signal'] = np.where(
        df['RSI'] < lower, 1,
        np.where(df['RSI'] > upper, -1, np.nan)
    )
    df['Signal'] = df['Signal'].ffill()

    # Strategy return
    df['Position_Change'] = df['Signal'].diff().abs()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']
    df['Cost'] = df['Position_Change'] * cost
    df['Net_Return'] = df['Strategy_Return'] - df['Cost']
    df['Cum_Strategy'] = (1 + df['Net_Return']).cumprod()

    return df.dropna()

ma_params = {
    'short': [5, 10, 20, 30, 60, 100, 120],
    'long':  [20, 30, 60, 100, 120, 150, 250]
}

macd_params = {
    'short': [8, 12, 16],
    'long':  [20, 26, 35],
    'signal': [6, 9]
}

rsi_params = {
    'window': [10, 14, 20],
    'lower': [25, 30, 35],
    'upper': [65, 70, 75]
}

results = []

for s in ma_params['short']:
    for l in ma_params['long']:
        if s >= l:
            continue

        bt = backtest_ma(data, s, l, transaction_cost)

        results.append({
            'Strategy': 'MA',
            'Params': f'S={s}, L={l}',
            'CAGR': calculate_cagr(bt['Cum_Strategy']),
            'Sharpe': calculate_sharpe(bt['Net_Return']),
            'MaxDD': calculate_max_drawdown(bt['Cum_Strategy']),
            'HitRate': calculate_hit_rate(bt['Net_Return'], bt['Signal'])
        })

for s in macd_params['short']:
    for l in macd_params['long']:
        if s >= l:
            continue
        for sig in macd_params['signal']:

            bt = backtest_macd(data, s, l, sig, transaction_cost)

            results.append({
                'Strategy': 'MACD',
                'Params': f'{s},{l},{sig}',
                'CAGR': calculate_cagr(bt['Cum_Strategy']),
                'Sharpe': calculate_sharpe(bt['Net_Return']),
                'MaxDD': calculate_max_drawdown(bt['Cum_Strategy']),
                'HitRate': calculate_hit_rate(bt['Net_Return'], bt['Signal'])
            })

for w in rsi_params['window']:
    for low in rsi_params['lower']:
        for up in rsi_params['upper']:
            if low >= up:
                continue

            bt = backtest_rsi(data, w, low, up, transaction_cost)

            results.append({
                'Strategy': 'RSI',
                'Params': f'{w},{low},{up}',
                'CAGR': calculate_cagr(bt['Cum_Strategy']),
                'Sharpe': calculate_sharpe(bt['Net_Return']),
                'MaxDD': calculate_max_drawdown(bt['Cum_Strategy']),
                'HitRate': calculate_hit_rate(bt['Net_Return'], bt['Signal'])
            })

results_df = pd.DataFrame(results)

best_overall = results_df.sort_values(
    by='CAGR', ascending=False
).iloc[0]

print("Best Overall Strategy:")
print(best_overall)

# Buy & Hold
data['BH_Return'] = data['Return']
data['BH_Cum'] = (1 + data['BH_Return']).cumprod()

bh_cagr = calculate_cagr(data['BH_Cum'].dropna())
bh_sharpe = calculate_sharpe(data['BH_Return'].dropna())
bh_mdd = calculate_max_drawdown(data['BH_Cum'].dropna())

# Best MA
best_ma = results_df[results_df['Strategy'] == 'MA'] \
    .sort_values('CAGR', ascending=False).iloc[0]

# Best MACD
best_macd = results_df[results_df['Strategy'] == 'MACD'] \
    .sort_values('CAGR', ascending=False).iloc[0]

# Best RSI
best_rsi = results_df[results_df['Strategy'] == 'RSI'] \
    .sort_values('CAGR', ascending=False).iloc[0]

s_ma, l_ma = map(int, best_ma['Params'].replace('S=', '').replace('L=', '').split(','))
bt_ma = backtest_ma(data, s_ma, l_ma, transaction_cost)

s_m, l_m, sig_m = map(int, best_macd['Params'].split(','))
bt_macd = backtest_macd(data, s_m, l_m, sig_m, transaction_cost)

w_r, low_r, up_r = map(int, best_rsi['Params'].split(','))
bt_rsi = backtest_rsi(data, w_r, low_r, up_r, transaction_cost)

comparison = pd.DataFrame({
    'Strategy': ['Buy & Hold', 'Optimised MA', 'Optimised MACD', 'Optimised RSI'],
    'CAGR': [
        bh_cagr,
        calculate_cagr(bt_ma['Cum_Strategy']),
        calculate_cagr(bt_macd['Cum_Strategy']),
        calculate_cagr(bt_rsi['Cum_Strategy'])
    ],
    'Sharpe Ratio': [
        bh_sharpe,
        calculate_sharpe(bt_ma['Net_Return']),
        calculate_sharpe(bt_macd['Net_Return']),
        calculate_sharpe(bt_rsi['Net_Return'])
    ],
    'Max Drawdown': [
        bh_mdd,
        calculate_max_drawdown(bt_ma['Cum_Strategy']),
        calculate_max_drawdown(bt_macd['Cum_Strategy']),
        calculate_max_drawdown(bt_rsi['Cum_Strategy'])
    ],
    'Hit Rate': [
        np.nan,
        calculate_hit_rate(bt_ma['Net_Return'], bt_ma['Signal']),
        calculate_hit_rate(bt_macd['Net_Return'], bt_macd['Signal']),
        calculate_hit_rate(bt_rsi['Net_Return'], bt_rsi['Signal'])
    ]
})

print(comparison)

import matplotlib.pyplot as plt

plt.figure(figsize=(11, 6))

plt.plot(data.index, data['BH_Cum'],
         label='Buy & Hold', linewidth=2, linestyle='--')

plt.plot(bt_ma.index, bt_ma['Cum_Strategy'],
         label='Optimised MA', linewidth=2)

plt.plot(bt_macd.index, bt_macd['Cum_Strategy'],
         label='Optimised MACD', linewidth=2)

plt.plot(bt_rsi.index, bt_rsi['Cum_Strategy'],
         label='Optimised RSI', linewidth=2)

plt.title('Cumulative Return Comparison: Buy & Hold vs Trading Strategies')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()

import pandas as pd

results = pd.DataFrame({
    "Strategy": ["Buy & Hold", "Optimised MA", "Optimised MACD", "Optimised RSI"],
    "CAGR": [0.024246, 0.033594, -0.031111, 0.099188],
    "Sharpe Ratio": [0.231730, 0.393589, 0.053135, 0.463693],
    "Max Drawdown": [-0.676256, -0.455342, -0.644517, -0.582847],
    "Hit Rate": [None, 0.505051, 0.486733, 0.503650]
})

def build_llm_input(df):
    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"""
Strategy: {row['Strategy']}
- CAGR: {row['CAGR']:.2%}
- Sharpe Ratio: {row['Sharpe Ratio']:.3f}
- Max Drawdown: {row['Max Drawdown']:.2%}
- Hit Rate: {"N/A" if pd.isna(row['Hit Rate']) else f"{row['Hit Rate']:.2%}"}
"""
        )
    return "\n".join(lines)

llm_input = build_llm_input(results)
print(llm_input)

def build_prompt(llm_input):
    prompt = f"""
You are acting as a buy-side quantitative analyst at an asset management firm.

Based strictly on the performance results provided below, generate a 1–2 page professional trade report.
Do NOT invent new data or modify any metrics.
Use a neutral, analytical tone suitable for an internal investment committee.

The report should include:
1. Executive summary
2. Description of evaluated strategies
3. Comparative performance analysis using CAGR, Sharpe ratio, maximum drawdown, and hit rate
4. Risk and robustness discussion
5. Overall strategy assessment and recommendation

Performance results:
{llm_input}
"""
    return prompt

from openai import OpenAI
import os

client = OpenAI(api_key='api link')
def generate_report_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional financial analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content


prompt = build_prompt(llm_input)
report = generate_report_openai(prompt)

print(report)

with open("LLM_Trade_Report.txt", "w", encoding="utf-8") as f:
    f.write(report)



