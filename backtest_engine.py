
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from market_data_02 import api_get
from analysis_engine import calculate_indicators


SYMBOL = "BTCUSD"
DAYS = 7
RESOLUTION = 60
MAX_HOLD_CANDLES = 5


def download_history():
    print("Downloading historical BTCUSD data...")

    end_time = int(time.time())
    end_time = end_time - (end_time % 60)

    start_time = end_time - DAYS * 86400

    all_candles = []
    current = start_time

    while current < end_time:
        batch_end = min(current + 21600, end_time)

        params = {
            "symbol": SYMBOL,
            "resolution": "1m",
            "start": current,
            "end": batch_end
        }

        for attempt in range(3):
            try:
                candles = api_get(
                    "/v2/history/candles",
                    params
                )
                all_candles.extend(candles)
                break

            except Exception as error:
                if attempt == 2:
                    raise error

                time.sleep(2)

        current = batch_end
        time.sleep(0.25)

    df = pd.DataFrame(all_candles)

    if df.empty:
        raise RuntimeError("No historical data received")

    df = calculate_indicators(all_candles)
    df = df.reset_index(drop=True)

    print("Candles downloaded:", len(df))

    return df


def get_signal(row):
    price = float(row["close"])
    ema9 = float(row["ema9"])
    ema21 = float(row["ema21"])
    rsi = float(row["rsi"])
    vwap = float(row["vwap"])
    atr = float(row["atr"])
    volume = float(row["volume"])
    avg_volume = float(row["avg_volume"])

    values = [
        price, ema9, ema21,
        rsi, vwap, atr, avg_volume
    ]

    if not np.isfinite(values).all():
        return None

    if atr < 10 or avg_volume <= 0:
        return None

    volume_strong = volume > avg_volume * 1.2

    if not volume_strong:
        return None

    if (
        price > ema9
        and ema9 > ema21
        and price > vwap
        and 52 < rsi < 70
    ):
        return "LONG"

    if (
        price < ema9
        and ema9 < ema21
        and price < vwap
        and 30 < rsi < 48
    ):
        return "SHORT"

    return None


def run_backtest(df):
    trades = []
    index = 50

    while index < len(df) - 1:
        signal_row = df.iloc[index]
        signal = get_signal(signal_row)

        if signal is None:
            index += 1
            continue

        entry_index = index + 1
        entry_row = df.iloc[entry_index]

        # Enter at the next candle open.
        entry = float(entry_row["open"])
        atr = float(signal_row["atr"])

        target_points = min(
            20.0, max(10.0, atr * 0.5)
        )

        if signal == "LONG":
            target = entry + target_points
            stop = entry - atr
        else:
            target = entry - target_points
            stop = entry + atr

        result = "TIMEOUT"
        exit_price = None
        exit_index = entry_index

        last_index = min(
            entry_index + MAX_HOLD_CANDLES - 1,
            len(df) - 1
        )

        for j in range(entry_index, last_index + 1):
            candle = df.iloc[j]

            high = float(candle["high"])
            low = float(candle["low"])

            if signal == "LONG":
                hit_target = high >= target
                hit_stop = low <= stop
            else:
                hit_target = low <= target
                hit_stop = high >= stop

            # Conservative assumption:
            # if both levels hit in one candle,
            # count stop-loss first.
            if hit_stop:
                exit_price = stop
                result = "LOSS"
                exit_index = j
                break

            if hit_target:
                exit_price = target
                result = "WIN"
                exit_index = j
                break

        if exit_price is None:
            exit_index = last_index
            exit_price = float(
                df.iloc[exit_index]["close"]
            )

        if signal == "LONG":
            pnl = exit_price - entry
        else:
            pnl = entry - exit_price

        if result == "TIMEOUT":
            if pnl > 0:
                result = "TIMEOUT PROFIT"
            elif pnl < 0:
                result = "TIMEOUT LOSS"
            else:
                result = "BREAKEVEN"

        trades.append({
            "time": datetime.fromtimestamp(
                float(signal_row["time"]),
                tz=timezone.utc
            ).isoformat(),
            "signal": signal,
            "entry": entry,
            "target": target,
            "stop": stop,
            "exit": exit_price,
            "result": result,
            "gross_points": pnl
        })

        # Do not overlap positions.
        index = exit_index + 1

    return pd.DataFrame(trades)


def show_results(trades):
    if trades.empty:
        print("No trades found.")
        return

    total = len(trades)
    wins = int((trades["gross_points"] > 0).sum())
    losses = int((trades["gross_points"] < 0).sum())

    win_rate = wins / total * 100
    gross_points = trades["gross_points"].sum()

    equity = trades["gross_points"].cumsum()
    running_peak = equity.cummax().clip(lower=0)
    drawdown = running_peak - equity

    max_drawdown = drawdown.max()

    consecutive = 0
    max_consecutive = 0

    for pnl in trades["gross_points"]:
        if pnl < 0:
            consecutive += 1
            max_consecutive = max(
                max_consecutive, consecutive
            )
        else:
            consecutive = 0

    print("\n" + "=" * 50)
    print("DELTA AI - BACKTEST RESULTS")
    print("=" * 50)

    print("Market:", SYMBOL)
    print("Period:", DAYS, "days")
    print("Total trades:", total)
    print("Profitable trades:", wins)
    print("Losing trades:", losses)
    print("Win rate:", round(win_rate, 2), "%")
    print("Gross points:", round(gross_points, 2))
    print("Average points/trade:",
          round(gross_points / total, 2))
    print("Maximum drawdown (points):",
          round(max_drawdown, 2))
    print("Maximum consecutive losses:",
          max_consecutive)

    print("\nNOTE:")
    print("Fees, slippage and funding are excluded.")
    print("Points are not account profit in USD.")
    print("Intrabar order is unknown.")
    print("This is not verified trading accuracy.")

    if win_rate >= 95:
        print("\nObserved win rate >= 95%.")
        print("Independent testing is still required.")
    else:
        print("\n95% win-rate target not achieved.")

    trades.to_csv(
        "backtest_results.csv",
        index=False
    )

    print("\nSaved: backtest_results.csv")


def main():
    print("DELTA AI BACKTEST ENGINE")

    df = download_history()
    trades = run_backtest(df)
    show_results(trades)


if __name__ == "__main__":
    main()
