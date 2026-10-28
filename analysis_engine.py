
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from market_data_02 import get_ticker, get_candles


SYMBOLS = ["BTCUSD", "ETHUSD"]
REFRESH_SECONDS = 10
MAX_DATA_AGE_SECONDS = 120


def calculate_indicators(candles):
    df = pd.DataFrame(candles)

    if len(df) < 50:
        raise ValueError("Not enough candles")

    df["time"] = pd.to_numeric(df["time"])

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column])

    df = df.sort_values("time").drop_duplicates("time")

    # Exclude the currently forming candle.
    now = time.time()
    df = df[df["time"] + 60 <= now].copy()

    if len(df) < 50:
        raise ValueError("Not enough completed candles")

    # Exponential Moving Averages
    df["ema9"] = df["close"].ewm(
        span=9, adjust=False
    ).mean()

    df["ema21"] = df["close"].ewm(
        span=21, adjust=False
    ).mean()

    # RSI 14 using Wilder smoothing
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    # Average True Range
    previous_close = df["close"].shift(1)

    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = true_range.ewm(
        alpha=1/14, adjust=False
    ).mean()

    # Rolling 50-candle VWAP
    typical_price = (
        df["high"] + df["low"] + df["close"]
    ) / 3

    pv = typical_price * df["volume"]

    volume_sum = df["volume"].rolling(50).sum()

    df["vwap"] = (
        pv.rolling(50).sum()
        / volume_sum.replace(0, np.nan)
    )

    df["avg_volume"] = df["volume"].rolling(20).mean()

    return df


def analyze_market(symbol):
    ticker = get_ticker(symbol)
    candles = get_candles(symbol)

    df = calculate_indicators(candles)
    last = df.iloc[-1]

    now = time.time()

    ticker_timestamp = float(ticker["timestamp"]) / 1_000_000
    candle_timestamp = float(last["time"])

    ticker_age = now - ticker_timestamp
    candle_age = now - (candle_timestamp + 60)

    print("\n" + "=" * 55)
    print("MARKET:", symbol)
    print("PRICE:", ticker["close"])
    print("EMA 9:", round(last["ema9"], 2))
    print("EMA 21:", round(last["ema21"], 2))
    print("RSI:", round(last["rsi"], 2))
    print("VWAP 50:", round(last["vwap"], 2))
    print("ATR:", round(last["atr"], 2))

    print("TICKER AGE:", round(ticker_age, 1), "seconds")
    print("CANDLE AGE:", round(candle_age, 1), "seconds")

    if (
        ticker_age < -30
        or ticker_age > MAX_DATA_AGE_SECONDS
        or candle_age < -30
        or candle_age > MAX_DATA_AGE_SECONDS
    ):
        print("SIGNAL: NO TRADE")
        print("REASON: Stale data or incorrect system clock")
        return

    price = float(ticker["close"])
    ema9 = float(last["ema9"])
    ema21 = float(last["ema21"])
    rsi = float(last["rsi"])
    vwap = float(last["vwap"])
    atr = float(last["atr"])

    avg_volume = float(last["avg_volume"])
    volume = float(last["volume"])

    if not np.isfinite([price, ema9, ema21, rsi,
                        vwap, atr, avg_volume]).all():
        print("SIGNAL: NO TRADE")
        print("REASON: Invalid indicator data")
        return

    volume_strong = (
        avg_volume > 0
        and volume > avg_volume * 1.2
    )

    bullish = (
        price > ema9
        and ema9 > ema21
        and price > vwap
        and 52 < rsi < 70
        and volume_strong
    )

    bearish = (
        price < ema9
        and ema9 < ema21
        and price < vwap
        and 30 < rsi < 48
        and volume_strong
    )

    # Preliminary fixed target for BTC only.
    # ETH needs its own calibrated target.
    if symbol != "BTCUSD":
        print("SIGNAL: ANALYSIS ONLY")
        print("REASON: ETH target not calibrated")
        return

    if atr < 10:
        print("SIGNAL: NO TRADE")
        print("REASON: Insufficient volatility")
        return

    target_points = min(20.0, max(10.0, atr * 0.5))

    if bullish:
        signal = "BUY / LONG"
        target = price + target_points
        stop_loss = price - atr

    elif bearish:
        signal = "SELL / SHORT"
        target = price - target_points
        stop_loss = price + atr

    else:
        print("SIGNAL: NO TRADE")
        print("REASON: Conditions not aligned")
        return

    print("SIGNAL:", signal)
    print("REFERENCE PRICE:", price)
    print("TARGET:", round(target, 2))
    print("STOP LOSS:", round(stop_loss, 2))
    print("TARGET POINTS:", round(target_points, 2))
    print("STATUS: EXPERIMENTAL - PAPER TRADING ONLY")


def main():
    print("DELTA AI - MARKET ANALYSIS ENGINE")
    print("Rule-based signal testing mode")

    while True:
        for symbol in SYMBOLS:
            try:
                analyze_market(symbol)

            except Exception as error:
                print("ERROR:", symbol, str(error))

        print("\nRefreshing in 10 seconds...")
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()
