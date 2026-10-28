
import time
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from market_data_02 import api_get


SYMBOL = "BTCUSD"
DAYS = 30
HORIZON = 5
TARGET_POINTS = 15
CONFIDENCE_THRESHOLD = 0.75

FEATURES = [
    "return_1",
    "return_3",
    "return_5",
    "ema_gap",
    "ema_slope",
    "rsi",
    "atr",
    "volume_ratio",
    "range_ratio",
    "body_ratio",
    "momentum_5",
    "volatility_10"
]


def download_data():
    print("Downloading 30 days of BTCUSD candles...")

    end_time = int(time.time())
    end_time -= end_time % 60

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

                print("Retrying:", error)
                time.sleep(2)

        current = batch_end
        time.sleep(0.2)

    df = pd.DataFrame(all_candles)

    if df.empty:
        raise RuntimeError("No candle data received")

    df["time"] = pd.to_numeric(df["time"])

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column])

    df = df.sort_values("time")
    df = df.drop_duplicates("time")
    df = df.reset_index(drop=True)

    # Remove incomplete and future candles.
    now = time.time()
    df = df[df["time"] + 60 <= now].copy()

    # Keep only continuous one-minute sequences.
    df["segment"] = (
        df["time"].diff().fillna(60) != 60
    ).cumsum()

    print("Candles downloaded:", len(df))

    return df


def create_features(df):
    result = []

    for _, segment in df.groupby("segment"):
        segment = segment.copy()

        if len(segment) < 100:
            continue

        close = segment["close"]
        high = segment["high"]
        low = segment["low"]
        volume = segment["volume"]

        ema9 = close.ewm(
            span=9, adjust=False
        ).mean()

        ema21 = close.ewm(
            span=21, adjust=False
        ).mean()

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1/14, adjust=False
        ).mean()

        avg_loss = loss.ewm(
            alpha=1/14, adjust=False
        ).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)

        segment["rsi"] = (
            100 - 100 / (1 + rs)
        ).fillna(50)

        previous_close = close.shift(1)

        true_range = pd.concat([
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ], axis=1).max(axis=1)

        segment["atr"] = true_range.ewm(
            alpha=1/14, adjust=False
        ).mean()

        segment["return_1"] = close.pct_change(1)
        segment["return_3"] = close.pct_change(3)
        segment["return_5"] = close.pct_change(5)

        segment["ema_gap"] = (
            ema9 - ema21
        ) / close

        segment["ema_slope"] = ema9.diff(3) / close

        average_volume = volume.rolling(20).mean()

        segment["volume_ratio"] = (
            volume / average_volume.replace(0, np.nan)
        )

        candle_range = (high - low).replace(0, np.nan)

        segment["range_ratio"] = (
            (high - low) / close
        )

        segment["body_ratio"] = (
            (close - segment["open"]) / candle_range
        )

        segment["momentum_5"] = close - close.shift(5)

        segment["volatility_10"] = (
            segment["return_1"].rolling(10).std()
        )

        # Future outcome measured from the next candle open.
        entry = segment["open"].shift(-1)

        future_highs = pd.concat(
            [high.shift(-i) for i in range(1, HORIZON + 1)],
            axis=1
        )

        future_lows = pd.concat(
            [low.shift(-i) for i in range(1, HORIZON + 1)],
            axis=1
        )

        future_high = future_highs.max(
            axis=1, skipna=False
        )

        future_low = future_lows.min(
            axis=1, skipna=False
        )

        up_hit = future_high >= entry + TARGET_POINTS
        down_hit = future_low <= entry - TARGET_POINTS

        # Label only unambiguous outcomes.
        #  1 = upward target only
        # -1 = downward target only
        #  0 = neither target
        # Rows hitting both directions are excluded.
        segment["target"] = np.select(
            [
                up_hit & ~down_hit,
                down_hit & ~up_hit
            ],
            [1, -1],
            default=0
        )

        segment["valid_label"] = (
            future_high.notna()
            & future_low.notna()
            & ~(up_hit & down_hit)
        )

        segment = segment[
            segment["valid_label"]
        ].copy()

        result.append(segment)

    if not result:
        raise RuntimeError("Insufficient continuous data")

    data = pd.concat(result, ignore_index=True)

    data = data.replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.dropna(
        subset=FEATURES + ["target"]
    )

    return data


def train_model(data):
    print("\nPreparing chronological train/test split...")

    split = int(len(data) * 0.8)

    # Purge samples whose 5-minute outcomes overlap
    # the test period.
    train = data.iloc[:max(0, split - HORIZON)].copy()
    test = data.iloc[split:].copy()

    if train.empty or test.empty:
        raise RuntimeError("Not enough training/test data")

    X_train = train[FEATURES]
    y_train = train["target"].astype(int)

    X_test = test[FEATURES]
    y_test = test["target"].astype(int)

    print("Training samples:", len(train))
    print("Testing samples:", len(test))

    print("\nTraining Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=30,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n" + "=" * 55)
    print("AI MODEL - OUT OF SAMPLE RESULTS")
    print("=" * 55)

    print(classification_report(
        y_test,
        predictions,
        labels=[-1, 0, 1],
        target_names=["DOWN", "NO MOVE", "UP"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_test, predictions, labels=[-1, 0, 1]
    ))

    probabilities = model.predict_proba(X_test)

    classes = model.classes_
    selected = []

    for i in range(len(test)):
        best_index = np.argmax(probabilities[i])
        confidence = probabilities[i][best_index]
        predicted_class = classes[best_index]

        if (
            confidence >= CONFIDENCE_THRESHOLD
            and predicted_class != 0
        ):
            selected.append({
                "time": test.iloc[i]["time"],
                "prediction": int(predicted_class),
                "actual": int(y_test.iloc[i]),
                "confidence": float(confidence)
            })

    signals = pd.DataFrame(selected)

    print("\n" + "=" * 55)
    print("HIGH CONFIDENCE SIGNAL RESULTS")
    print("=" * 55)

    if signals.empty:
        print("No signals above confidence threshold.")
    else:
        correct = (
            signals["prediction"] == signals["actual"]
        ).sum()

        accuracy = correct / len(signals) * 100

        print("Signals:", len(signals))
        print("Correct:", correct)
        print("Directional label accuracy:",
              round(accuracy, 2), "%")
        print("Average confidence:",
              round(signals["confidence"].mean() * 100, 2),
              "%")

        signals.to_csv(
            "ai_test_signals.csv",
            index=False
        )

        print("Saved: ai_test_signals.csv")

    with open("btc_ai_model.pkl", "wb") as file:
        pickle.dump({
            "model": model,
            "features": FEATURES,
            "symbol": SYMBOL,
            "horizon": HORIZON,
            "target_points": TARGET_POINTS,
            "trained_at": datetime.now(
                timezone.utc
            ).isoformat()
        }, file)

    print("\nModel saved: btc_ai_model.pkl")
    print("NOTE: Class probabilities are not calibrated.")
    print("NOTE: This is not a trade-profit backtest.")


def main():
    print("DELTA AI - MACHINE LEARNING ENGINE")

    df = download_data()
    data = create_features(df)

    print("Usable labeled samples:", len(data))

    train_model(data)


if __name__ == "__main__":
    main()
