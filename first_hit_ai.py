
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

from ai_model import download_data


TARGET = 15.0
STOP = 15.0
HORIZON = 5
THRESHOLD = 0.70

# Illustrative round-trip cost in price points.
# Replace with actual exchange fees and slippage later.
COST_POINTS = 2.0

FEATURES = [
    "return_1", "return_3", "return_5",
    "ema_gap", "ema_slope",
    "rsi", "atr", "volume_ratio",
    "body_ratio", "range_ratio",
    "volatility_10", "momentum_5"
]


def make_features(df):
    parts = []

    for _, segment in df.groupby("segment"):
        s = segment.copy().reset_index(drop=True)

        if len(s) < 100:
            continue

        close = s["close"]
        high = s["high"]
        low = s["low"]
        volume = s["volume"]

        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        ag = gain.ewm(alpha=1/14, adjust=False).mean()
        al = loss.ewm(alpha=1/14, adjust=False).mean()

        rs = ag / al.replace(0, np.nan)
        s["rsi"] = (100 - 100 / (1 + rs)).fillna(50)

        prev = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev).abs(),
            (low - prev).abs()
        ], axis=1).max(axis=1)

        s["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

        s["return_1"] = close.pct_change(1)
        s["return_3"] = close.pct_change(3)
        s["return_5"] = close.pct_change(5)

        s["ema_gap"] = (ema9 - ema21) / close
        s["ema_slope"] = ema9.diff(3) / close

        avg_vol = volume.rolling(20).mean()
        s["volume_ratio"] = volume / avg_vol.replace(0, np.nan)

        rng = high - low

        s["body_ratio"] = (
            (close - s["open"]) / rng.replace(0, np.nan)
        )

        s["range_ratio"] = rng / close

        s["volatility_10"] = (
            s["return_1"].rolling(10).std()
        )

        s["momentum_5"] = close - close.shift(5)

        long_labels = []
        short_labels = []
        long_pnl = []
        short_pnl = []

        # Exit index for non-overlapping test trades.
        long_exits = []
        short_exits = []

        for i in range(len(s)):
            if i + HORIZON >= len(s):
                long_labels.append(np.nan)
                short_labels.append(np.nan)
                long_pnl.append(np.nan)
                short_pnl.append(np.nan)
                long_exits.append(np.nan)
                short_exits.append(np.nan)
                continue

            entry = float(s.iloc[i + 1]["open"])
            future = s.iloc[i + 1:i + HORIZON + 1]

            outcomes = {}

            for direction in ["LONG", "SHORT"]:
                exit_price = float(future.iloc[-1]["close"])
                exit_offset = HORIZON
                won = 0

                for k, (_, candle) in enumerate(
                    future.iterrows(), start=1
                ):
                    h = float(candle["high"])
                    l = float(candle["low"])

                    if direction == "LONG":
                        target_hit = h >= entry + TARGET
                        stop_hit = l <= entry - STOP
                    else:
                        target_hit = l <= entry - TARGET
                        stop_hit = h >= entry + STOP

                    # Conservative stop-first assumption.
                    if stop_hit:
                        exit_price = (
                            entry - STOP if direction == "LONG"
                            else entry + STOP
                        )
                        exit_offset = k
                        break

                    if target_hit:
                        exit_price = (
                            entry + TARGET if direction == "LONG"
                            else entry - TARGET
                        )
                        exit_offset = k
                        won = 1
                        break

                gross = (
                    exit_price - entry
                    if direction == "LONG"
                    else entry - exit_price
                )

                outcomes[direction] = (
                    won, gross, exit_offset
                )

            long_labels.append(outcomes["LONG"][0])
            short_labels.append(outcomes["SHORT"][0])

            long_pnl.append(outcomes["LONG"][1])
            short_pnl.append(outcomes["SHORT"][1])

            long_exits.append(outcomes["LONG"][2])
            short_exits.append(outcomes["SHORT"][2])

        s["long_label"] = long_labels
        s["short_label"] = short_labels

        s["long_pnl"] = long_pnl
        s["short_pnl"] = short_pnl

        s["long_exit"] = long_exits
        s["short_exit"] = short_exits

        parts.append(s)

    data = pd.concat(parts, ignore_index=True)

    data = data.replace([np.inf, -np.inf], np.nan)

    data = data.dropna(
        subset=FEATURES + [
            "long_label", "short_label",
            "long_pnl", "short_pnl"
        ]
    ).reset_index(drop=True)

    return data


def train_and_test(data):
    split = int(len(data) * 0.8)

    test = data.iloc[split:].copy()

    # Purge overlapping future labels from training.
    test_start_time = float(test.iloc[0]["time"])

    train = data.iloc[:split].copy()

    train = train[
        train["time"] + HORIZON * 60 < test_start_time
    ]

    models = {}
    probabilities = {}

    print("\nTraining samples:", len(train))
    print("Testing samples:", len(test))

    for direction in ["LONG", "SHORT"]:
        label = direction.lower() + "_label"

        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_leaf_nodes=15,
            learning_rate=0.05,
            l2_regularization=10,
            min_samples_leaf=100,
            random_state=42
        )

        model.fit(
            train[FEATURES],
            train[label].astype(int)
        )

        prob = model.predict_proba(test[FEATURES])[:, 1]

        models[direction] = model
        probabilities[direction] = prob

        actual = test[label].astype(int)

        print("\n", direction)
        print("Baseline target-first rate:",
              round(actual.mean() * 100, 2), "%")

        if actual.nunique() == 2:
            print("ROC AUC:",
                  round(roc_auc_score(actual, prob), 4))

        print("Brier score:",
              round(brier_score_loss(actual, prob), 4))

    test["p_long"] = probabilities["LONG"]
    test["p_short"] = probabilities["SHORT"]

    trades = []
    i = 0

    while i < len(test):
        row = test.iloc[i]

        p_long = float(row["p_long"])
        p_short = float(row["p_short"])

        if max(p_long, p_short) < THRESHOLD:
            i += 1
            continue

        direction = (
            "LONG" if p_long >= p_short else "SHORT"
        )

        key = direction.lower()

        pnl = float(row[key + "_pnl"])
        exit_offset = int(row[key + "_exit"])

        trades.append({
            "time": row["time"],
            "direction": direction,
            "probability": max(p_long, p_short),
            "target_first": int(row[key + "_label"]),
            "gross_points": pnl,
            "net_points": pnl - COST_POINTS
        })

        # Avoid overlapping positions.
        exit_time = float(row["time"]) + exit_offset * 60

        i += 1

        while (
            i < len(test)
            and float(test.iloc[i]["time"]) <= exit_time
        ):
            i += 1

    print("\n" + "=" * 55)
    print("AI FIRST-HIT TEST RESULTS")
    print("=" * 55)

    print("Confidence threshold:", THRESHOLD)
    print("Target:", TARGET, "points")
    print("Stop:", STOP, "points")
    print("Illustrative cost:", COST_POINTS, "points/trade")

    if not trades:
        print("NO TRADES - Model found no qualifying signals.")
    else:
        results = pd.DataFrame(trades)

        wins = int(results["target_first"].sum())
        total = len(results)

        print("Total trades:", total)
        print("Target-first wins:", wins)
        print("Target-first win rate:",
              round(wins / total * 100, 2), "%")

        print("Gross points:",
              round(results["gross_points"].sum(), 2))

        print("Net points (illustrative costs):",
              round(results["net_points"].sum(), 2))

        print("Average net points:",
              round(results["net_points"].mean(), 2))

        results.to_csv(
            "first_hit_ai_results.csv", index=False
        )

        print("Saved: first_hit_ai_results.csv")

    with open("first_hit_ai_model.pkl", "wb") as f:
        pickle.dump({
            "models": models,
            "features": FEATURES,
            "target": TARGET,
            "stop": STOP,
            "horizon": HORIZON
        }, f)

    print("\nSaved: first_hit_ai_model.pkl")
    print("Model probabilities are not calibrated.")
    print("No live trading is performed.")


def main():
    print("DELTA AI - FIRST HIT MACHINE LEARNING")

    df = download_data()
    data = make_features(df)

    print("Valid samples:", len(data))

    train_and_test(data)


if __name__ == "__main__":
    main()
