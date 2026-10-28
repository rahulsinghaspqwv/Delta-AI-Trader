
from ai_model import download_data

TARGET = 15.0
STOP = 15.0
HORIZON = 5


def main():
    print("DELTA AI - FIRST HIT ANALYSIS")

    df = download_data()

    results = {
        "LONG": {"WIN": 0, "LOSS": 0, "TIMEOUT": 0},
        "SHORT": {"WIN": 0, "LOSS": 0, "TIMEOUT": 0}
    }

    ambiguous = 0
    evaluated = 0

    for _, segment in df.groupby("segment"):
        segment = segment.reset_index(drop=True)

        for i in range(len(segment) - HORIZON):
            entry = float(segment.iloc[i + 1]["open"])

            future = segment.iloc[
                i + 1:i + HORIZON + 1
            ]

            evaluated += 1

            for direction in ["LONG", "SHORT"]:
                outcome = "TIMEOUT"

                for _, candle in future.iterrows():
                    high = float(candle["high"])
                    low = float(candle["low"])

                    if direction == "LONG":
                        target_hit = high >= entry + TARGET
                        stop_hit = low <= entry - STOP
                    else:
                        target_hit = low <= entry - TARGET
                        stop_hit = high >= entry + STOP

                    if target_hit and stop_hit:
                        ambiguous += 1
                        outcome = "LOSS"
                        break

                    if stop_hit:
                        outcome = "LOSS"
                        break

                    if target_hit:
                        outcome = "WIN"
                        break

                results[direction][outcome] += 1

    print("\n" + "=" * 48)
    print("FIRST-HIT BASELINE RESULTS")
    print("=" * 48)

    print("Evaluated entry times:", evaluated)
    print("Ambiguous direction-cases:", ambiguous)

    for direction, counts in results.items():
        wins = counts["WIN"]
        losses = counts["LOSS"]
        timeouts = counts["TIMEOUT"]

        total = wins + losses + timeouts

        win_rate = 100 * wins / total if total else 0

        print("\nDIRECTION:", direction)
        print("Wins:", wins)
        print("Losses:", losses)
        print("Timeouts:", timeouts)
        print("Target hit rate:", round(win_rate, 2), "%")

    print("\nThis is a baseline, not an AI backtest.")
    print("Every eligible minute is evaluated independently.")
    print("Overlapping entries are not executable trades.")
    print("Fees and slippage are not included.")


if __name__ == "__main__":
    main()
