
import numpy as np
import pandas as pd

from ai_model import (
    download_data,
    create_features,
    TARGET_POINTS,
    HORIZON
)


def main():
    print("DELTA AI - DATA DIAGNOSTICS")

    df = download_data()

    total = len(df)
    both_count = 0
    up_only = 0
    down_only = 0
    neither = 0
    incomplete = 0

    for _, segment in df.groupby("segment"):
        segment = segment.reset_index(drop=True)

        for i in range(len(segment)):
            if i + HORIZON >= len(segment):
                incomplete += 1
                continue

            entry = float(segment.iloc[i + 1]["open"])

            future = segment.iloc[
                i + 1:i + HORIZON + 1
            ]

            up_hit = (
                future["high"].max()
                >= entry + TARGET_POINTS
            )

            down_hit = (
                future["low"].min()
                <= entry - TARGET_POINTS
            )

            if up_hit and down_hit:
                both_count += 1
            elif up_hit:
                up_only += 1
            elif down_hit:
                down_only += 1
            else:
                neither += 1

    print("\n" + "=" * 45)
    print("DATA QUALITY REPORT")
    print("=" * 45)

    print("Total candles:", total)
    print("UP only:", up_only)
    print("DOWN only:", down_only)
    print("Both directions:", both_count)
    print("Neither direction:", neither)
    print("Incomplete:", incomplete)

    print("\nPercentages:")

    for name, value in [
        ("UP only", up_only),
        ("DOWN only", down_only),
        ("Both", both_count),
        ("Neither", neither),
        ("Incomplete", incomplete)
    ]:
        print(
            name + ":",
            round(value / total * 100, 2),
            "%"
        )

    labeled = create_features(df)

    print("\nUsable model samples:", len(labeled))
    print("Excluded candles:",
          total - len(labeled))


if __name__ == "__main__":
    main()
