
import requests
import time
from datetime import datetime, timezone

BASE_URL = "https://api.india.delta.exchange"
SYMBOLS = ["BTCUSD", "ETHUSD"]


def api_get(endpoint, params=None):
    response = requests.get(
        BASE_URL + endpoint,
        params=params,
        headers={"Accept": "application/json"},
        timeout=15
    )

    response.raise_for_status()
    data = response.json()

    if data.get("success") is not True:
        raise RuntimeError(str(data))

    return data["result"]


def get_ticker(symbol):
    return api_get("/v2/tickers/" + symbol)


def get_candles(symbol):
    end_time = int(time.time())
    start_time = end_time - (110 * 60)

    params = {
        "symbol": symbol,
        "resolution": "1m",
        "start": start_time,
        "end": end_time
    }

    candles = api_get("/v2/history/candles", params)

    return sorted(
        candles,
        key=lambda candle: candle["time"]
    )


def show_market(symbol):
    ticker = get_ticker(symbol)
    candles = get_candles(symbol)

    print("\n" + "=" * 50)
    print("MARKET:", symbol)
    print("LAST PRICE:", ticker.get("close"))
    print("MARK PRICE:", ticker.get("mark_price"))
    print("SPOT PRICE:", ticker.get("spot_price"))
    print("OPEN INTEREST:", ticker.get("oi"))
    print("24H VOLUME:", ticker.get("volume"))
    print("CANDLES RECEIVED:", len(candles))

    print("\nLAST 5 COMPLETED CANDLES")

    for candle in candles[-6:-1]:
        timestamp = datetime.fromtimestamp(
            int(candle["time"]),
            tz=timezone.utc
        ).strftime("%H:%M")

        print(
            timestamp,
            "O:", candle["open"],
            "H:", candle["high"],
            "L:", candle["low"],
            "C:", candle["close"],
            "V:", candle["volume"]
        )


def main():
    print("DELTA AI MARKET ANALYZER")
    print("Connecting to Delta Exchange India...")

    while True:
        for symbol in SYMBOLS:
            try:
                show_market(symbol)

            except (
                requests.RequestException,
                ValueError,
                RuntimeError,
                KeyError,
                TypeError
            ) as error:
                print(
                    "\nERROR FOR",
                    symbol,
                    ":",
                    error
                )

        print("\nRefreshing in 10 seconds...")
        time.sleep(10)


if __name__ == "__main__":
    main()
