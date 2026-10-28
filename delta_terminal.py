
import sys
import time
import sqlite3
import requests
import numpy as np
import pandas as pd

from datetime import datetime, timezone

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView,
    QTextEdit, QComboBox
)


API = "https://api.india.delta.exchange"
DB_FILE = "delta_market.db"

SYMBOLS = ["BTCUSD", "ETHUSD"]
REFRESH = 10
TARGET_POINTS = 15.0
MAX_AGE = 120


def number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def init_database():
    connection = sqlite3.connect(DB_FILE)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            symbol TEXT,
            price REAL,
            bid REAL,
            ask REAL,
            spread REAL,
            imbalance REAL,
            buy_volume REAL,
            sell_volume REAL,
            ema9 REAL,
            ema21 REAL,
            rsi REAL,
            atr REAL,
            open_interest REAL,
            funding_rate REAL,
            bias TEXT
        )
    """)

    connection.commit()
    connection.close()


def indicators(candles):
    df = pd.DataFrame(candles)

    if len(df) < 50:
        raise ValueError("Insufficient candles")

    for key in [
        "time", "open", "high",
        "low", "close", "volume"
    ]:
        df[key] = pd.to_numeric(df[key])

    df = df.sort_values("time")
    df = df.drop_duplicates("time")

    now = time.time()

    df = df[
        df["time"] + 60 <= now
    ].copy()

    if len(df) < 50:
        raise ValueError("Insufficient completed candles")

    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["ema9"] = close.ewm(
        span=9, adjust=False
    ).mean()

    df["ema21"] = close.ewm(
        span=21, adjust=False
    ).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    ag = gain.ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    al = loss.ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    rs = ag / al.replace(0, np.nan)

    df["rsi"] = 100 - 100 / (1 + rs)
    df["rsi"] = df["rsi"].fillna(50)

    previous = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - previous).abs(),
        (low - previous).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    last = df.iloc[-1]

    return {
        "ema9": float(last["ema9"]),
        "ema21": float(last["ema21"]),
        "rsi": float(last["rsi"]),
        "atr": float(last["atr"]),
        "candle_time": float(last["time"])
    }


class MarketWorker(QThread):

    updated = Signal(dict)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.session = None

    def get(self, endpoint, params=None):
        response = self.session.get(
            API + endpoint,
            params=params,
            timeout=12
        )

        response.raise_for_status()
        data = response.json()

        if data.get("success") is not True:
            raise RuntimeError(str(data))

        return data["result"]

    def fetch_market(self, symbol):
        ticker = self.get(
            "/v2/tickers/" + symbol
        )

        book = self.get(
            "/v2/l2orderbook/" + symbol,
            {"depth": 15}
        )

        trades_response = self.get(
            "/v2/trades/" + symbol
        )

        end = int(time.time())
        start = end - 120 * 60

        candles = self.get(
            "/v2/history/candles",
            {
                "symbol": symbol,
                "resolution": "1m",
                "start": start,
                "end": end
            }
        )

        stats = indicators(candles)

        bids = book.get("buy", [])
        asks = book.get("sell", [])

        if not bids or not asks:
            raise ValueError("Order book empty")

        bid = number(bids[0]["price"])
        ask = number(asks[0]["price"])

        spread = ask - bid

        if bid <= 0 or ask <= bid:
            raise ValueError("Invalid bid/ask")

        bid_volume = sum(
            number(level.get("size"))
            for level in bids[:15]
        )

        ask_volume = sum(
            number(level.get("size"))
            for level in asks[:15]
        )

        total_depth = bid_volume + ask_volume

        imbalance = (
            (bid_volume - ask_volume) / total_depth
            if total_depth > 0 else 0
        )

        recent_trades = trades_response.get(
            "trades", []
        )

        buy_volume = 0.0
        sell_volume = 0.0

        now = time.time()

        # Trade timestamps are microseconds.
        for trade in recent_trades:
            timestamp = number(
                trade.get("timestamp")
            ) / 1_000_000

            if not 0 <= now - timestamp <= 60:
                continue

            size = number(trade.get("size"))

            if trade.get("side") == "buy":
                buy_volume += size

            elif trade.get("side") == "sell":
                sell_volume += size

        flow_total = buy_volume + sell_volume

        flow = (
            (buy_volume - sell_volume) / flow_total
            if flow_total > 0 else 0
        )

        price = number(ticker.get("close"))

        ticker_time = number(
            ticker.get("timestamp")
        ) / 1_000_000

        ticker_age = now - ticker_time
        candle_age = now - (
            stats["candle_time"] + 60
        )

        book_time = number(
            book.get("last_updated_at")
        ) / 1_000_000

        book_age = (
            now - book_time
            if book_time > 0 else float("inf")
        )

        fresh = (
            -30 <= ticker_age <= MAX_AGE
            and -30 <= candle_age <= MAX_AGE
            and -30 <= book_age <= MAX_AGE
        )

        bullish = (
            price > stats["ema9"]
            and stats["ema9"] > stats["ema21"]
            and 52 < stats["rsi"] < 70
            and imbalance > 0.15
            and flow > 0.10
            and flow_total > 0
        )

        bearish = (
            price < stats["ema9"]
            and stats["ema9"] < stats["ema21"]
            and 30 < stats["rsi"] < 48
            and imbalance < -0.15
            and flow < -0.10
            and flow_total > 0
        )

        bias = "NEUTRAL"

        if fresh and bullish:
            bias = "BULLISH"

        elif fresh and bearish:
            bias = "BEARISH"

        # Do not issue executable signals until
        # model validation and actual costs are known.
        signal = "NO TRADE"

        if not fresh:
            reason = "STALE MARKET DATA"

        elif spread >= TARGET_POINTS * 0.25:
            reason = "SPREAD TOO HIGH"

        elif symbol == "ETHUSD":
            reason = "ETH TARGET NOT CALIBRATED"

        elif bias == "NEUTRAL":
            reason = "NO STRONG ALIGNMENT"

        else:
            reason = "SETUP FOUND - NOT VALIDATED"

        if symbol == "BTCUSD":
            target_points = TARGET_POINTS
        else:
            target_points = None

        target = None
        stop = None

        if target_points and bias == "BULLISH":
            target = ask + target_points
            stop = ask - target_points

        elif target_points and bias == "BEARISH":
            target = bid - target_points
            stop = bid + target_points

        return {
            "timestamp": now,
            "symbol": symbol,
            "price": price,
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "imbalance": imbalance,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "flow": flow,
            "ema9": stats["ema9"],
            "ema21": stats["ema21"],
            "rsi": stats["rsi"],
            "atr": stats["atr"],
            "oi": number(ticker.get("oi")),
            "funding": number(
                ticker.get("funding_rate")
            ),
            "bias": bias,
            "signal": signal,
            "reason": reason,
            "target": target,
            "stop": stop,
            "fresh": fresh
        }

    def save(self, data):
        connection = sqlite3.connect(DB_FILE)

        try:
            connection.execute("""
                INSERT INTO snapshots (
                    timestamp, symbol, price,
                    bid, ask, spread, imbalance,
                    buy_volume, sell_volume,
                    ema9, ema21, rsi, atr,
                    open_interest, funding_rate, bias
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["timestamp"],
                data["symbol"],
                data["price"],
                data["bid"],
                data["ask"],
                data["spread"],
                data["imbalance"],
                data["buy_volume"],
                data["sell_volume"],
                data["ema9"],
                data["ema21"],
                data["rsi"],
                data["atr"],
                data["oi"],
                data["funding"],
                data["bias"]
            ))

            connection.commit()
        finally:
            connection.close()

    def run(self):
        self.session = requests.Session()

        while self.running:
            cycle_start = time.monotonic()

            for symbol in SYMBOLS:
                if not self.running:
                    break

                try:
                    data = self.fetch_market(symbol)
                    self.save(data)
                    self.updated.emit(data)

                except Exception as error:
                    self.failed.emit(
                        symbol + ": " + str(error)
                    )

            while (
                self.running
                and time.monotonic() - cycle_start < REFRESH
            ):
                self.msleep(200)

        self.session.close()

    def stop(self):
        self.running = False


class DeltaTerminal(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "DELTA AI TRADER | Market Intelligence"
        )

        self.resize(1200, 750)

        init_database()

        self.market_data = {}

        self.setup_ui()

        self.worker = MarketWorker()
        self.worker.updated.connect(self.update_market)
        self.worker.failed.connect(self.show_error)
        self.worker.start()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        title = QLabel("DELTA AI TRADER")
        title.setObjectName("title")

        subtitle = QLabel(
            "LIVE MARKET INTELLIGENCE  |  RESEARCH MODE"
        )

        layout.addWidget(title)
        layout.addWidget(subtitle)

        controls = QHBoxLayout()

        controls.addWidget(QLabel("Market:"))

        self.selector = QComboBox()
        self.selector.addItems(SYMBOLS)

        self.selector.currentTextChanged.connect(
            self.refresh_details
        )

        controls.addWidget(self.selector)

        self.status = QLabel("Connecting...")
        controls.addWidget(self.status)

        controls.addStretch()

        layout.addLayout(controls)

        self.table = QTableWidget(2, 9)

        self.table.setHorizontalHeaderLabels([
            "MARKET", "PRICE", "SPREAD",
            "ORDER IMBALANCE", "TRADE FLOW",
            "RSI", "ATR", "BIAS", "SIGNAL"
        ])

        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self.table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )

        layout.addWidget(self.table)

        self.details = QTextEdit()
        self.details.setReadOnly(True)

        layout.addWidget(self.details)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)

        layout.addWidget(QLabel("SYSTEM LOG"))
        layout.addWidget(self.log)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #10141d;
                color: #e8edf7;
                font-family: Segoe UI;
                font-size: 13px;
            }

            QLabel#title {
                font-size: 28px;
                font-weight: bold;
                color: #64e5bd;
            }

            QTableWidget, QTextEdit {
                background-color: #181f2c;
                color: #e8edf7;
                border: 1px solid #303b4d;
                border-radius: 6px;
            }

            QHeaderView::section {
                background-color: #263246;
                color: white;
                padding: 8px;
                border: none;
            }

            QComboBox {
                background-color: #263246;
                color: white;
                padding: 6px;
            }
        """)

    def update_market(self, data):
        symbol = data["symbol"]

        self.market_data[symbol] = data

        row = SYMBOLS.index(symbol)

        values = [
            symbol,
            f'{data["price"]:.2f}',
            f'{data["spread"]:.2f}',
            f'{data["imbalance"] * 100:.1f}%',
            f'{data["flow"] * 100:.1f}%',
            f'{data["rsi"]:.2f}',
            f'{data["atr"]:.2f}',
            data["bias"],
            data["signal"]
        ]

        for column, value in enumerate(values):
            item = QTableWidgetItem(value)

            item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter
            )

            self.table.setItem(
                row, column, item
            )

        self.status.setText(
            "CONNECTED | " +
            datetime.now().strftime("%H:%M:%S")
        )

        if symbol == self.selector.currentText():
            self.refresh_details()

    def refresh_details(self):
        symbol = self.selector.currentText()

        if symbol not in self.market_data:
            self.details.setPlainText(
                "Waiting for market data..."
            )
            return

        d = self.market_data[symbol]

        target = (
            f'{d["target"]:.2f}'
            if d["target"] is not None
            else "N/A"
        )

        stop = (
            f'{d["stop"]:.2f}'
            if d["stop"] is not None
            else "N/A"
        )

        report = f"""
MARKET: {symbol}

LAST PRICE: {d['price']:.2f}
BEST BID: {d['bid']:.2f}
BEST ASK: {d['ask']:.2f}
SPREAD: {d['spread']:.2f}

EMA 9: {d['ema9']:.2f}
EMA 21: {d['ema21']:.2f}
RSI 14: {d['rsi']:.2f}
ATR 14: {d['atr']:.2f}

ORDER BOOK IMBALANCE: {d['imbalance'] * 100:.2f}%
RECENT TRADE FLOW: {d['flow'] * 100:.2f}%
RECENT BUY VOLUME: {d['buy_volume']:.2f}
RECENT SELL VOLUME: {d['sell_volume']:.2f}

OPEN INTEREST: {d['oi']:.4f}
FUNDING RATE (RAW API VALUE): {d['funding']}

MARKET BIAS: {d['bias']}
TRADE SIGNAL: {d['signal']}

HYPOTHETICAL TARGET: {target}
HYPOTHETICAL STOP: {stop}

STATUS: {d['reason']}

NOTE:
Directional bias is rule-based, not an AI forecast.
Targets are hypothetical, not executable orders.
Historical validation and actual costs are pending.
Market snapshots are saved automatically.
"""

        self.details.setPlainText(report)

    def show_error(self, message):
        self.status.setText("DATA ERROR")
        self.log.append(message)

        symbol = message.split(":")[0]

        if symbol in self.market_data:
            self.market_data.pop(symbol)

        if symbol in SYMBOLS:
            row = SYMBOLS.index(symbol)

            for column in range(9):
                self.table.setItem(
                    row,
                    column,
                    QTableWidgetItem(
                        "DATA UNAVAILABLE"
                    )
                )

        self.refresh_details()

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait(15000)
        event.accept()


def main():
    app = QApplication(sys.argv)

    window = DeltaTerminal()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

