import time
import unittest
from unittest import mock

import market_data_sources as mds


class _Response:
    def __init__(self, payload, status_code=200, ok=True):
        self.payload = payload
        self.status_code = status_code
        self.ok = ok

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class KrakenFetchTests(unittest.TestCase):
    def setUp(self):
        mds.KRAKEN_KLINE_CACHE.clear()
        mds._KRAKEN_LAST_REQUEST_TS["ts"] = 0.0

    def test_parses_ohlc_rows_and_caches(self):
        payload = {
            "error": [],
            "result": {
                "XXBTZUSD": [
                    [1700000000, "100", "101", "99", "100.5", "100.2", "10", 5],
                    [1700000060, "100.5", "102", "100", "101.5", "101.0", "20", 8],
                ],
                "last": 1700000060,
            },
        }
        with mock.patch.object(mds.requests, "get", return_value=_Response(payload)) as get:
            rows = mds.fetch_kraken_klines("BTCUSDT", "1m", limit=10)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], 1700000000000)
        self.assertEqual(rows[0][4], "100.5")
        get.assert_called_once()
        self.assertIn(("XBTUSD", "1m", 0, 0), mds.KRAKEN_KLINE_CACHE)

    def test_empty_result_raises(self):
        payload = {"error": [], "result": {"XXBTZUSD": [], "last": 0}}
        with mock.patch.object(mds.requests, "get", return_value=_Response(payload)):
            with self.assertRaises(RuntimeError):
                mds.fetch_kraken_klines("BTCUSDT", "1m", limit=10)

    def test_unsupported_interval_raises_without_request(self):
        with mock.patch.object(mds.requests, "get") as get:
            with self.assertRaises(RuntimeError):
                mds.fetch_kraken_klines("BTCUSDT", "2h", limit=10)
        get.assert_not_called()

    def test_enforces_minimum_request_gap(self):
        payload = {"error": [], "result": {"XXBTZUSD": [[1700000000, "1", "1", "1", "1", "1", "1", 1]], "last": 0}}
        sleeps = []
        with (
            mock.patch.object(mds.requests, "get", return_value=_Response(payload)),
            mock.patch.object(mds.time, "sleep", side_effect=sleeps.append),
            mock.patch.dict(mds.os.environ, {"KRAKEN_REQUEST_MIN_GAP_SEC": "5"}, clear=False),
        ):
            mds._KRAKEN_LAST_REQUEST_TS["ts"] = time.time()
            mds.fetch_kraken_klines("BTCUSDT", "5m", limit=10)
        self.assertTrue(sleeps, "expected fetch_kraken_klines to wait out the configured min gap")
        self.assertGreater(sleeps[0], 0)


class CoinbaseFetchTests(unittest.TestCase):
    def test_parses_candles_oldest_first(self):
        payload = [
            [1700000060, 99.0, 101.0, 100.5, 100.8, 12.0],
            [1700000000, 98.0, 100.0, 99.0, 99.5, 10.0],
        ]
        with mock.patch.object(mds.requests, "get", return_value=_Response(payload)):
            rows = mds.fetch_coinbase_klines("BTCUSDT", "1h", limit=10)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], 1700000000000)
        self.assertEqual(rows[1][0], 1700000060000)

    def test_unsupported_interval_raises(self):
        with self.assertRaises(RuntimeError):
            mds.fetch_coinbase_klines("BTCUSDT", "12h", limit=10)


class TwelveDataFetchTests(unittest.TestCase):
    def setUp(self):
        mds.TWELVE_DATA_KLINE_CACHE.clear()
        mds.TWELVE_DATA_USAGE_STATE.update({"day": "", "count": 0, "last_request_ts": 0.0})

    def test_missing_api_key_raises_without_request(self):
        with (
            mock.patch.dict(mds.os.environ, {"TWELVE_DATA_API_KEY": ""}, clear=False),
            mock.patch.object(mds.requests, "get") as get,
        ):
            with self.assertRaises(RuntimeError):
                mds.fetch_twelve_data_klines("BTCUSDT", "1h", limit=10)
        get.assert_not_called()

    def test_parses_values_and_caches(self):
        payload = {
            "status": "ok",
            "values": [
                {"datetime": "2023-11-14 00:00:00", "open": "100", "high": "101", "low": "99", "close": "100.5", "volume": "10"},
            ],
        }
        with (
            mock.patch.dict(mds.os.environ, {"TWELVE_DATA_API_KEY": "test-key", "TWELVE_DATA_REQUEST_MIN_GAP_SEC": "0"}, clear=False),
            mock.patch.object(mds.requests, "get", return_value=_Response(payload)) as get,
            mock.patch.object(mds, "save_twelve_data_usage_state"),
        ):
            rows = mds.fetch_twelve_data_klines("BTCUSDT", "1d", limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "100")
        get.assert_called_once()
        self.assertEqual(mds.TWELVE_DATA_USAGE_STATE["count"], 1)

    def test_daily_limit_blocks_further_requests(self):
        with (
            mock.patch.dict(
                mds.os.environ,
                {"TWELVE_DATA_API_KEY": "test-key", "TWELVE_DATA_DAILY_REQUEST_LIMIT": "1"},
                clear=False,
            ),
            mock.patch.object(mds.requests, "get") as get,
            mock.patch.object(mds, "save_twelve_data_usage_state"),
        ):
            today = mds.datetime.datetime.now(mds.datetime.timezone.utc).strftime("%Y-%m-%d")
            mds.TWELVE_DATA_USAGE_STATE.update({"day": today, "count": 1})
            with self.assertRaises(RuntimeError):
                mds.fetch_twelve_data_klines("BTCUSDT", "1h", limit=10)
        get.assert_not_called()

    def test_enforces_minimum_request_gap(self):
        payload = {
            "status": "ok",
            "values": [
                {"datetime": "2023-11-14 00:00:00", "open": "100", "high": "101", "low": "99", "close": "100.5", "volume": "10"},
            ],
        }
        sleeps = []
        with (
            mock.patch.dict(
                mds.os.environ,
                {"TWELVE_DATA_API_KEY": "test-key", "TWELVE_DATA_REQUEST_MIN_GAP_SEC": "6"},
                clear=False,
            ),
            mock.patch.object(mds.requests, "get", return_value=_Response(payload)),
            mock.patch.object(mds.time, "sleep", side_effect=sleeps.append),
            mock.patch.object(mds, "save_twelve_data_usage_state"),
        ):
            mds.TWELVE_DATA_USAGE_STATE["last_request_ts"] = time.time()
            mds.fetch_twelve_data_klines("BTCUSDT", "5m", limit=10)
        self.assertTrue(sleeps, "expected fetch_twelve_data_klines to wait out the configured min gap")
        self.assertGreater(sleeps[0], 0)


class TradingViewHelperTests(unittest.TestCase):
    def setUp(self):
        mds.TRADINGVIEW_FAILURE_COOLDOWN.clear()

    def test_symbol_uses_explicit_map_first(self):
        self.assertEqual(mds._tradingview_symbol("BTCUSDT"), "BINANCE:BTCUSDT.P")

    def test_symbol_falls_back_to_usdt_perp_guess(self):
        self.assertEqual(mds._tradingview_symbol("SOLUSDT"), "BINANCE:SOLUSDT.P")

    def test_filter_rows_by_time_range(self):
        rows = [[100, "1"], [200, "1"], [300, "1"]]
        filtered = mds._filter_tradingview_rows(rows, start_time_ms=150, end_time_ms=250)
        self.assertEqual(filtered, [[200, "1"]])

    def test_cooldown_marks_and_expires(self):
        symbol, interval = "BTCUSDT", "1h"
        self.assertFalse(mds.is_tradingview_in_cooldown(symbol, interval))
        with mock.patch.dict(mds.os.environ, {"TRADINGVIEW_FAILURE_COOLDOWN_SEC": "30"}, clear=False):
            mds.mark_tradingview_failure(symbol, interval)
        self.assertTrue(mds.is_tradingview_in_cooldown(symbol, interval))
        mds.TRADINGVIEW_FAILURE_COOLDOWN[mds.tradingview_cooldown_key(symbol, interval)] = time.time() - 1
        self.assertFalse(mds.is_tradingview_in_cooldown(symbol, interval))


if __name__ == "__main__":
    unittest.main()
