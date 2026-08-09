import unittest
from unittest import mock

import pandas as pd

import market_history
import monthly5_selector_cache


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(self.pages.pop(0))


class MarketHistoryApiTests(unittest.TestCase):
    def test_futures_api_paginates_from_last_close(self):
        first = [
            0,
            "100", "101", "99", "100.5", "10",
            299_999,
            "0", 0, "0", "0", "0",
        ]
        page = [list(first) for _ in range(1500)]
        for index, row in enumerate(page):
            row[0] = index * 300_000
            row[6] = row[0] + 299_999
        final = [list(first)]
        final[0][0] = 1500 * 300_000
        final[0][6] = final[0][0] + 299_999
        session = _Session([page, final])
        frame = market_history.fetch_klines_from_binance_api(
            "BTCUSDT", "5m", 0, final[0][6], session=session
        )
        self.assertEqual(len(frame), 1501)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(session.calls[1][1]["params"]["startTime"], page[-1][6] + 1)
        self.assertEqual(frame.attrs["kline_source"], "binance_futures_api")

    def test_selector_history_fills_missing_archive_prefix(self):
        columns = {
            "open": [100.0], "high": [101.0], "low": [99.0],
            "close": [100.5], "volume": [10.0],
        }
        archive = pd.DataFrame(
            columns,
            index=pd.DatetimeIndex(["2020-01-02T00:04:59.999Z"]),
        )
        archive.attrs["kline_source"] = "binance_history_um:1d"
        prefix = pd.DataFrame(
            columns,
            index=pd.DatetimeIndex(["2019-12-28T00:04:59.999Z"]),
        )
        prefix.attrs["kline_source"] = "binance_futures_api"
        with (
            mock.patch.object(
                market_history, "fetch_klines_from_binance_history", return_value=archive
            ),
            mock.patch.object(
                market_history, "fetch_klines_from_binance_api", return_value=prefix
            ) as api,
        ):
            frame = monthly5_selector_cache.load_history("2020-01-01", "2020-01-02")
        self.assertEqual(len(frame), 2)
        self.assertTrue(frame.index.is_monotonic_increasing)
        self.assertEqual(
            frame.attrs["kline_source"],
            "binance_futures_api+binance_history_um:1d",
        )
        api.assert_called_once()


if __name__ == "__main__":
    unittest.main()
