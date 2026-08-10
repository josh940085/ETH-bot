import unittest

import market_history


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


if __name__ == "__main__":
    unittest.main()
