import unittest
from unittest.mock import patch

import eth


def _row(open_ms, close="100"):
    return [open_ms, "99", "101", "98", close, "10", open_ms + 299999, "0", 0, "0", "0", "0"]


class TwelveDataMaintenanceTests(unittest.TestCase):
    def test_closed_kline_map_excludes_current_candle(self):
        step = eth.KLINE_INTERVAL_MS["5m"]
        now_ms = step * 10 + 1000
        parsed = eth._closed_kline_map([_row(step * 9), _row(step * 10)], "5m", now_ms=now_ms)
        self.assertEqual(list(parsed), [step * 9])

    def test_missing_kline_ranges_groups_each_gap(self):
        step = eth.KLINE_INTERVAL_MS["5m"]
        ranges = eth._missing_kline_ranges([step, step * 2, step * 5, step * 8], "5m")
        self.assertEqual(ranges, [(step * 3, step * 4), (step * 6, step * 7)])

    def test_daily_interval_is_supported_by_history_path(self):
        self.assertEqual(eth.TWELVE_DATA_INTERVAL_MAP["1d"], "1day")
        self.assertTrue(str(eth._twelve_history_path("ETHUSDT", "1d")).endswith("ETHUSDT_1d.csv"))

    def test_quality_check_compares_matching_closed_candle(self):
        step = eth.KLINE_INTERVAL_MS["5m"]
        with (
            patch.object(eth.time, "time", return_value=(step * 10 + 1000) / 1000),
            patch.object(
                eth,
                "_fetch_twelve_data_kline_rows",
                return_value=[_row(step * 8, "100"), _row(step * 9, "100")],
            ),
            patch.object(
                eth,
                "_fetch_binance_kline_rows",
                return_value=([_row(step * 8, "100.1"), _row(step * 9, "100.2")], "futures"),
            ),
        ):
            result = eth._inspect_twelve_data_quality()

        self.assertTrue(result["ok"])
        self.assertEqual(result["latest_open_ms"], step * 9)
        self.assertEqual(result["compared_bars"], 2)
        self.assertEqual(result["twelve_data_missing_bars"], 0)


if __name__ == "__main__":
    unittest.main()
