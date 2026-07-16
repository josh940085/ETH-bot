import os
import unittest

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth


class LiquidationClusterTests(unittest.TestCase):
    def test_builds_dense_long_liquidation_zone_below_price(self):
        now = 10_000.0
        events = [
            {"ts": now - 30, "price": 1990.0, "qty": 20.0, "liquidation_side": "long"},
            {"ts": now - 60, "price": 1991.0, "qty": 18.0, "liquidation_side": "long"},
            {"ts": now - 90, "price": 2015.0, "qty": 2.0, "liquidation_side": "short"},
        ]
        snapshot = eth._summarize_liquidation_clusters(events, 2000.0, now_ts=now)

        self.assertEqual(snapshot["liquidation_cluster_count"], 1)
        self.assertEqual(snapshot["nearest_liquidation_below"]["dominant_side"], "long")
        self.assertLess(snapshot["liquidation_pressure"], -0.8)
        self.assertGreaterEqual(snapshot["liquidation_cluster_risk"], 0.65)
        self.assertIn("多單靠近爆倉密集區", eth._liquidation_cluster_guard_reason("long", snapshot))
        self.assertEqual(eth._liquidation_cluster_guard_reason("short", snapshot), "")

    def test_ignores_expired_and_single_small_events(self):
        now = 20_000.0
        events = [
            {"ts": now - eth.LIQUIDATION_CLUSTER_WINDOW_SEC - 1, "price": 2000.0, "qty": 100.0, "liquidation_side": "long"},
            {"ts": now - 10, "price": 2001.0, "qty": 1.0, "liquidation_side": "short"},
        ]
        snapshot = eth._summarize_liquidation_clusters(events, 2000.0, now_ts=now)

        self.assertEqual(snapshot["liquidation_event_count"], 1)
        self.assertEqual(snapshot["liquidation_cluster_count"], 0)
        self.assertEqual(snapshot["liquidation_cluster_risk"], 0.0)

    def test_force_order_side_mapping(self):
        sell = {
            "e": "forceOrder",
            "E": 1_700_000_000_000,
            "o": {"s": "ETHUSDT", "S": "SELL", "ap": "2000", "z": "30", "T": 1_700_000_000_000},
        }
        buy = {
            "e": "forceOrder",
            "E": 1_700_000_001_000,
            "o": {"s": "ETHUSDT", "S": "BUY", "ap": "2005", "z": "30", "T": 1_700_000_001_000},
        }
        old_path = eth.LIQUIDATION_EVENTS_PATH
        try:
            eth.LIQUIDATION_EVENTS_PATH = old_path.with_name("test_liquidation_events.json")
            with eth.LIQUIDATION_EVENTS_LOCK:
                eth.LIQUIDATION_EVENTS.clear()
                eth.LIQUIDATION_EVENTS_LOADED = True
            self.assertTrue(eth._record_liquidation_force_order(sell))
            self.assertTrue(eth._record_liquidation_force_order(buy))
            with eth.LIQUIDATION_EVENTS_LOCK:
                sides = [row["liquidation_side"] for row in eth.LIQUIDATION_EVENTS]
            self.assertEqual(sides, ["long", "short"])
        finally:
            eth.LIQUIDATION_EVENTS_PATH = old_path
            test_path = old_path.with_name("test_liquidation_events.json")
            if test_path.exists():
                test_path.unlink()


if __name__ == "__main__":
    unittest.main()
