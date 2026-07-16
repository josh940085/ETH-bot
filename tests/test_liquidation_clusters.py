import os
import unittest

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth


class LiquidationClusterTests(unittest.TestCase):
    def test_predicted_liquidation_zones_are_shadow_only(self):
        now = 30_000.0
        cohorts = [
            {"ts": now - 60, "entry_price": 2000.0, "liquidation_price": 1905.0, "notional_usdt": 250000.0, "side": "long", "leverage": 20, "confidence": 0.7},
            {"ts": now - 60, "entry_price": 2000.0, "liquidation_price": 2095.0, "notional_usdt": 250000.0, "side": "short", "leverage": 20, "confidence": 0.7},
        ]
        snapshot = eth._summarize_predicted_liquidation_zones(cohorts, 2000.0, now_ts=now)

        self.assertEqual(snapshot["predicted_liquidation_mode"], "shadow")
        self.assertEqual(snapshot["predicted_liquidation_zone_count"], 2)
        self.assertEqual(snapshot["nearest_predicted_liquidation_below"]["dominant_side"], "long")
        self.assertEqual(snapshot["nearest_predicted_liquidation_above"]["dominant_side"], "short")
        self.assertEqual(eth._liquidation_cluster_guard_reason("long", snapshot), "")

    def test_oi_increase_creates_long_and_short_leverage_cohorts(self):
        old_path = eth.PREDICTED_LIQUIDATION_STATE_PATH
        old_loaded = eth.PREDICTED_LIQUIDATION_LOADED
        with eth.PREDICTED_LIQUIDATION_LOCK:
            old_cohorts = list(eth.PREDICTED_LIQUIDATION_COHORTS)
            old_stats = dict(eth.PREDICTED_LIQUIDATION_STATS)
        try:
            eth.PREDICTED_LIQUIDATION_STATE_PATH = old_path.with_name("test_predicted_liquidation_state.json")
            with eth.PREDICTED_LIQUIDATION_LOCK:
                eth.PREDICTED_LIQUIDATION_COHORTS.clear()
                for key in eth.PREDICTED_LIQUIDATION_STATS:
                    eth.PREDICTED_LIQUIDATION_STATS[key] = 0.0
                eth.PREDICTED_LIQUIDATION_LOADED = True
            eth._score_predicted_liquidation_event({"ts": 40_000.0, "price": 1900.0, "qty": 10.0, "liquidation_side": "long"})
            self.assertEqual(eth.PREDICTED_LIQUIDATION_STATS["evaluated_events"], 0.0)
            created = eth._record_predicted_liquidation_cohorts(1000.0, 1100.0, 2000.0, 0.65, 0.0001, now_ts=40_000.0)
            self.assertTrue(created)
            with eth.PREDICTED_LIQUIDATION_LOCK:
                rows = list(eth.PREDICTED_LIQUIDATION_COHORTS)
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(row["liquidation_price"] < 2000.0 for row in rows if row["side"] == "long"))
            self.assertTrue(all(row["liquidation_price"] > 2000.0 for row in rows if row["side"] == "short"))
        finally:
            eth.PREDICTED_LIQUIDATION_STATE_PATH = old_path
            eth.PREDICTED_LIQUIDATION_LOADED = old_loaded
            with eth.PREDICTED_LIQUIDATION_LOCK:
                eth.PREDICTED_LIQUIDATION_COHORTS.clear()
                eth.PREDICTED_LIQUIDATION_COHORTS.extend(old_cohorts)
                eth.PREDICTED_LIQUIDATION_STATS.update(old_stats)
            test_path = old_path.with_name("test_predicted_liquidation_state.json")
            if test_path.exists():
                test_path.unlink()

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
