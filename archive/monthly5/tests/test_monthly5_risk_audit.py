import unittest

import monthly5_risk_audit


class Monthly5RiskAuditTests(unittest.TestCase):
    def test_pairs_recent_long_round_trips_and_counts_net_loss(self):
        rows = [
            {"time": 1000, "side": "BUY", "positionSide": "LONG", "price": "100", "qty": "0.001", "realizedPnl": "0", "commission": "0.01"},
            {"time": 2000, "side": "SELL", "positionSide": "LONG", "price": "99", "qty": "0.001", "realizedPnl": "-0.1", "commission": "0.01"},
            {"time": 3000, "side": "BUY", "positionSide": "LONG", "price": "99", "qty": "0.001", "realizedPnl": "0", "commission": "0.01"},
            {"time": 4000, "side": "SELL", "positionSide": "LONG", "price": "101", "qty": "0.001", "realizedPnl": "0.2", "commission": "0.01"},
        ]

        trades = monthly5_risk_audit.pair_futures_round_trips(rows)
        summary = monthly5_risk_audit.summarize_round_trips(trades)

        self.assertEqual(len(trades), 2)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertAlmostEqual(summary["net_pnl_sum"], 0.06)

    def test_underperforming_monthly5_and_low_live_win_rate_blocks(self):
        position = {
            "monthly5_readiness": {
                "promotion_ready": False,
                "promotion_blockers": [
                    "shadow_monthly_target",
                    "active_underperforming_plan",
                    "recovery_probe_probe_failed",
                ],
                "shadow_paper_return_pct": -0.67,
                "shadow_rolling_paper_return_pct": -0.01,
            },
            "last_close_reason": "SL",
        }
        shadow = {
            "market_selection": {
                "selected_plan": "underperforming_wait",
                "shadow_action": "wait",
                "exposure_cap": 0,
                "reason_codes": ["underperforming_plan_wait"],
            }
        }
        losses = [
            {"outcome": "loss", "gross_pnl": -0.1, "fees": 0.02, "net_pnl": -0.12}
            for _ in range(8)
        ]
        trades = [{"outcome": "win", "gross_pnl": 0.2, "fees": 0.02, "net_pnl": 0.18}] + losses

        audit = monthly5_risk_audit.build_audit(
            position=position,
            shadow=shadow,
            round_trips=trades,
        )

        self.assertEqual(audit["severity"], "block")
        self.assertIn("active_underperforming_plan", audit["findings"])
        self.assertIn("recent_live_win_rate_low", audit["findings"])
        self.assertIn("recent_live_loss_streak", audit["findings"])
        self.assertIn("block_monthly5_entry_until_recovery_probe_success", audit["recommended_actions"])


if __name__ == "__main__":
    unittest.main()
