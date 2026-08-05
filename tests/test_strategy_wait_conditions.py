import os
import unittest
from unittest.mock import patch

import eth


class StrategyWaitConditionsTests(unittest.TestCase):
    def test_stop_distance_explains_current_and_required_values(self):
        decision = {
            "final": "觀望（停損距離過近）",
            "risk_rate": 0.0018,
            "multitimeframe_bull_reclaim": {
                "enabled": True,
                "applied": False,
                "diagnostics": {"required_price": 1943.73},
            },
        }
        with (
            patch.dict(os.environ, {"TRADE_MIN_ENTRY_RISK_RATE": "0.003"}, clear=False),
            patch.dict(
                eth.POSITION_PANEL_STATE,
                {"last_close_reason": "", "last_sl_review": {}},
                clear=False,
            ),
        ):
            result = eth._build_strategy_wait_conditions(
                decision,
                1930.76,
                "waiting",
                "觀望（停損距離過近）",
            )

        self.assertEqual(result[0]["key"], "stop_distance")
        self.assertEqual(result[0]["current"], "0.180%")
        self.assertEqual(result[0]["target"], "至少 0.300%")
        self.assertEqual(result[1]["key"], "bull_reclaim_price")
        self.assertEqual(result[1]["target"], "站上 1943.73")

    def test_pending_confirmation_shows_hold_duration_and_price_validation(self):
        with patch.dict(os.environ, {"TRADE_ENTRY_CONFIRM_MIN_WAIT_SEC": "15"}, clear=False):
            result = eth._build_strategy_wait_conditions(
                {},
                1930.76,
                "pending_confirmation",
                "等待進場確認",
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["key"], "entry_confirmation")
        self.assertEqual(result[0]["target"], "維持 15 秒")
        self.assertIn("Binance Mark Price", result[0]["detail"])

    def test_recent_sl_opposite_review_is_shown_before_reclaim_hint(self):
        decision = {
            "final": "觀望（RR不足）",
            "rr_at_entry": 1.6,
            "breakout_attempt": 1,
            "breakout_quality_score": 0.8,
            "breakout_quality_required": 3.5,
            "resistance_break_level": 64147.2,
            "multitimeframe_bull_reclaim": {
                "enabled": True,
                "applied": False,
                "diagnostics": {"required_price": 64615.15},
            },
        }
        sl_review = {
            "opposite_direction_review": {
                "opposite_direction": "short",
                "ready_for_fresh_evaluation": False,
                "missing_conditions": [
                    "4H未轉空",
                    "反向突破未確認",
                    "宏觀未支持反向",
                ],
            },
        }

        with patch.dict(
            eth.POSITION_PANEL_STATE,
            {
                "last_close_reason": "SL",
                "last_sl_review": sl_review,
            },
            clear=False,
        ):
            result = eth._build_strategy_wait_conditions(
                decision,
                64150.5,
                "waiting",
                "觀望（RR不足）",
            )

        self.assertEqual([item["key"] for item in result], [
            "resistance_breakout",
            "risk_reward",
            "post_sl_opposite",
        ])
        self.assertEqual(result[2]["label"], "反向做空")
        self.assertIn("4H未轉空", result[2]["current"])
        self.assertIn("不自動反手", result[2]["detail"])

    def test_monthly5_profile_quality_block_explains_missing_override_gates(self):
        decision = {
            "final": "觀望（MLX回測輪廓不佳）",
            "breakout_attempt": 1,
            "breakout_quality_score": 2.8,
            "breakout_quality_required": 3.5,
            "resistance_break_level": 64147.2,
            "monthly5_signal_override": {
                "reason": "monthly5_profile_wait_quality_block",
                "direction": "long",
                "market_bias": "bullish",
                "bull_score": 2.8,
                "bear_score": 1.0,
                "min_score_gap": 2.0,
                "net_edge_rate_est": 0.0001,
                "min_net_edge_rate_est": 0.003,
                "breakout_quality_score": 2.8,
                "breakout_quality_required": 3.5,
                "breakout_confirmed": False,
                "direction_prob": 0.7338,
                "min_direction_prob": 0.65,
            },
        }

        with patch.dict(
            eth.POSITION_PANEL_STATE,
            {"last_close_reason": "", "last_sl_review": {}},
            clear=False,
        ):
            result = eth._build_strategy_wait_conditions(
                decision,
                64154.2,
                "waiting",
                "觀望（MLX回測輪廓不佳）",
            )

        self.assertEqual([item["key"] for item in result], [
            "resistance_breakout",
            "monthly5_profile_quality",
        ])
        self.assertEqual(result[1]["label"], "月報酬5%接管品質")
        self.assertIn("bias差 1.8<2.0", result[1]["current"])
        self.assertIn("期望值 +0.010%<+0.300%", result[1]["current"])
        self.assertIn("突破 2.8/3.5", result[1]["current"])


if __name__ == "__main__":
    unittest.main()
