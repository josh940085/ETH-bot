import unittest

import numpy as np
import pandas as pd

import monthly5_matched_execution_replay as replay


class Monthly5MatchedExecutionReplayTests(unittest.TestCase):
    def test_parse_candidate_treats_reduced_leverage_as_absolute(self):
        parsed = replay.parse_candidate(
            "mom480_ls|lev5|stopNone|target0.05|redlev0.5"
        )
        self.assertEqual(parsed["leverage"], 5.0)
        self.assertEqual(parsed["target"], 0.05)
        self.assertEqual(parsed["reduced_leverage"], 0.5)

    def test_simulator_reduces_to_absolute_leverage_after_target(self):
        index = pd.date_range("2026-01-01 01:00:00Z", periods=5, freq="1h")
        close = np.asarray([100.0, 102.0, 103.0, 104.0, 105.0])
        frame = pd.DataFrame({"close": close}, index=index)
        key = "buy_hold|lev5|stopNone|target0.05|redlev0.5"
        factors, exposures = replay.simulate(
            frame,
            {"2026-01": key},
            {key: np.ones(len(frame))},
            replay.ACCOUNT_OVERLAYS[0],
        )
        self.assertEqual(exposures[0], 5.0)
        self.assertEqual(exposures[1], 5.0)
        self.assertEqual(exposures[2], 0.5)
        self.assertTrue(np.all(factors > 0.0))

    def test_hard_stop_flattens_following_bar(self):
        index = pd.date_range("2026-01-01 01:00:00Z", periods=4, freq="1h")
        frame = pd.DataFrame(
            {"close": np.asarray([100.0, 98.0, 97.0, 96.0])}, index=index
        )
        key = "buy_hold|lev5|stopNone|targetNone|redlev1.0"
        factors, exposures = replay.simulate(
            frame,
            {"2026-01": key},
            {key: np.ones(len(frame))},
            replay.ACCOUNT_OVERLAYS[1],
        )
        self.assertEqual(exposures[1], 5.0)
        self.assertEqual(exposures[2], 0.0)
        self.assertEqual(exposures[3], 0.0)
        self.assertAlmostEqual(float(np.prod(factors)), 0.92, places=10)
        self.assertTrue(np.all(factors > 0.0))

    def test_completed_4h_force_uses_regime_direction(self):
        index = pd.date_range("2026-01-01 01:00:00Z", periods=3, freq="1h")
        frame = pd.DataFrame({"close": [100.0, 101.0, 100.0]}, index=index)
        key = "buy_hold|lev1|stopNone|targetNone|redlev1.0"
        overlay = {
            **replay.ACCOUNT_OVERLAYS[0],
            "direction_policy": "completed_4h_force",
        }
        _, exposures = replay.simulate(
            frame,
            {"2026-01": key},
            {key: np.ones(len(frame))},
            overlay,
            market_regimes=np.asarray(["up", "down", "range"]),
        )
        self.assertEqual(exposures.tolist(), [1.0, -1.0, 1.0])

    def test_completed_4h_specialist_uses_directional_paths(self):
        index = pd.date_range("2026-01-01 01:00:00Z", periods=3, freq="1h")
        frame = pd.DataFrame({"close": [100.0, 101.0, 100.0]}, index=index)
        key = "buy_hold|lev1|stopNone|targetNone|redlev1.0"
        overlay = {
            **replay.ACCOUNT_OVERLAYS[0],
            "direction_policy": "completed_4h_specialist_ma24_96",
        }
        _, exposures = replay.simulate(
            frame,
            {"2026-01": key},
            {key: np.ones(len(frame))},
            overlay,
            market_regimes=np.asarray(["up", "down", "range"]),
            specialist_desired={
                "ma24_96": {
                    "long": np.asarray([1.0, 1.0, 1.0]),
                    "short": np.asarray([0.0, -1.0, -1.0]),
                }
            },
        )
        self.assertEqual(exposures.tolist(), [1.0, -1.0, 1.0])

    def test_recovery_uses_inverse_selected_at_bounded_leverage(self):
        index = pd.date_range("2026-01-01 01:00:00Z", periods=4, freq="1h")
        frame = pd.DataFrame(
            {"close": [100.0, 99.0, 98.0, 97.0]}, index=index
        )
        key = "buy_hold|lev5|stopNone|targetNone|redlev1.0"
        overlay = {
            **replay.ACCOUNT_OVERLAYS[-1],
            "direction_policy": "selected_signal",
            "recovery_policy": "inverse_selected",
            "recovery_trigger": -0.02,
            "recovery_leverage": 0.5,
            "recovery_exit": 0.0,
        }
        _, exposures = replay.simulate(
            frame,
            {"2026-01": key},
            {key: np.ones(len(frame))},
            overlay,
        )
        self.assertEqual(exposures[1], 5.0)
        self.assertEqual(exposures[2], -0.5)
        self.assertEqual(exposures[3], -0.5)


if __name__ == "__main__":
    unittest.main()
