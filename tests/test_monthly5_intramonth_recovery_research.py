import unittest

import numpy as np
import pandas as pd

import monthly5_intramonth_recovery_research as recovery
import monthly5_risk_profile_walkforward as risk_walkforward
import monthly5_volatility_regime_research as volatility


class Monthly5IntramonthRecoveryResearchTests(unittest.TestCase):
    def _frame(self, closes):
        index = pd.date_range("2026-01-01T00:05:00Z", periods=len(closes), freq="5min")
        values = np.asarray(closes, dtype="float64")
        return pd.DataFrame(
            {
                "open": values,
                "high": values,
                "low": values,
                "close": values,
            },
            index=index,
        )

    def test_no_recovery_matches_pipeline_without_account_state_transition(self):
        frame = self._frame([100.0, 101.0, 102.0, 101.0, 100.0])
        desired = np.ones(len(frame))
        profile_ids = np.zeros(len(frame), dtype="int32")
        components = risk_walkforward.simulate_dynamic_risk_path(
            frame,
            desired,
            profile_ids,
        )
        expected, expected_scales = volatility.apply_components(frame, components)
        actual, scales, flags, triggers, positions = recovery.simulate_account_path(
            frame,
            desired,
            profile_ids,
            desired,
            {"mode": "none", "trigger": None, "scale": 0.0, "exit": 0.0},
        )
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(scales, expected_scales)
        self.assertFalse(flags.any())
        self.assertEqual(triggers, 0)
        self.assertTrue((positions != 0.0).any())

    def test_drawdown_activates_recovery_on_next_bar(self):
        frame = self._frame([100.0, 99.0, 99.0, 99.0])
        desired = np.ones(len(frame))
        _, scales, flags, triggers, _ = recovery.simulate_account_path(
            frame,
            desired,
            np.zeros(len(frame), dtype="int32"),
            desired * -1.0,
            {"mode": "inverse", "trigger": -0.005, "scale": 0.5, "exit": 0.0},
        )
        self.assertEqual(triggers, 1)
        self.assertFalse(flags[1])
        self.assertTrue(flags[2])
        self.assertEqual(scales[2], 0.5)


if __name__ == "__main__":
    unittest.main()
