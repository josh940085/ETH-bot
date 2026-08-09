import unittest

import numpy as np

import monthly5_monthly_regime_selector as selector


class Monthly5MonthlyRegimeSelectorTests(unittest.TestCase):
    def _matrix(self, returns):
        values = np.asarray(returns, dtype="float64")
        return {
            "keys": np.asarray([f"candidate_{index}" for index in range(len(values))]),
            "months": np.asarray([f"2020-{month:02d}" for month in range(1, values.shape[1] + 1)]),
            "returns": values,
            "flats": np.zeros_like(values),
            "leverages": np.ones(len(values), dtype="int32"),
        }

    def _config(self, **overrides):
        config = {
            "use_regime": False,
            "min_regime_months": 3,
            "lookback_months": 12,
            "q25_weight": 0.0,
            "hit_weight": 0.0,
            "volatility_weight": 0.0,
        }
        config.update(overrides)
        return config

    def test_selection_does_not_use_current_month_return(self):
        matrix = self._matrix(
            [
                [0.06, 0.06, 0.06, 0.06, -0.50],
                [0.01, 0.01, 0.01, 0.01, 2.00],
            ]
        )
        result = selector.run_causal_selector(
            matrix,
            np.asarray(["range"] * 5),
            self._config(),
        )
        self.assertEqual(result["selected_indices"][-1], 0)

    def test_matching_regime_history_can_select_bear_candidate(self):
        matrix = self._matrix(
            [
                [0.08, -0.02, 0.08, -0.02, 0.08, -0.02, 0.0, 0.0],
                [-0.02, 0.08, -0.02, 0.08, -0.02, 0.08, 0.0, 0.0],
            ]
        )
        regimes = np.asarray(["up", "down"] * 4)
        result = selector.run_causal_selector(
            matrix,
            regimes,
            self._config(use_regime=True, min_regime_months=3),
        )
        self.assertEqual(result["selected_indices"][6], 0)
        self.assertEqual(result["selected_indices"][7], 1)

    def test_monthly_screening_floor_is_bounded(self):
        matrix = self._matrix([[-0.50, -0.50, -0.50, -0.50]])
        result = selector.run_causal_selector(
            matrix,
            np.asarray(["range"] * 4),
            self._config(),
        )
        self.assertTrue(np.all(result["returns"] == selector.MONTHLY_SCREENING_FLOOR))
        self.assertTrue(np.all(result["raw_returns"] == -0.50))


if __name__ == "__main__":
    unittest.main()
