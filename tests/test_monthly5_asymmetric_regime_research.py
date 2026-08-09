import unittest

import pandas as pd

import monthly5_asymmetric_regime_research as research


class Monthly5AsymmetricRegimeResearchTests(unittest.TestCase):
    def test_up_and_down_use_independent_confirmation_counts(self):
        index = pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC")
        frame = pd.DataFrame({"close": range(12)}, index=index)
        labels = pd.Series(
            ["up", "down", "range", "up", "down"],
            index=index[[0, 2, 5, 7, 10]],
        )
        position, regimes = research.build_asymmetric_position(
            frame,
            labels,
            up_confirmation_bars=2,
            down_confirmation_bars=3,
            range_grace_bars=1,
        )
        self.assertEqual(regimes.tolist(), [
            "up", "up", "down", "down", "down", "range",
            "range", "up", "up", "up", "down", "down",
        ])
        self.assertEqual(position.tolist(), [
            0.0, 1.0, 1.0, 1.0, -1.0, -1.0,
            0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ])

    def test_candidate_grid_contains_current_symmetric_baseline(self):
        rows = research.configs()
        self.assertEqual(len(rows), 48)
        self.assertTrue(
            any(
                row["up_confirmation_bars"] == 1
                and row["down_confirmation_bars"] == 1
                and row["range_grace_bars"] == 6
                for row in rows
            )
        )


if __name__ == "__main__":
    unittest.main()
