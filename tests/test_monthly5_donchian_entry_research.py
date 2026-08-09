import unittest

import numpy as np
import pandas as pd

import monthly5_donchian_entry_research as donchian


class Monthly5DonchianEntryResearchTests(unittest.TestCase):
    def test_gate_requires_directional_channel_edge(self):
        frame = pd.DataFrame(
            index=pd.date_range("2024-01-01", periods=4, freq="5min", tz="UTC")
        )
        position = pd.Series([0.85, 0.15, 0.60, 0.40], index=frame.index)
        allowed = donchian.build_entry_allowed(
            frame,
            np.array([1.0, -1.0, 1.0, -1.0]),
            {"window": 6, "edge": 0.8},
            position,
        )
        self.assertEqual(allowed.tolist(), [True, True, False, False])

    def test_baseline_allows_all_entries(self):
        frame = pd.DataFrame(
            index=pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC")
        )
        allowed = donchian.build_entry_allowed(
            frame, np.array([1.0, -1.0, 0.0]), donchian.BASELINE_CONFIG
        )
        self.assertEqual(allowed.tolist(), [True, True, True])


if __name__ == "__main__":
    unittest.main()
