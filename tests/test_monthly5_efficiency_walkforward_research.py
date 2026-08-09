import unittest

import numpy as np

import monthly5_efficiency_walkforward_research as walkforward


class Monthly5EfficiencyWalkforwardResearchTests(unittest.TestCase):
    def test_selected_entry_gate_uses_candidate_for_each_bar(self):
        matrix = np.array(
            [
                [True, True, True, True],
                [False, False, False, False],
            ]
        )
        selected = np.array([0, 1, 1, 0], dtype="int32")
        actual = matrix[selected, np.arange(4)]
        self.assertEqual(actual.tolist(), [True, False, False, True])

    def test_walkforward_grid_stays_bounded(self):
        self.assertEqual(len(walkforward.LOOKBACK_VALUES) * len(walkforward.SCORE_MODES), 8)


if __name__ == "__main__":
    unittest.main()
