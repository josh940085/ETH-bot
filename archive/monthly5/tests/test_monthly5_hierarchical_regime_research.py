import unittest

import numpy as np

import monthly5_hierarchical_regime_research as hierarchy


class Monthly5HierarchicalRegimeResearchTests(unittest.TestCase):
    def test_fallback_only_fills_primary_flat_bars(self):
        desired, strategy_ids = hierarchy.combine_paths(
            np.array([1.0, 0.0, -1.0, 0.0]),
            np.array([0, 0, 1, 1]),
            np.array([-1.0, 1.0, 1.0, -1.0]),
        )
        self.assertEqual(desired.tolist(), [1.0, 1.0, -1.0, -1.0])
        self.assertEqual(strategy_ids.tolist(), [0, 10, 1, 10])


if __name__ == "__main__":
    unittest.main()
