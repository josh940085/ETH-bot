import os
import unittest

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth


class LeverageConfigTests(unittest.TestCase):
    def test_default_and_live_cap_are_five_x(self):
        self.assertEqual(eth.DEFAULT_LEV, 5)
        self.assertEqual(eth.COPY_TRADE_MAX_LEVERAGE, 5)
        configured = eth._safe_int(os.getenv("COPY_TRADE_LEVERAGE"), eth.DEFAULT_LEV)
        self.assertEqual(min(configured, eth.COPY_TRADE_MAX_LEVERAGE), 5)


if __name__ == "__main__":
    unittest.main()
