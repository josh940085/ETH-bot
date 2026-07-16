import unittest
from unittest.mock import patch

import panel_realtime_server


class PanelApiTokenUsageTests(unittest.TestCase):
    def test_openai_monitor_is_not_exposed(self):
        with patch.object(panel_realtime_server, "_runtime_env_value", return_value=""):
            payload = panel_realtime_server._build_api_token_usage()

        item_ids = {item.get("id") for item in payload.get("items", [])}
        self.assertNotIn("openai", item_ids)
        self.assertEqual(item_ids, {"twelve_data", "binance", "telegram"})


if __name__ == "__main__":
    unittest.main()
