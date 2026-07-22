import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth
import news
import openai_chat
import runtime_config


class RuntimeConfigTests(unittest.TestCase):
    def test_env_file_loading_preserves_process_values_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, ".env").write_text("EXISTING=file\nNEW_VALUE=loaded\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"EXISTING": "process"}, clear=True):
                runtime_config.load_local_env(repo_dir=tmp, names=(".env",))
                self.assertEqual(os.environ["EXISTING"], "process")
                self.assertEqual(os.environ["NEW_VALUE"], "loaded")

    def test_env_file_loading_can_explicitly_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, ".env").write_text("EXISTING=file\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"EXISTING": "process"}, clear=True):
                runtime_config.load_local_env(overwrite=True, repo_dir=tmp, names=(".env",))
                self.assertEqual(os.environ["EXISTING"], "file")


class OpenAIChatBoundaryTests(unittest.TestCase):
    def test_news_and_trading_core_share_payload_helpers(self):
        self.assertIs(eth._build_openai_chat_payload, openai_chat.build_openai_chat_payload)
        self.assertIs(news._build_openai_chat_payload, openai_chat.build_openai_chat_payload)

    def test_reasoning_and_standard_payloads_keep_supported_fields(self):
        with mock.patch.dict(os.environ, {"OPENAI_REASONING_EFFORT": "low"}):
            reasoning = openai_chat.build_openai_chat_payload("gpt-5-mini", [], temperature=0)
        standard = openai_chat.build_openai_chat_payload("gpt-4o-mini", [], temperature=0)
        self.assertEqual(reasoning["reasoning_effort"], "low")
        self.assertNotIn("temperature", reasoning)
        self.assertEqual(standard["temperature"], 0)


if __name__ == "__main__":
    unittest.main()
