import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth
import local_chat
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


class LocalChatBoundaryTests(unittest.TestCase):
    def test_trading_core_uses_local_response_parser(self):
        self.assertIs(eth._extract_chat_text, local_chat.extract_chat_text)

    def test_local_response_parser_accepts_message_content(self):
        payload = {"choices": [{"message": {"content": "  本機分析  "}}]}
        self.assertEqual(local_chat.extract_chat_text(payload), "本機分析")


if __name__ == "__main__":
    unittest.main()
