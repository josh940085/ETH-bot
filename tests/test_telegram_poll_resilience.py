import unittest
from types import SimpleNamespace
from unittest.mock import patch

import telegram


class TelegramPollResilienceTests(unittest.TestCase):
    def setUp(self):
        self.backoff = telegram.TELEGRAM_POLL_BACKOFF_SEC
        self.error_key = telegram.TELEGRAM_POLL_LAST_ERROR_KEY
        self.log_ts = telegram.TELEGRAM_POLL_LAST_LOG_TS

    def tearDown(self):
        telegram.TELEGRAM_POLL_BACKOFF_SEC = self.backoff
        telegram.TELEGRAM_POLL_LAST_ERROR_KEY = self.error_key
        telegram.TELEGRAM_POLL_LAST_LOG_TS = self.log_ts

    def test_retry_after_is_read_from_telegram_response(self):
        response = SimpleNamespace(
            json=lambda: {"parameters": {"retry_after": 17}},
            headers={},
        )
        error = SimpleNamespace(response=response)

        self.assertEqual(telegram._telegram_poll_retry_after(error), 17)

    def test_bot_token_is_redacted_from_poll_error(self):
        message = "429 for https://api.telegram.org/bot12345:secret/getUpdates"

        redacted = telegram._redact_telegram_error(message)

        self.assertNotIn("12345:secret", redacted)
        self.assertIn("bot<redacted>", redacted)

    @patch("telegram.time.sleep")
    @patch("builtins.print")
    def test_rate_limit_waits_for_retry_after(self, _print, sleep):
        response = SimpleNamespace(
            json=lambda: {"parameters": {"retry_after": 13}},
            headers={},
        )
        error = SimpleNamespace(response=response)
        telegram.TELEGRAM_POLL_BACKOFF_SEC = 1.0
        telegram.TELEGRAM_POLL_LAST_ERROR_KEY = ""
        telegram.TELEGRAM_POLL_LAST_LOG_TS = 0.0

        telegram._handle_telegram_poll_error(error)

        sleep.assert_called_once_with(13.0)


if __name__ == "__main__":
    unittest.main()
