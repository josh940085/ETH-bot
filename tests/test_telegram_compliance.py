import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import telegram


class TelegramComplianceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.originals = {
            "state": telegram.TELEGRAM_STATE_FILE,
            "key": telegram.TELEGRAM_STATE_KEY_FILE,
            "send_lock": telegram.TELEGRAM_SEND_LOCK_FILE,
        }
        telegram.TELEGRAM_STATE_FILE = root / "data" / ".telegram_state.json"
        telegram.TELEGRAM_STATE_KEY_FILE = root / "secrets" / "telegram_state.key"
        telegram.TELEGRAM_SEND_LOCK_FILE = root / "data" / ".telegram_send_rate"

    def tearDown(self):
        telegram.TELEGRAM_STATE_FILE = self.originals["state"]
        telegram.TELEGRAM_STATE_KEY_FILE = self.originals["key"]
        telegram.TELEGRAM_SEND_LOCK_FILE = self.originals["send_lock"]
        self.tempdir.cleanup()

    def test_plaintext_state_is_migrated_to_encrypted_private_files(self):
        telegram.TELEGRAM_STATE_FILE.parent.mkdir(parents=True)
        telegram.TELEGRAM_STATE_FILE.write_text(
            json.dumps({"last_update_id": 7, "username": "private-user"}),
            encoding="utf-8",
        )

        self.assertTrue(telegram.ensure_telegram_state_encrypted())

        raw = telegram.TELEGRAM_STATE_FILE.read_text(encoding="utf-8")
        self.assertTrue(raw.startswith(telegram.TELEGRAM_STATE_ENCRYPTION_PREFIX))
        self.assertNotIn("private-user", raw)
        self.assertEqual(telegram.read_telegram_state_locked()["last_update_id"], 7)
        self.assertEqual(stat.S_IMODE(telegram.TELEGRAM_STATE_FILE.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(telegram.TELEGRAM_STATE_KEY_FILE.stat().st_mode), 0o600)
        self.assertNotEqual(telegram.TELEGRAM_STATE_FILE.parent, telegram.TELEGRAM_STATE_KEY_FILE.parent)

    def test_authorization_is_private_whitelist_and_fail_closed(self):
        with patch.dict(
            os.environ,
            {
                "POSITION_PANEL_ALLOWED_TELEGRAM_USER_IDS": "123",
                "TELEGRAM_PRIVATE_CHAT_ID": "",
                "TELEGRAM_CHAT_ID": "",
            },
            clear=False,
        ):
            self.assertTrue(telegram.is_authorized_telegram_user(123, 123, "private"))
            self.assertFalse(telegram.is_authorized_telegram_user(124, 124, "private"))
            self.assertFalse(telegram.is_authorized_telegram_user(123, 123, "group"))
            self.assertFalse(telegram.is_authorized_telegram_user(123, 999, "private"))

        with patch.dict(
            os.environ,
            {
                "POSITION_PANEL_ALLOWED_TELEGRAM_USER_IDS": "",
                "TELEGRAM_PRIVATE_CHAT_ID": "",
                "TELEGRAM_CHAT_ID": "",
            },
            clear=False,
        ):
            self.assertFalse(telegram.is_authorized_telegram_user(123, 123, "private"))

    def test_opt_out_overrides_configured_notification_target(self):
        with patch.dict(
            os.environ,
            {
                "POSITION_PANEL_ALLOWED_TELEGRAM_USER_IDS": "123",
                "TELEGRAM_PRIVATE_CHAT_ID": "123",
                "TELEGRAM_CHAT_ID": "",
            },
            clear=False,
        ):
            self.assertEqual(telegram.get_notification_chat_ids(), ["123"])
            self.assertTrue(telegram.set_notification_opt_out(123, True))
            self.assertEqual(telegram.get_notification_chat_ids(), [])
            self.assertTrue(telegram.set_notification_opt_out(123, False))
            self.assertEqual(telegram.get_notification_chat_ids(), ["123"])

    def test_delete_user_data_removes_stored_identifiers(self):
        with patch.dict(
            os.environ,
            {"POSITION_PANEL_ALLOWED_TELEGRAM_USER_IDS": "123"},
            clear=False,
        ):
            telegram.update_telegram_state(
                lambda payload: payload.update(
                    {
                        "notify_chat_ids": ["123"],
                        "last_private_chat_id": 123,
                        "pending_commands": [
                            {"chat_id": 123, "user_id": 123, "username": "private-user", "text": "/help"}
                        ],
                        "telegram_delivery_events": [{"chat_id": "123", "ok": True, "ts": 1}],
                    }
                )
            )
            self.assertTrue(telegram.delete_telegram_user_data(123, 123))
            state = telegram.read_telegram_state_locked()
            self.assertEqual(state.get("notify_chat_ids"), [])
            self.assertNotIn("last_private_chat_id", state)
            self.assertEqual(state.get("pending_commands"), [])
            self.assertEqual(state.get("telegram_delivery_events"), [])

    def test_unauthorized_restart_is_consumed_without_execution_or_storage(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "ok": True,
                    "result": [
                        {
                            "update_id": 42,
                            "message": {
                                "text": "/restart",
                                "chat": {"id": 999, "type": "private"},
                                "from": {
                                    "id": 999,
                                    "username": "intruder",
                                    "first_name": "Unknown",
                                },
                            },
                        }
                    ],
                }

        with patch.dict(
            os.environ,
            {
                "POSITION_PANEL_ALLOWED_TELEGRAM_USER_IDS": "123",
                "TELEGRAM_PRIVATE_CHAT_ID": "123",
                "TELEGRAM_CHAT_ID": "",
            },
            clear=False,
        ), patch.object(telegram.HTTP_SESSION, "get", return_value=FakeResponse()), patch.object(
            telegram, "send_message"
        ) as send:
            telegram.poll_telegram_commands(token="test-token")

        state = telegram.read_telegram_state_locked()
        self.assertEqual(state.get("last_update_id"), 42)
        self.assertFalse(state.get("restart_requested", False))
        self.assertEqual(state.get("pending_commands", []), [])
        self.assertNotIn("intruder", telegram.TELEGRAM_STATE_FILE.read_text(encoding="utf-8"))
        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
