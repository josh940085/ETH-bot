import os
import unittest
from unittest.mock import Mock, patch

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import n8n_client
import n8n_service
import news


class N8nNotificationTests(unittest.TestCase):
    def test_disabled_live_runtime_does_not_call_n8n(self):
        session = Mock()
        with patch.dict(os.environ, {"ETH_BOT_DISABLE_LIVE": "1", "N8N_NOTIFICATIONS_ENABLED": "1"}):
            result = n8n_client.post_n8n_notification(
                "telegram", {"chat_id": "1", "text": "test"}, session=session
            )

        self.assertIsNone(result)
        session.post.assert_not_called()

    def test_supported_destination_is_sent_to_local_webhook(self):
        response = Mock(status_code=200)
        session = Mock()
        session.post.return_value = response
        with patch.dict(
            os.environ,
            {
                "ETH_BOT_DISABLE_LIVE": "0",
                "N8N_NOTIFICATIONS_ENABLED": "1",
                "N8N_NOTIFICATION_URL": "http://127.0.0.1:5678/webhook/eth-bot-notifications",
            },
        ), patch.object(n8n_client, "_get_webhook_secret", return_value="test-secret"):
            result = n8n_client.post_n8n_notification(
                "discord_news", {"content": "test"}, wait_for_response=True, session=session
            )

        self.assertIs(result, response)
        body = session.post.call_args.kwargs["json"]
        self.assertEqual(body["destination"], "discord_news")
        self.assertTrue(body["wait_for_response"])
        self.assertEqual(body["secret"], "test-secret")

    def test_discord_uses_direct_fallback_when_n8n_is_unavailable(self):
        direct_response = Mock(status_code=200)
        direct_response.raise_for_status.return_value = None
        with (
            patch.object(news, "DISCORD_AUTO_DELETE_SEC", 0),
            patch.object(news, "post_n8n_notification", return_value=None),
            patch.object(news.HTTP_SESSION, "post", return_value=direct_response) as direct_post,
        ):
            news._post_discord_webhook("https://discord.test/webhook", "hello")

        direct_post.assert_called_once_with(
            "https://discord.test/webhook", json={"content": "hello"}, timeout=5
        )

    def test_discord_keeps_auto_delete_when_n8n_delivers(self):
        response = Mock(status_code=200)
        response.json.return_value = {"id": "discord-message-id"}
        with (
            patch.object(news, "DISCORD_NEWS", "https://discord.test/news"),
            patch.object(news, "DISCORD_AUTO_DELETE_SEC", 60),
            patch.object(news, "post_n8n_notification", return_value=response),
            patch.object(news, "_schedule_discord_message_delete") as schedule_delete,
            patch.object(news.HTTP_SESSION, "post") as direct_post,
        ):
            news._post_discord_webhook("https://discord.test/news", "hello")

        direct_post.assert_not_called()
        schedule_delete.assert_called_once_with(
            "https://discord.test/news", "discord-message-id", 60
        )

    def test_n8n_service_only_receives_notification_secrets(self):
        values = {
            "TELEGRAM_TOKEN": "telegram-secret",
            "DISCORD_WEBHOOK": "trade-secret",
            "DISCORD_NEWS": "news-secret",
            "BINANCE_API_KEY": "must-not-be-shared",
            "OPENAI_API_KEY": "must-not-be-shared",
        }
        with (
            patch.object(n8n_service, "read_local_env_values", return_value=values),
            patch.object(n8n_service, "_ensure_encryption_key", return_value="local-key"),
            patch.object(n8n_service, "_ensure_webhook_secret", return_value="webhook-key"),
        ):
            environment = n8n_service.build_n8n_environment()

        self.assertEqual(environment["TELEGRAM_TOKEN"], "telegram-secret")
        self.assertNotIn("BINANCE_API_KEY", environment)
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertEqual(environment["N8N_AI_ENABLED"], "false")
        self.assertIn("n8n-nodes-base.openAi", environment["NODES_EXCLUDE"])


if __name__ == "__main__":
    unittest.main()
