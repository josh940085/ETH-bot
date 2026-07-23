import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import maintenance


class N8nMaintenanceTests(unittest.TestCase):
    def test_n8n_health_requires_running_service_and_health_endpoint(self):
        response = Mock(status_code=200)
        response.json.return_value = {"status": "ok"}
        command_result = SimpleNamespace(
            returncode=0,
            stdout="n8n RUNNING pid 123, uptime 0:01:00",
        )
        binary_path = Mock()
        binary_path.exists.return_value = True
        workflow_path = Mock(name="workflow_path")
        workflow_path.exists.return_value = True
        workflow_path.name = "eth-bot-notifications.json"

        with (
            patch.object(maintenance, "N8N_BINARY_PATH", binary_path),
            patch.object(maintenance, "N8N_WORKFLOW_PATH", workflow_path),
            patch.object(maintenance, "_run_command", return_value=command_result),
            patch.object(maintenance.requests, "get", return_value=response),
        ):
            result = maintenance._check_n8n_health()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["healthz"], 200)


if __name__ == "__main__":
    unittest.main()
