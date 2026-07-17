import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import program


class RealOrderPriorityTests(unittest.TestCase):
    def test_background_work_is_deferred_for_open_binance_position(self):
        payload = {"open": True, "position_source": "binance"}
        env = {"REAL_ORDER_PRIORITY_ENABLED": "1", "BINANCE_REAL_COPY_ENABLED": "1"}
        path = SimpleNamespace(read_text=lambda **_: json.dumps(payload))
        with patch.object(program, "POSITION_PANEL_PATH", path):
            self.assertTrue(program._real_execution_busy(env))

    def test_background_work_is_deferred_while_order_is_submitting(self):
        payload = {"open": False, "strategy_execution_status": "submitting"}
        env = {"REAL_ORDER_PRIORITY_ENABLED": "1", "BINANCE_REAL_COPY_ENABLED": "1"}
        path = SimpleNamespace(read_text=lambda **_: json.dumps(payload))
        with patch.object(program, "POSITION_PANEL_PATH", path):
            self.assertTrue(program._real_execution_busy(env))

    def test_background_work_runs_when_real_execution_is_idle(self):
        payload = {"open": False, "strategy_execution_status": "waiting"}
        env = {"REAL_ORDER_PRIORITY_ENABLED": "1", "BINANCE_REAL_COPY_ENABLED": "1"}
        path = SimpleNamespace(read_text=lambda **_: json.dumps(payload))
        with patch.object(program, "POSITION_PANEL_PATH", path):
            self.assertFalse(program._real_execution_busy(env))

    def test_running_background_job_is_preempted_once(self):
        process = SimpleNamespace(
            poll=lambda: None,
            terminate=unittest.mock.Mock(),
        )
        with patch("builtins.print"):
            self.assertTrue(program._preempt_background_process_for_real_order(process, "回測"))
            self.assertFalse(program._preempt_background_process_for_real_order(process, "回測"))
        process.terminate.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
