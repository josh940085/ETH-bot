import json
import tempfile
import unittest
import datetime as dt
from pathlib import Path
from unittest.mock import patch

import package_restart
import package_updates
import program


class PackageUpdateTests(unittest.TestCase):
    def test_maintenance_defaults_package_updates_to_saturday(self):
        with patch.dict(program.os.environ, {}, clear=True):
            settings = program._get_maintenance_settings()
        self.assertTrue(settings["package_auto_update"])
        self.assertEqual(settings["package_update_weekday"], 5)

    def test_package_update_due_only_on_configured_weekday(self):
        timezone = dt.datetime.now().astimezone().tzinfo
        friday = dt.datetime(2026, 8, 7, 4, 30, tzinfo=timezone).timestamp()
        saturday = dt.datetime(2026, 8, 8, 4, 30, tzinfo=timezone).timestamp()
        settings = {"package_auto_update": True, "package_update_weekday": 5}

        self.assertFalse(program._maintenance_package_update_due(settings, friday))
        self.assertTrue(program._maintenance_package_update_due(settings, saturday))

    def test_package_update_can_be_disabled(self):
        settings = {"package_auto_update": False, "package_update_weekday": 5}
        self.assertFalse(program._maintenance_package_update_due(settings))

    def test_maintenance_command_updates_only_on_saturday(self):
        timezone = dt.datetime.now().astimezone().tzinfo
        settings = {
            "smoke_backtest_days": 3,
            "smoke_backtest_warmup_bars": 600,
            "smoke_backtest_weekday": 5,
            "package_auto_update": True,
            "package_update_weekday": 5,
            "notify": False,
        }
        friday = dt.datetime(2026, 8, 7, 4, 30, tzinfo=timezone).timestamp()
        saturday = dt.datetime(2026, 8, 8, 4, 30, tzinfo=timezone).timestamp()
        with patch.object(program.subprocess, "Popen") as popen:
            program._start_maintenance_process({}, settings, now_ts=friday)
            friday_command = popen.call_args.args[0]
            program._start_maintenance_process({}, settings, now_ts=saturday)
            saturday_command = popen.call_args.args[0]

        self.assertNotIn("--update-packages", friday_command)
        self.assertIn("--update-packages", saturday_command)

    def test_major_version_parser(self):
        self.assertEqual(package_updates._major_version("2.31.5"), 2)
        self.assertEqual(package_updates._major_version(""), -1)

    def test_package_names_are_unique(self):
        rows = [{"name": "pandas"}, {"name": "pandas"}, {"name": "numpy"}]
        self.assertEqual(package_updates._package_names(rows), ["pandas", "numpy"])

    def test_installed_python_versions_keeps_missing_package_visible(self):
        with patch.object(
            package_updates.importlib.metadata,
            "version",
            side_effect=["1.0", package_updates.importlib.metadata.PackageNotFoundError],
        ):
            self.assertEqual(
                package_updates._installed_python_versions(["present", "missing"]),
                {"present": "1.0", "missing": ""},
            )

    def test_package_restart_waits_if_position_state_is_unreadable(self):
        with patch.object(package_restart, "POSITION_PATH", Path("/missing/position.json")):
            self.assertTrue(package_restart._position_busy())

    def test_package_restart_allows_flat_confirmed_position(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            position_path = Path(temp_dir) / "position.json"
            position_path.write_text(
                json.dumps(
                    {
                        "open": False,
                        "position_source": "none",
                        "strategy_execution_status": "waiting",
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(package_restart, "POSITION_PATH", position_path):
                self.assertFalse(package_restart._position_busy())

    def test_package_restart_knows_regime_drift_service(self):
        self.assertIn("regime-drift", package_restart.SERVICE_ORDER)


if __name__ == "__main__":
    unittest.main()
