import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import package_restart
import package_updates
import program


class PackageUpdateTests(unittest.TestCase):
    def test_daily_maintenance_enables_compatible_package_updates(self):
        with patch.dict(program.os.environ, {}, clear=True):
            settings = program._get_maintenance_settings()
        self.assertTrue(settings["package_auto_update"])

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


if __name__ == "__main__":
    unittest.main()
