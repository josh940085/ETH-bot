import datetime as dt
import unittest
from unittest.mock import patch

import program


class WeeklyBacktestScheduleTests(unittest.TestCase):
    def _ts(self, value):
        return dt.datetime.fromisoformat(value).timestamp()

    def test_backtest_defaults_to_saturday_morning(self):
        with patch.dict("os.environ", {}, clear=True):
            settings = program._get_backtest_settings()

        self.assertEqual(settings["weekly_weekday"], 5)
        self.assertEqual(settings["weekly_hour"], 6)
        self.assertEqual(settings["weekly_minute"], 0)

    def test_weekly_schedule_waits_until_saturday_morning(self):
        settings = {"weekly_weekday": 5, "weekly_hour": 6, "weekly_minute": 0, "startup_delay_sec": 600}
        now = self._ts("2026-08-07T10:00:00+08:00")

        scheduled = program._compute_initial_weekly_run_ts(settings, now_ts=now)

        self.assertEqual(
            dt.datetime.fromtimestamp(scheduled).astimezone().strftime("%Y-%m-%d %H:%M"),
            "2026-08-08 06:00",
        )

    def test_weekly_schedule_runs_after_saturday_time_when_not_completed(self):
        settings = {"weekly_weekday": 5, "weekly_hour": 6, "weekly_minute": 0, "startup_delay_sec": 600}
        now = self._ts("2026-08-08T06:30:00+08:00")

        scheduled = program._compute_initial_weekly_run_ts(settings, now_ts=now)

        self.assertEqual(scheduled, now + 600)

    def test_weekly_schedule_moves_to_next_week_after_completion(self):
        settings = {"weekly_weekday": 5, "weekly_hour": 6, "weekly_minute": 0, "startup_delay_sec": 600}
        now = self._ts("2026-08-08T07:00:00+08:00")
        completed_at = dt.datetime.fromisoformat("2026-08-08T06:45:00+08:00")

        scheduled = program._compute_initial_weekly_run_ts(settings, completed_at=completed_at, now_ts=now)

        self.assertEqual(
            dt.datetime.fromtimestamp(scheduled).astimezone().strftime("%Y-%m-%d %H:%M"),
            "2026-08-15 06:00",
        )

    def test_maintenance_smoke_backtest_only_runs_saturday_morning(self):
        settings = {"smoke_backtest_weekday": 5}

        self.assertFalse(program._maintenance_smoke_backtest_due(settings, now_ts=self._ts("2026-08-07T04:30:00+08:00")))
        self.assertTrue(program._maintenance_smoke_backtest_due(settings, now_ts=self._ts("2026-08-08T04:30:00+08:00")))
        self.assertFalse(program._maintenance_smoke_backtest_due(settings, now_ts=self._ts("2026-08-08T12:00:00+08:00")))


if __name__ == "__main__":
    unittest.main()
