import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import monthly5_shadow


def taipei_ts(value):
    return datetime.fromisoformat(value).replace(tzinfo=ZoneInfo("Asia/Taipei")).timestamp()


class Monthly5ShadowTests(unittest.TestCase):
    def test_initializes_month_and_day_equity(self):
        state = monthly5_shadow.update_shadow_state(
            {},
            now_ts=taipei_ts("2026-08-05T12:00:00"),
            margin_balance=1000.0,
            mark_price=112000.0,
        )

        self.assertEqual(state["month_key"], "2026-08")
        self.assertEqual(state["day_key"], "2026-08-05")
        self.assertEqual(state["month_start_equity"], 1000.0)
        self.assertEqual(state["day_start_equity"], 1000.0)
        self.assertEqual(state["mode"], "normal")
        self.assertEqual(state["suggested_exposure_scale"], 1.0)
        self.assertTrue(state["shadow_only"])

    def test_recovery_mode_triggers_below_monthly_loss_threshold(self):
        previous = {
            "month_key": "2026-08",
            "day_key": "2026-08-05",
            "month_start_equity": 1000.0,
            "day_start_equity": 950.0,
        }
        state = monthly5_shadow.update_shadow_state(
            previous,
            now_ts=taipei_ts("2026-08-05T13:00:00"),
            margin_balance=910.0,
        )

        self.assertEqual(state["mode"], "recovery")
        self.assertTrue(state["recovery_active"])
        self.assertEqual(state["suggested_exposure_scale"], 0.5)

    def test_post_lock_floor_guard_zeroes_shadow_exposure(self):
        locked = monthly5_shadow.update_shadow_state(
            {
                "month_key": "2026-08",
                "day_key": "2026-08-05",
                "month_start_equity": 1000.0,
                "day_start_equity": 1000.0,
            },
            now_ts=taipei_ts("2026-08-05T14:00:00"),
            margin_balance=1060.0,
        )
        guarded = monthly5_shadow.update_shadow_state(
            locked,
            now_ts=taipei_ts("2026-08-05T15:00:00"),
            margin_balance=1040.0,
        )

        self.assertEqual(locked["mode"], "post_lock")
        self.assertTrue(guarded["floor_guard_required"])
        self.assertEqual(guarded["mode"], "post_lock_floor_guard")
        self.assertEqual(guarded["suggested_exposure_scale"], 0.0)

    def test_intraday_stop_overrides_recovery_exposure(self):
        previous = {
            "month_key": "2026-08",
            "day_key": "2026-08-05",
            "month_start_equity": 1000.0,
            "day_start_equity": 1000.0,
        }
        state = monthly5_shadow.update_shadow_state(
            previous,
            now_ts=taipei_ts("2026-08-05T16:00:00"),
            margin_balance=920.0,
        )

        self.assertEqual(state["mode"], "intraday_stop")
        self.assertTrue(state["intraday_stop_active"])
        self.assertEqual(state["suggested_exposure_scale"], 0.0)

    def test_persistence_round_trip(self):
        payload = {"strategy_id": monthly5_shadow.STRATEGY_ID, "mode": "normal"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shadow.json"
            monthly5_shadow.save_state(path, payload)
            self.assertEqual(monthly5_shadow.load_state(path), payload)


if __name__ == "__main__":
    unittest.main()
