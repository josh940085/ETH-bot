import json
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

    def test_missing_account_equity_does_not_trigger_false_intraday_stop(self):
        previous = {
            "month_key": "2026-08",
            "day_key": "2026-08-05",
            "month_start_equity": 1000.0,
            "day_start_equity": 1000.0,
            "current_equity": 1000.0,
        }
        state = monthly5_shadow.update_shadow_state(
            previous,
            now_ts=taipei_ts("2026-08-05T16:05:00"),
            wallet_balance=0.0,
            margin_balance=0.0,
            unrealized_pnl=-0.04,
            position_open=True,
            position_side="long",
            position_notional=64.0,
        )

        self.assertFalse(state["equity_valid"])
        self.assertEqual(state["current_equity"], 1000.0)
        self.assertEqual(state["monthly_pnl_pct"], 0.0)
        self.assertEqual(state["intraday_pnl_pct"], 0.0)
        self.assertFalse(state["intraday_stop_active"])
        self.assertEqual(state["mode"], "normal")

    def test_persistence_round_trip(self):
        payload = {"strategy_id": monthly5_shadow.STRATEGY_ID, "mode": "normal"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shadow.json"
            monthly5_shadow.save_state(path, payload)
            self.assertEqual(monthly5_shadow.load_state(path), payload)

    def test_history_append_records_shadow_decision_and_dedupes(self):
        snapshot = monthly5_shadow.update_shadow_state(
            {},
            now_ts=taipei_ts("2026-08-05T12:00:00"),
            margin_balance=1000.0,
            mark_price=112000.0,
        )
        selection = monthly5_shadow.build_market_selection(
            snapshot,
            strategy_signal="wait",
            strategy_context={"htf": 1, "mid_trend": 1, "macro_bias": 1.0},
            host_logic={"direction": "long", "confidence": 0.8},
            macro_alignment={"score": 2.0, "hard_block": False},
        )
        snapshot["market_selection"] = selection
        guard = monthly5_shadow.build_execution_guard(snapshot, direction="wait", requested_size=0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.jsonl"
            self.assertTrue(monthly5_shadow.append_history(path, snapshot, guard, min_interval_sec=300))
            self.assertFalse(monthly5_shadow.append_history(path, snapshot, guard, min_interval_sec=300))
            rows = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(rows), 1)
        self.assertIn('"selected_plan": "normal_long_selector"', rows[0])
        self.assertIn('"guard_allowed": true', rows[0])

    def test_history_append_carries_active_selection_for_open_position_wait_row(self):
        active = monthly5_shadow.update_shadow_state(
            {},
            now_ts=taipei_ts("2026-08-05T12:00:00"),
            margin_balance=1000.0,
            mark_price=112000.0,
            position_open=True,
            position_side="long",
            position_notional=100.0,
        )
        active["market_selection"] = {
            "market_bias": "bullish",
            "market_state": "chop",
            "selected_plan": "normal_long_selector",
            "shadow_action": "evaluate_long",
            "exposure_cap": 0.35,
            "strategy_signal": "long",
        }
        wait = monthly5_shadow.update_shadow_state(
            active,
            now_ts=taipei_ts("2026-08-05T12:05:01"),
            margin_balance=1001.0,
            mark_price=112100.0,
            position_open=True,
            position_side="long",
            position_notional=100.0,
        )
        wait["market_selection"] = {
            "market_bias": "neutral",
            "market_state": "",
            "selected_plan": "normal_wait",
            "shadow_action": "wait",
            "exposure_cap": 1.0,
            "strategy_signal": "wait",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.jsonl"
            self.assertTrue(monthly5_shadow.append_history(path, active, {"allowed": True}, min_interval_sec=0))
            self.assertTrue(monthly5_shadow.append_history(path, wait, {"allowed": True}, min_interval_sec=0))
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(rows[-1]["selected_plan"], "normal_long_selector")
        self.assertEqual(rows[-1]["shadow_action"], "evaluate_long")
        self.assertEqual(rows[-1]["strategy_signal"], "long")
        self.assertEqual(rows[-1]["exposure_cap"], 0.35)

    def test_history_append_infers_side_for_open_position_wait_row(self):
        active = monthly5_shadow.update_shadow_state(
            {},
            now_ts=taipei_ts("2026-08-05T12:00:00"),
            margin_balance=1000.0,
            mark_price=112000.0,
            position_open=True,
            position_side="long",
            position_notional=100.0,
        )
        active["market_selection"] = {
            "market_bias": "bullish",
            "market_state": "chop",
            "selected_plan": "normal_long_selector",
            "shadow_action": "evaluate_long",
            "exposure_cap": 0.35,
            "strategy_signal": "long",
        }
        wait = monthly5_shadow.update_shadow_state(
            active,
            now_ts=taipei_ts("2026-08-05T12:05:01"),
            margin_balance=1001.0,
            mark_price=112100.0,
            position_open=True,
            position_side="",
            position_notional=100.0,
        )
        wait["market_selection"] = {
            "selected_plan": "normal_wait",
            "shadow_action": "wait",
            "exposure_cap": 1.0,
            "strategy_signal": "wait",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.jsonl"
            monthly5_shadow.append_history(path, active, {"allowed": True}, min_interval_sec=0)
            monthly5_shadow.append_history(path, wait, {"allowed": True}, min_interval_sec=0)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(rows[-1]["position_side"], "long")
        self.assertEqual(rows[-1]["selected_plan"], "normal_long_selector")
        self.assertEqual(rows[-1]["shadow_action"], "evaluate_long")

    def test_market_selection_chooses_normal_long_for_bullish_context(self):
        selection = monthly5_shadow.build_market_selection(
            {"mode": "normal", "suggested_exposure_scale": 1.0, "max_leverage": 5},
            strategy_signal="wait",
            strategy_context={"htf": 1, "mid_trend": 1, "macro_bias": 1.2},
            host_logic={"direction": "long", "confidence": 0.8},
            macro_alignment={"score": 2.0, "hard_block": False},
            donchian_state={"state": "trend", "action": "open"},
        )

        self.assertEqual(selection["market_bias"], "bullish")
        self.assertEqual(selection["selected_plan"], "normal_long_selector")
        self.assertEqual(selection["shadow_action"], "evaluate_long")
        self.assertEqual(selection["exposure_cap"], 1.0)

    def test_market_selection_uses_recovery_long_flat_only_when_bullish(self):
        selection = monthly5_shadow.build_market_selection(
            {"mode": "recovery", "suggested_exposure_scale": 0.5, "max_leverage": 5},
            strategy_signal="wait",
            strategy_context={"htf": 1, "mid_trend": 1, "macro_bias": 1.0},
            host_logic={"direction": "long", "confidence": 0.76},
            macro_alignment={"score": 1.5, "hard_block": False},
        )

        self.assertEqual(selection["selected_plan"], "recovery_long_flat_selector")
        self.assertEqual(selection["shadow_action"], "evaluate_long")
        self.assertEqual(selection["exposure_cap"], 0.5)

    def test_market_selection_risk_off_overrides_bullish_context(self):
        selection = monthly5_shadow.build_market_selection(
            {
                "mode": "post_lock_floor_guard",
                "suggested_exposure_scale": 0.0,
                "max_leverage": 5,
                "reason_codes": ["post_lock_floor_guard"],
            },
            strategy_signal="long",
            strategy_context={"htf": 1, "mid_trend": 1, "macro_bias": 1.0},
            host_logic={"direction": "long", "confidence": 0.9},
            macro_alignment={"score": 2.0, "hard_block": False},
        )

        self.assertEqual(selection["selected_plan"], "risk_off")
        self.assertEqual(selection["shadow_action"], "risk_off")
        self.assertEqual(selection["exposure_cap"], 0.0)

    def test_market_selection_reduces_exposure_in_chop(self):
        selection = monthly5_shadow.build_market_selection(
            {"mode": "normal", "suggested_exposure_scale": 1.0, "max_leverage": 5},
            strategy_signal="wait",
            strategy_context={"htf": 1, "mid_trend": 1, "macro_bias": 1.2},
            host_logic={"direction": "long", "confidence": 0.8},
            macro_alignment={"score": 2.0, "hard_block": False},
            donchian_state={"state": "chop", "action": "reduce"},
        )

        self.assertEqual(selection["selected_plan"], "normal_long_selector")
        self.assertEqual(selection["exposure_cap"], 0.35)
        self.assertIn("chop_market_reduce", selection["reason_codes"])

    def test_market_selection_waits_on_underperforming_plan_key(self):
        selection = monthly5_shadow.build_market_selection(
            {"mode": "normal", "suggested_exposure_scale": 1.0, "max_leverage": 5},
            strategy_signal="wait",
            strategy_context={"htf": 1, "mid_trend": 1, "macro_bias": 1.2},
            host_logic={"direction": "long", "confidence": 0.8},
            macro_alignment={"score": 2.0, "hard_block": False},
            donchian_state={"state": "chop", "action": "reduce"},
            underperforming_plan_keys=[
                "normal_long_selector|evaluate_long|bullish|chop",
            ],
        )

        self.assertEqual(selection["selected_plan"], "underperforming_wait")
        self.assertEqual(selection["shadow_action"], "wait")
        self.assertEqual(selection["exposure_cap"], 0.0)
        self.assertEqual(selection["suppressed_plan"], "normal_long_selector")
        self.assertEqual(selection["suppressed_action"], "evaluate_long")
        self.assertEqual(selection["suppressed_key"], "normal_long_selector|evaluate_long|bullish|chop")
        self.assertEqual(selection["suppressed_exposure_cap"], 0.35)
        self.assertIn("underperforming_plan_wait", selection["reason_codes"])

    def test_market_selection_uses_recovery_probe_when_suppressed_key_recovers(self):
        selection = monthly5_shadow.build_market_selection(
            {"mode": "normal", "suggested_exposure_scale": 1.0, "max_leverage": 5},
            strategy_signal="wait",
            strategy_context={"htf": 1, "mid_trend": 1, "macro_bias": 1.2},
            host_logic={"direction": "long", "confidence": 0.8},
            macro_alignment={"score": 2.0, "hard_block": False},
            donchian_state={"state": "chop", "action": "reduce"},
            recovering_plan_keys=[
                "normal_long_selector|evaluate_long|bullish|chop",
            ],
        )

        self.assertEqual(selection["selected_plan"], "normal_long_selector")
        self.assertEqual(selection["shadow_action"], "evaluate_long")
        self.assertEqual(selection["exposure_cap"], monthly5_shadow.RECOVERY_PROBE_EXPOSURE_CAP)
        self.assertTrue(selection["recovery_probe"])
        self.assertEqual(selection["recovery_probe_key"], "normal_long_selector|evaluate_long|bullish|chop")
        self.assertIn("underperforming_recovery_probe", selection["reason_codes"])

    def test_market_selection_restores_full_cap_after_probe_success(self):
        selection = monthly5_shadow.build_market_selection(
            {"mode": "normal", "suggested_exposure_scale": 1.0, "max_leverage": 5},
            strategy_signal="wait",
            strategy_context={"htf": 1, "mid_trend": 1, "macro_bias": 1.2},
            host_logic={"direction": "long", "confidence": 0.8},
            macro_alignment={"score": 2.0, "hard_block": False},
            donchian_state={"state": "chop", "action": "reduce"},
            recovering_plan_keys=[
                "normal_long_selector|evaluate_long|bullish|chop",
            ],
            underperforming_plan_keys=[
                "normal_long_selector|evaluate_long|bullish|chop",
            ],
            probe_candidate_plan_keys=[
                "normal_long_selector|evaluate_long|bullish|chop",
            ],
            probe_success_plan_keys=[
                "normal_long_selector|evaluate_long|bullish|chop",
            ],
        )

        self.assertEqual(selection["selected_plan"], "normal_long_selector")
        self.assertEqual(selection["shadow_action"], "evaluate_long")
        self.assertEqual(selection["exposure_cap"], 0.35)
        self.assertFalse(selection["recovery_probe"])
        self.assertIn("underperforming_probe_success", selection["reason_codes"])
        self.assertNotIn("underperforming_micro_probe", selection["reason_codes"])
        self.assertNotIn("underperforming_plan_wait", selection["reason_codes"])

    def test_market_selection_allows_micro_probe_for_positive_half_probe_candidate(self):
        selection = monthly5_shadow.build_market_selection(
            {"mode": "normal", "suggested_exposure_scale": 1.0, "max_leverage": 5},
            strategy_signal="wait",
            strategy_context={"htf": 1, "mid_trend": 1, "macro_bias": 1.2},
            host_logic={"direction": "long", "confidence": 0.8},
            macro_alignment={"score": 2.0, "hard_block": False},
            donchian_state={"state": "chop", "action": "reduce"},
            underperforming_plan_keys=[
                "normal_long_selector|evaluate_long|bullish|chop",
            ],
            probe_candidate_plan_keys=[
                "normal_long_selector|evaluate_long|bullish|chop",
            ],
        )

        self.assertEqual(selection["selected_plan"], "normal_long_selector")
        self.assertEqual(selection["shadow_action"], "evaluate_long")
        self.assertEqual(selection["exposure_cap"], monthly5_shadow.UNDERPERFORMING_MICRO_PROBE_EXPOSURE_CAP)
        self.assertTrue(selection["recovery_probe"])
        self.assertEqual(selection["recovery_probe_key"], "normal_long_selector|evaluate_long|bullish|chop")
        self.assertEqual(selection["suppressed_key"], "")
        self.assertIn("underperforming_micro_probe", selection["reason_codes"])

    def test_market_selection_resumes_when_underperforming_key_expires(self):
        selection = monthly5_shadow.build_market_selection(
            {"mode": "normal", "suggested_exposure_scale": 1.0, "max_leverage": 5},
            strategy_signal="wait",
            strategy_context={"htf": 1, "mid_trend": 1, "macro_bias": 1.2},
            host_logic={"direction": "long", "confidence": 0.8},
            macro_alignment={"score": 2.0, "hard_block": False},
            donchian_state={"state": "chop", "action": "reduce"},
            underperforming_plan_keys=[],
        )

        self.assertEqual(selection["selected_plan"], "normal_long_selector")
        self.assertEqual(selection["shadow_action"], "evaluate_long")
        self.assertEqual(selection["exposure_cap"], 0.35)
        self.assertNotIn("underperforming_plan_wait", selection["reason_codes"])

    def test_execution_guard_blocks_risk_off(self):
        guard = monthly5_shadow.build_execution_guard(
            {
                "mode": "intraday_stop",
                "max_leverage": 5,
                "market_selection": {
                    "selected_plan": "risk_off",
                    "shadow_action": "risk_off",
                    "exposure_cap": 0.0,
                },
            },
            direction="long",
            requested_size=0.2,
        )

        self.assertFalse(guard["allowed"])
        self.assertEqual(guard["adjusted_size"], 0.0)
        self.assertEqual(guard["reason_code"], "monthly5_risk_off")

    def test_execution_guard_caps_allowed_size(self):
        guard = monthly5_shadow.build_execution_guard(
            {
                "mode": "normal",
                "max_leverage": 5,
                "market_selection": {
                    "selected_plan": "normal_long_selector",
                    "shadow_action": "evaluate_long",
                    "exposure_cap": 0.35,
                },
            },
            direction="long",
            requested_size=0.8,
        )

        self.assertTrue(guard["allowed"])
        self.assertTrue(guard["capped"])
        self.assertEqual(guard["adjusted_size"], 0.35)

    def test_execution_guard_blocks_direction_mismatch(self):
        guard = monthly5_shadow.build_execution_guard(
            {
                "mode": "normal",
                "max_leverage": 5,
                "market_selection": {
                    "selected_plan": "normal_long_selector",
                    "shadow_action": "evaluate_long",
                    "exposure_cap": 1.0,
                },
            },
            direction="short",
            requested_size=0.2,
        )

        self.assertFalse(guard["allowed"])
        self.assertEqual(guard["reason_code"], "monthly5_direction_mismatch")

    def test_position_guard_reduces_to_exposure_cap(self):
        guard = monthly5_shadow.build_position_guard(
            {
                "mode": "post_lock",
                "market_selection": {
                    "selected_plan": "post_lock_low_exposure",
                    "shadow_action": "reduced_exposure",
                    "exposure_cap": 0.15,
                },
            },
            current_size=0.5,
        )

        self.assertEqual(guard["action"], "reduce_to_cap")
        self.assertEqual(guard["target_size"], 0.15)
        self.assertEqual(guard["reduce_delta"], 0.35)

    def test_position_guard_closes_on_floor_guard(self):
        guard = monthly5_shadow.build_position_guard(
            {
                "mode": "post_lock_floor_guard",
                "market_selection": {
                    "selected_plan": "risk_off",
                    "shadow_action": "risk_off",
                    "exposure_cap": 0.0,
                },
            },
            current_size=0.5,
        )

        self.assertEqual(guard["action"], "close_all")
        self.assertEqual(guard["target_size"], 0.0)
        self.assertEqual(guard["reason_code"], "monthly5_close_all")

    def test_position_guard_holds_within_cap(self):
        guard = monthly5_shadow.build_position_guard(
            {
                "mode": "normal",
                "market_selection": {
                    "selected_plan": "normal_long_selector",
                    "shadow_action": "evaluate_long",
                    "exposure_cap": 0.5,
                },
            },
            current_size=0.2,
        )

        self.assertEqual(guard["action"], "hold")
        self.assertEqual(guard["target_size"], 0.2)


if __name__ == "__main__":
    unittest.main()
