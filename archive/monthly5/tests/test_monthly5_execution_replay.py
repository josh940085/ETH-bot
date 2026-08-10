import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import monthly5_execution_replay


class Monthly5ExecutionReplayTests(unittest.TestCase):
    def test_replays_long_with_fee_and_directional_slippage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            signal_path = Path(tmpdir) / "signals.csv"
            pd.DataFrame(
                [
                    {
                        "opened_at": "2020-01-01T00:00:01Z",
                        "closed_at": "2020-01-01T00:00:03Z",
                        "direction": "long",
                        "size": 2.0,
                        "entry": 100.0,
                        "exit": 102.0,
                    }
                ]
            ).to_csv(signal_path, index=False)
            ticks = pd.DataFrame(
                {
                    "event_time": pd.to_datetime(
                        ["2020-01-01T00:00:02Z", "2020-01-01T00:00:04Z"], utc=True
                    ),
                    "price": [101.0, 103.0],
                    "quantity": [10.0, 10.0],
                }
            )
            with mock.patch.object(
                monthly5_execution_replay.binance_trade_history,
                "load_trade_day",
                return_value=ticks,
            ):
                evidence, report = monthly5_execution_replay.replay_signals(
                    signal_path, taker_fee_rate=0.001, candidate="candidate_a"
                )

        self.assertTrue(report["complete"])
        self.assertEqual(len(evidence), 1)
        row = evidence.iloc[0]
        self.assertEqual(row["gross_pnl"], 4.0)
        self.assertAlmostEqual(row["fee"], 0.408)
        self.assertEqual(row["entry_slippage"], 2.0)
        self.assertEqual(row["exit_slippage"], -2.0)
        self.assertEqual(row["slippage"], 0.0)
        self.assertTrue(row["data_source"].startswith("binance_public_data_"))
        self.assertEqual(row["candidate"], "candidate_a")
        self.assertEqual(report["candidate"], "candidate_a")

    def test_rejects_aggregate_monthly_rows_without_signal_times(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            signal_path = Path(tmpdir) / "monthly.csv"
            pd.DataFrame([{"month": "2020-01", "return_pct": 6.0}]).to_csv(
                signal_path, index=False
            )
            with self.assertRaisesRegex(ValueError, "signal evidence missing columns"):
                monthly5_execution_replay.replay_signals(signal_path)


if __name__ == "__main__":
    unittest.main()
