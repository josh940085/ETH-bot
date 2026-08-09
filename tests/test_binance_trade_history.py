import datetime as dt
import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import binance_trade_history


def _archive_bytes(day=1, include_header=False):
    start = int(dt.datetime(2020, 1, day, tzinfo=dt.timezone.utc).timestamp() * 1000)
    rows = []
    if include_header:
        rows.append(
            "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker"
        )
    rows.extend(
        [
            f"1,100.0,2.0,1,1,{start + 1000},true",
            f"2,101.0,1.0,2,2,{start + 2000},false",
        ]
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"BTCUSDT-aggTrades-2020-01-{day:02d}.csv", "\n".join(rows))
    return buffer.getvalue()


class BinanceTradeHistoryTests(unittest.TestCase):
    def test_validates_headerless_archive_and_checksum(self):
        payload = _archive_bytes()
        checksum = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trades.zip"
            path.write_bytes(payload)
            self.assertTrue(
                binance_trade_history.validate_trade_archive(
                    path, "aggTrades", 2020, 1, day=1, checksum=checksum
                )
            )
            self.assertFalse(
                binance_trade_history.validate_trade_archive(
                    path, "aggTrades", 2020, 1, day=1, checksum="0" * 64
                )
            )

    def test_download_requires_official_checksum_match(self):
        payload = _archive_bytes()
        checksum = hashlib.sha256(payload).hexdigest()
        response_checksum = mock.Mock(status_code=200, text=f"{checksum}  file.zip")
        response_checksum.raise_for_status.return_value = None
        response_archive = mock.Mock(status_code=200, content=payload)
        response_archive.raise_for_status.return_value = None
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            binance_trade_history, "trade_archive_cache_path", return_value=Path(tmpdir) / "file.zip"
        ), mock.patch.object(
            binance_trade_history.HTTP_SESSION,
            "get",
            side_effect=[response_checksum, response_archive],
        ):
            path = binance_trade_history.download_trade_archive(
                "BTCUSDT", "aggTrades", 2020, 1, day=1
            )

        self.assertEqual(path.name, "file.zip")


if __name__ == "__main__":
    unittest.main()
