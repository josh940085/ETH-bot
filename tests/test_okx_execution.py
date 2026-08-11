import os
import unittest
from unittest.mock import patch

import okx_execution as okx


class _Response:
    status_code = 200
    def __init__(self, payload): self.payload = payload
    def json(self): return self.payload


class OKXExecutionTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"OKX_API_KEY": "key", "OKX_API_SECRET": "secret", "OKX_API_PASSPHRASE": "pass", "OKX_INST_ID": "BTC-USDT-SWAP"})
        self.env.start()
        self.addCleanup(self.env.stop)

    @patch("okx_execution.requests.request")
    def test_long_short_open_uses_contract_size_and_pos_side(self, request):
        request.side_effect = [
            _Response({"code": "0", "data": [{"ctVal": "0.01", "ctValCcy": "BTC", "lotSz": "0.01", "minSz": "0.01"}]}),
            _Response({"code": "0", "data": [{"posMode": "long_short_mode"}]}),
            _Response({"code": "0", "data": [{"ordId": "1", "sCode": "0"}]}),
        ]
        result = okx.place_market("long", 0.012)
        body = request.call_args.kwargs["data"]
        self.assertIn('"sz":"1.2"', body)
        self.assertIn('"posSide":"long"', body)
        self.assertEqual(result["base_qty"], 0.012)

    @patch("okx_execution.requests.request")
    def test_protection_is_mark_price_and_directional(self, request):
        request.side_effect = [
            _Response({"code": "0", "data": [{"ctVal": "0.01", "ctValCcy": "BTC", "lotSz": "0.01", "minSz": "0.01"}]}),
            _Response({"code": "0", "data": []}),
            _Response({"code": "0", "data": [{"posMode": "long_short_mode"}]}),
            _Response({"code": "0", "data": [{"sCode": "0"}]}),
            _Response({"code": "0", "data": [{"sCode": "0"}]}),
            _Response({"code": "0", "data": [{"algoClOrdId": "ethbottp1"}, {"algoClOrdId": "ethbotsl1"}]}),
        ]
        okx.install_protections("short", 0.01, 90000, 100000)
        bodies = [call.kwargs.get("data") or "" for call in request.call_args_list]
        self.assertTrue(any('"tpTriggerPxType":"mark"' in body and '"posSide":"short"' in body for body in bodies))
        self.assertTrue(any('"slTriggerPxType":"mark"' in body for body in bodies))
