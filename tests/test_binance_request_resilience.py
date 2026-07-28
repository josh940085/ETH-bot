import os
import unittest
from unittest.mock import patch

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth


class _Response:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.ok = 200 <= status_code < 300

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class BinanceRequestResilienceTests(unittest.TestCase):
    def setUp(self):
        eth.BINANCE_RATE_LIMIT_STATE.clear()

    def tearDown(self):
        eth.BINANCE_RATE_LIMIT_STATE.clear()

    def test_market_rate_limits_do_not_block_signed_orders(self):
        for status_code, retry_after, expected_cooldown in (
            (418, "30", 900.0),
            (429, "10", 10.0),
        ):
            with self.subTest(status_code=status_code):
                eth.BINANCE_RATE_LIMIT_STATE.clear()
                response = _Response(
                    status_code,
                    {"code": -1003},
                    {"Retry-After": retry_after},
                )
                with patch.object(eth.time, "time", return_value=1000.0):
                    eth._note_binance_rate_limit_response(
                        response,
                        prefix="Binance derivatives openInterest",
                    )
                    self.assertEqual(
                        eth._binance_rate_limit_remaining_sec("Binance futures order signed"),
                        0.0,
                    )
                    self.assertEqual(
                        eth._binance_rate_limit_remaining_sec("Binance derivatives openInterest"),
                        expected_cooldown,
                    )

    def test_gateway_get_retries_before_returning_success(self):
        responses = [
            _Response(502, {}),
            _Response(503, {}),
            _Response(200, {"ok": True}),
        ]
        with (
            patch.object(eth.HTTP_SESSION, "get", side_effect=responses) as request,
            patch.object(eth.time, "sleep"),
            patch.dict(os.environ, {"BINANCE_HTTP_GATEWAY_RETRIES": "2"}),
        ):
            response = eth._binance_request_get(
                "https://fapi.binance.com/fapi/v1/premiumIndex",
                prefix="Binance market",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(request.call_count, 3)

    def test_rate_limit_and_gateway_order_responses_are_reconciled_by_client_id(self):
        for status_code in (418, 429, 502, 503):
            with self.subTest(status_code=status_code):
                response = _Response(status_code, ValueError("html"))
                confirmed = {"orderId": 77, "status": "FILLED"}
                with (
                    patch.dict(
                        os.environ,
                        {"BINANCE_API_KEY": "test-key", "BINANCE_API_SECRET": "test-secret"},
                    ),
                    patch.object(eth, "_binance_request_post", return_value=response) as request,
                    patch.object(eth, "_query_binance_order_by_client_id", return_value=confirmed) as query,
                ):
                    payload = eth._binance_futures_signed_request(
                        "POST",
                        "/fapi/v1/order",
                        {
                            "symbol": "ETHUSDT",
                            "side": "BUY",
                            "type": "MARKET",
                            "quantity": 0.012,
                        },
                    )

                sent_params = request.call_args.kwargs["params"]
                client_id = sent_params["newClientOrderId"]
                self.assertTrue(client_id.startswith(eth.BINANCE_PROTECTION_CLIENT_PREFIX))
                self.assertLessEqual(len(client_id), 36)
                query.assert_called_once_with("ETHUSDT", client_id)
                self.assertTrue(payload["reconciled_after_error"])

    def test_signed_gateway_html_raises_structured_error(self):
        gateway_response = _Response(502, ValueError("html"))

        with (
            patch.dict(
                os.environ,
                {"BINANCE_API_KEY": "test-key", "BINANCE_API_SECRET": "test-secret"},
            ),
            patch.object(eth, "_binance_request_get", return_value=gateway_response),
        ):
            with self.assertRaises(eth.BinanceAPIRequestError) as raised:
                eth._binance_futures_signed_get(
                    "/fapi/v2/positionRisk",
                    {"symbol": "ETHUSDT"},
                )

        self.assertEqual(raised.exception.status_code, 502)


if __name__ == "__main__":
    unittest.main()
