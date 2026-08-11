import os
import unittest
from unittest import mock

import runtime_config


class EnvFloatTests(unittest.TestCase):
    def test_returns_parsed_value_when_set(self):
        with mock.patch.dict(os.environ, {"X_FLOAT": "1.5"}, clear=True):
            self.assertEqual(runtime_config.env_float("X_FLOAT", 0.0), 1.5)

    def test_returns_default_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runtime_config.env_float("X_FLOAT", 2.5), 2.5)

    def test_returns_default_when_unparseable(self):
        with mock.patch.dict(os.environ, {"X_FLOAT": "not-a-number"}, clear=True):
            self.assertEqual(runtime_config.env_float("X_FLOAT", 3.0), 3.0)


class EnvIntTests(unittest.TestCase):
    def test_returns_parsed_value_when_set(self):
        with mock.patch.dict(os.environ, {"X_INT": "7"}, clear=True):
            self.assertEqual(runtime_config.env_int("X_INT", 0), 7)

    def test_strips_whitespace(self):
        with mock.patch.dict(os.environ, {"X_INT": "  9  "}, clear=True):
            self.assertEqual(runtime_config.env_int("X_INT", 0), 9)

    def test_returns_default_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runtime_config.env_int("X_INT", 4), 4)

    def test_returns_default_when_unparseable(self):
        with mock.patch.dict(os.environ, {"X_INT": "not-a-number"}, clear=True):
            self.assertEqual(runtime_config.env_int("X_INT", 5), 5)


class EnvBoolTests(unittest.TestCase):
    def test_returns_default_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(runtime_config.env_bool("X_BOOL", True))
            self.assertFalse(runtime_config.env_bool("X_BOOL", False))

    def test_recognizes_falsy_strings(self):
        for raw in ("0", "false", "no", "off", "FALSE", "Off"):
            with mock.patch.dict(os.environ, {"X_BOOL": raw}, clear=True):
                self.assertFalse(runtime_config.env_bool("X_BOOL", True))

    def test_recognizes_truthy_strings(self):
        for raw in ("1", "true", "yes", "on", "anything-else"):
            with mock.patch.dict(os.environ, {"X_BOOL": raw}, clear=True):
                self.assertTrue(runtime_config.env_bool("X_BOOL", False))


class IsTruthyTests(unittest.TestCase):
    def test_recognizes_truthy_values(self):
        for raw in ("1", "true", "yes", "on", "  YES  ", "On"):
            self.assertTrue(runtime_config.is_truthy(raw))

    def test_recognizes_falsy_values(self):
        for raw in ("0", "false", "no", "off", "", None, "maybe"):
            self.assertFalse(runtime_config.is_truthy(raw))


if __name__ == "__main__":
    unittest.main()
