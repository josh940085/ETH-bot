import os
import unittest

os.environ["ETH_BOT_DISABLE_LIVE"] = "1"

import eth


class TrimRecentNewsKeysTests(unittest.TestCase):
    def test_keeps_most_recently_inserted_keys(self):
        seen = {}
        for i in range(500):
            seen[f"headline_{i:04d}"] = True

        trimmed = eth._trim_recent_news_keys(seen, max_keep=200)

        self.assertEqual(len(trimmed), 200)
        # The 200 most-recently-inserted keys are headline_0300..headline_0499.
        expected = {f"headline_{i:04d}" for i in range(300, 500)}
        self.assertEqual(set(trimmed.keys()), expected)

    def test_reinserting_an_existing_key_does_not_change_its_order(self):
        # Plain dict assignment to an existing key updates the value in
        # place without moving it - this is FIFO-by-first-sighting, which is
        # what run_bot wants: a headline inspected again is a no-op, it
        # should not artificially extend that headline's retention.
        seen = {}
        for i in range(500):
            seen[f"headline_{i:04d}"] = True
        seen["headline_0010"] = True

        trimmed = eth._trim_recent_news_keys(seen, max_keep=200)

        self.assertNotIn("headline_0010", trimmed)

    def test_does_not_mutate_input(self):
        seen = {f"headline_{i}": True for i in range(10)}
        original = dict(seen)

        eth._trim_recent_news_keys(seen, max_keep=5)

        self.assertEqual(seen, original)

    def test_max_keep_zero_returns_empty(self):
        seen = {"a": True, "b": True}
        self.assertEqual(eth._trim_recent_news_keys(seen, max_keep=0), {})


if __name__ == "__main__":
    unittest.main()
