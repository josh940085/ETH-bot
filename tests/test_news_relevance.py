import unittest

import eth


class NewsRelevanceTests(unittest.TestCase):
    def test_rejects_recent_non_market_pushes(self):
        headlines = [
            "DOJ sentences two former TD Bank employees to prison",
            "Standard Nuclear prices IPO at $15 per share",
            "Standard Nuclear prices IPO at $15 per share on NYSE",
            "AMD: UBS hikes price target ahead of AI event on stronger GPU outlook",
            "Ex-Fed advisor gets over three years in prison for lying about China ties",
            "Form 4 Clear Secure Inc For: 15 July",
            "Warren Buffett initiated Berkshire Hathaway's investment in Alphabet",
            "Whether chipmakers can keep gaining remains uncertain",
            "Bitcoin stalls below $65,600 resistance: Live levels",
            "Ethereum price prediction for this weekend",
        ]
        for headline in headlines:
            with self.subTest(headline=headline):
                self.assertEqual(eth._news_relevance_reason(headline), "")

    def test_keeps_news_that_can_move_eth(self):
        expected = {
            "Bitcoin turns lower as soft U.S. inflation data is offset by Iran tensions": "crypto",
            "Ethereum ETF inflows hit a record high": "crypto",
            "US CPI rises less than expected": "macro",
            "Fed signals rates may stay higher as inflation persists": "macro",
            "US stock futures fall as Treasury yields surge": "cross_asset",
            "Asian shares slump as inflation reshapes rate expectations": "cross_asset",
            "Oil rises as Iran threatens Hormuz disruption": "geopolitical",
            "Russia-Ukraine ceasefire talks collapse amid escalation": "geopolitical",
        }
        for headline, reason in expected.items():
            with self.subTest(headline=headline):
                self.assertEqual(eth._news_relevance_reason(headline), reason)

    def test_dedupe_ignores_source_and_exchange_suffix(self):
        first = "[Investing] Standard Nuclear prices IPO at $15 per share"
        second = "[Investing Crypto] Standard Nuclear prices IPO at $15 per share on NYSE"
        self.assertEqual(eth._news_dedupe_key(first), eth._news_dedupe_key(second))


if __name__ == "__main__":
    unittest.main()
