import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
PRODUCTION_FILES = (
    REPO_DIR / "eth.py",
    REPO_DIR / "news.py",
    REPO_DIR / "mlx_learning.py",
    REPO_DIR / "maintenance.py",
)
PAID_OPENAI_MARKERS = (
    "api.openai.com",
    "OPENAI_API_KEY",
    "OPENAI_PAID_API_ENABLED",
    "OPENAI_CHAT_MODEL",
    "OPENAI_TRANSLATION_MODEL",
    "OPENAI_REASONING_EFFORT",
    "MLX_GPT_TEACHER",
    "gpt_teacher_review",
)


class NoPaidOpenAIFeaturesTests(unittest.TestCase):
    def test_production_code_has_no_paid_openai_path(self):
        for path in PRODUCTION_FILES:
            source = path.read_text(encoding="utf-8")
            for marker in PAID_OPENAI_MARKERS:
                with self.subTest(path=path.name, marker=marker):
                    self.assertNotIn(marker, source)


if __name__ == "__main__":
    unittest.main()
