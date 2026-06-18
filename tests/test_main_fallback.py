import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from main import prepare_llm_fallback  # noqa: E402


class MainFallbackTest(unittest.TestCase):
    def test_prepare_llm_fallback_preserves_versioned_paper_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "ranked.json"
            output_path = tmp_path / "llm.json"
            input_path.write_text(
                json.dumps(
                    {
                        "papers": [
                            {
                                "id": "2606.12345v1",
                                "title": "A Useful Paper",
                                "abstract": "abstract",
                            }
                        ],
                        "queries": [
                            {
                                "paper_tag": "query:test",
                                "query_text": "useful papers",
                                "ranked": [
                                    {
                                        "paper_id": "2606.12345",
                                        "score": 0.75,
                                        "star_rating": 4,
                                    }
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            self.assertTrue(prepare_llm_fallback(str(input_path), str(output_path), reason="test"))
            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(data["llm_ranked"][0]["paper_id"], "2606.12345v1")
        self.assertEqual(data["llm_ranked"][0]["score"], 7.5)
        self.assertEqual(data["llm_ranked"][0]["matched_query_tag"], "query:test")
        self.assertEqual(data["llm_fallback"]["reason"], "test")


if __name__ == "__main__":
    unittest.main()
