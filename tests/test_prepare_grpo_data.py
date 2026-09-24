import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_grpo_data import prepare_grpo_data


class PrepareGRPODataTest(unittest.TestCase):
    def test_held_out_prompt_is_removed_without_modifying_raw_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "rlaif.jsonl"
            heldout_path = root / "heldout.jsonl"
            output_path = root / "train.jsonl"
            manifest_path = root / "manifest.json"
            raw_rows = [
                {
                    "conversations": [
                        {"role": "user", "content": "What is 2 + 2?"},
                        {"role": "assistant", "content": "reference answer"},
                    ]
                },
                {
                    "conversations": [
                        {"role": "user", "content": "Explain the moon."},
                        {"role": "assistant", "content": "another reference"},
                    ]
                },
            ]
            raw_bytes = "".join(json.dumps(row) + "\n" for row in raw_rows).encode("utf-8")
            raw_path.write_bytes(raw_bytes)
            heldout_path.write_text(
                json.dumps({"prompt": "  WHAT is 2 + 2?  "}) + "\n", encoding="utf-8"
            )

            result = prepare_grpo_data(
                str(raw_path),
                str(heldout_path),
                str(output_path),
                str(manifest_path),
                source_url="https://example.invalid/rlaif",
                source_revision="test-revision",
                source_license="test-license",
            )

            self.assertEqual(raw_path.read_bytes(), raw_bytes)
            self.assertEqual(result["rows_scanned"], 2)
            self.assertEqual(result["rows_removed"], 1)
            self.assertEqual(result["rows_kept"], 1)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8"))["conversations"][0]["content"], "Explain the moon.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["rows_removed"], 1)
            self.assertEqual(manifest["source_revision"], "test-revision")
            self.assertEqual(len(manifest["source_sha256"]), 64)
            self.assertEqual(len(manifest["training_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
