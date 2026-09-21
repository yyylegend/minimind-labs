import json
import tempfile
import unittest
from pathlib import Path

from dataset.lm_dataset import _load_json_dataset, _conversation_features


class SFTDatasetSchemaTest(unittest.TestCase):
    def test_mixed_tool_fields_use_explicit_conversation_schema(self):
        rows = [
            {
                "conversations": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！"},
                ]
            },
            {
                "conversations": [
                    {"role": "system", "content": "", "tools": "[]", "tool_calls": ""},
                    {"role": "user", "content": "查天气"},
                    {"role": "assistant", "content": "好的。", "reasoning_content": ""},
                ]
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "sft.jsonl")
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

            dataset = _load_json_dataset(str(path), features=_conversation_features())

        self.assertEqual(len(dataset), 2)
        self.assertIn("conversations", dataset.features)


if __name__ == "__main__":
    unittest.main()
