import json
import tempfile
import unittest
from pathlib import Path

from dataset.lm_dataset import RLAIFDataset


class TinyChatTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        open_thinking=False,
        tools=None,
    ):
        text = "".join(f"<{message['role']}>{message['content']}<eos>" for message in messages)
        if add_generation_prompt:
            text += "<assistant><think>" if open_thinking else "<assistant>"
        return text


class RLAIFDatasetTest(unittest.TestCase):
    def test_prompt_uses_context_but_not_the_reference_assistant_answer(self):
        row = {
            "source": "rlaif",
            "gt": [],
            "conversations": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "4"},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory, "rlaif.jsonl")
            data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = RLAIFDataset(
                str(data_path), TinyChatTokenizer(), thinking_ratio=0.0
            )

        sample = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertIn("What is 2 + 2?", sample["prompt"])
        self.assertIn("<assistant>", sample["prompt"])
        self.assertNotIn("4", sample["prompt"])
        self.assertEqual(sample["messages"][-1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
