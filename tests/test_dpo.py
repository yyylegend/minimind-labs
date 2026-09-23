import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from dataset.lm_dataset import DPODataset
from trainer.dpo_utils import dpo_loss


class TinyTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{message['role']}>{message['content']}<eos>\n" for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [ord(character) + 1 for character in text]}


class DPODatasetTest(unittest.TestCase):
    def test_pair_masks_prompt_and_keeps_both_preference_responses(self):
        row = {
            "chosen": [
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            "rejected": [
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "5"},
            ],
        }
        tokenizer = TinyTokenizer()
        prompt_text = tokenizer.apply_chat_template(
            row["chosen"][:-1], tokenize=False, add_generation_prompt=True
        )
        prompt_length = len(tokenizer(prompt_text)["input_ids"])
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory, "dpo.jsonl")
            data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = DPODataset(str(data_path), tokenizer, max_length=64)
            sample = dataset[0]

        self.assertEqual(sample["chosen_input_ids"].shape, (64,))
        self.assertEqual(sample["rejected_input_ids"].shape, (64,))
        self.assertGreater(sample["chosen_response_mask"].sum().item(), 0)
        self.assertGreater(sample["rejected_response_mask"].sum().item(), 0)
        self.assertEqual(sample["chosen_response_mask"][:prompt_length].sum().item(), 0)
        self.assertEqual(sample["rejected_response_mask"][:prompt_length].sum().item(), 0)
        self.assertGreater(sample["chosen_response_mask"][prompt_length:].sum().item(), 0)
        self.assertGreater(sample["rejected_response_mask"][prompt_length:].sum().item(), 0)
        self.assertNotEqual(
            sample["chosen_input_ids"].tolist(),
            sample["rejected_input_ids"].tolist(),
        )


class DPOLossTest(unittest.TestCase):
    def test_preferring_chosen_has_lower_loss_and_identical_policy_is_finite(self):
        reference_chosen = torch.tensor([-1.0])
        reference_rejected = torch.tensor([-1.0])
        preferred = dpo_loss(
            policy_chosen_logps=torch.tensor([-0.2]),
            policy_rejected_logps=torch.tensor([-2.0]),
            reference_chosen_logps=reference_chosen,
            reference_rejected_logps=reference_rejected,
            beta=0.1,
        )
        dispreferred = dpo_loss(
            policy_chosen_logps=torch.tensor([-2.0]),
            policy_rejected_logps=torch.tensor([-0.2]),
            reference_chosen_logps=reference_chosen,
            reference_rejected_logps=reference_rejected,
            beta=0.1,
        )
        unchanged = dpo_loss(
            policy_chosen_logps=reference_chosen,
            policy_rejected_logps=reference_rejected,
            reference_chosen_logps=reference_chosen,
            reference_rejected_logps=reference_rejected,
            beta=0.1,
        )

        self.assertTrue(torch.isfinite(preferred).item())
        self.assertLess(preferred.item(), dispreferred.item())
        self.assertTrue(math.isclose(unchanged.item(), math.log(2), rel_tol=1e-6))


if __name__ == "__main__":
    unittest.main()
