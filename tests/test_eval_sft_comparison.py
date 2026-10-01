import unittest

import torch
from transformers.tokenization_utils_base import BatchEncoding

from scripts.eval_sft_comparison import _generate


class FakeTokenizer:
    def __init__(self, prompt_encoding):
        self.prompt_encoding = prompt_encoding

    def apply_chat_template(self, *args, **kwargs):
        return self.prompt_encoding

    def decode(self, token_ids, skip_special_tokens=True):
        return "answer"


class FakeModel:
    def generate(self, input_ids, **kwargs):
        self.input_ids = input_ids
        torch.ones_like(input_ids)
        next_token = torch.tensor([[3]], device=input_ids.device)
        return torch.cat((input_ids, next_token), dim=1)


class EvalSFTComparisonTest(unittest.TestCase):
    def test_generate_unwraps_batch_encoding(self) -> None:
        prompt_encoding = BatchEncoding(data={"input_ids": torch.tensor([[1, 2]])})
        model = FakeModel()

        answer = _generate(
            model,
            FakeTokenizer(prompt_encoding),
            {"eval_prompt": "question"},
            torch.device("cpu"),
            max_new_tokens=1,
        )

        self.assertEqual(answer, "answer")
        self.assertTrue(torch.is_tensor(model.input_ids))
        self.assertEqual(tuple(model.input_ids.shape), (1, 2))

    def test_generate_keeps_tensor_return_from_chat_template(self) -> None:
        prompt_ids = torch.tensor([[1, 2]])
        model = FakeModel()

        answer = _generate(
            model,
            FakeTokenizer(prompt_ids),
            {"eval_prompt": "question"},
            torch.device("cpu"),
            max_new_tokens=1,
        )

        self.assertEqual(answer, "answer")
        self.assertTrue(torch.is_tensor(model.input_ids))
        self.assertEqual(tuple(model.input_ids.shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
