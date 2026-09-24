import unittest

import torch

from minimind_lab import MiniMindConfig, MiniMindForCausalLM
from trainer.grpo_rollout import TorchRolloutEngine, compute_completion_logps, trim_completion


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 19

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None, return_tensors=None):
        ids = [1 + (ord(character) % 17) for character in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = [int(token) for token in token_ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in {self.pad_token_id, self.eos_token_id}]
        return " ".join(str(token) for token in tokens)


class GRPORolloutTest(unittest.TestCase):
    def test_completion_ends_at_the_first_eos_including_eos(self):
        completion = torch.tensor([3, 4, 19, 19, 19])

        trimmed = trim_completion(completion, eos_token_id=19)

        self.assertEqual(trimmed.tolist(), [3, 4, 19])

    def test_torch_rollout_generates_each_prompt_group_and_returns_old_logps(self):
        torch.manual_seed(7)
        tokenizer = TinyTokenizer()
        config = MiniMindConfig(
            vocab_size=20,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=12,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = MiniMindForCausalLM(config)
        engine = TorchRolloutEngine(
            model,
            tokenizer,
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_seq_len=8,
            max_gen_len=3,
            temperature=1.0,
        )

        result = engine.rollout(
            [{"prompt": "ab"}, {"prompt": "abc"}], num_generations=2
        )

        self.assertEqual(len(result.responses), 4)
        self.assertEqual(result.input_ids.shape[0], 4)
        self.assertEqual(result.old_logps.shape, result.completion_mask.shape)
        self.assertTrue(torch.isfinite(result.old_logps).all().item())
        self.assertTrue(torch.all(result.completion_mask.sum(dim=1) >= 1).item())
        self.assertTrue(torch.all(result.old_logps[result.completion_mask == 0] == 0).item())
        model.eval()
        with torch.no_grad():
            recomputed_logps = compute_completion_logps(
                model,
                result.input_ids,
                result.attention_mask,
                result.prompt_lengths,
                result.completion_mask,
            )
        torch.testing.assert_close(result.old_logps, recomputed_logps)


if __name__ == "__main__":
    unittest.main()
