import unittest
from unittest.mock import Mock, patch

import torch

from trainer.grpo_rewards import MiniMindRewardModel, score_completions


class FixedRewardModel:
    def __init__(self, score):
        self.score = score
        self.calls = []

    def get_score(self, messages, response):
        self.calls.append((messages, response))
        return self.score


class GRPORewardTest(unittest.TestCase):
    def test_reward_model_uses_slow_sentencepiece_tokenizer(self):
        tokenizer = object()
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model

        with (
            patch("trainer.grpo_rewards.AutoTokenizer.from_pretrained") as load_tokenizer,
            patch("trainer.grpo_rewards.AutoModel.from_pretrained", return_value=model),
        ):
            load_tokenizer.return_value = tokenizer
            reward_model = MiniMindRewardModel(
                "reward-model",
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        load_tokenizer.assert_called_once_with(
            "reward-model", trust_remote_code=True, use_fast=False
        )
        self.assertIs(reward_model.tokenizer, tokenizer)

    def test_reward_breakdown_matches_official_length_and_reward_model_terms(self):
        messages = [[{"role": "user", "content": "What is 2 + 2?"}]]
        reward_model = FixedRewardModel(1.5)

        results = score_completions(
            messages,
            ["short answer"],
            reward_model,
            num_generations=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].reward_model, 1.5)
        self.assertEqual(results[0].length, -0.5)
        self.assertEqual(results[0].thinking, 0.0)
        self.assertEqual(results[0].repetition_penalty, 0.0)
        self.assertEqual(results[0].total, 1.0)
        self.assertEqual(reward_model.calls[0][0], messages[0])
        self.assertEqual(reward_model.calls[0][1], "short answer")


if __name__ == "__main__":
    unittest.main()
