import unittest

from trainer.grpo_rewards import score_completions


class FixedRewardModel:
    def __init__(self, score):
        self.score = score
        self.calls = []

    def get_score(self, messages, response):
        self.calls.append((messages, response))
        return self.score


class GRPORewardTest(unittest.TestCase):
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
