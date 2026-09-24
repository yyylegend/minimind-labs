import unittest

import torch

from trainer.grpo_objective import cispo_loss, group_relative_advantages


class GroupRelativeAdvantagesTest(unittest.TestCase):
    def test_rewards_are_normalized_inside_each_prompt_group(self):
        rewards = torch.tensor([1.0, 3.0, 5.0, 10.0, 10.0, 10.0])

        advantages = group_relative_advantages(rewards, num_generations=3)

        torch.testing.assert_close(
            advantages,
            torch.tensor([-1.2247, 0.0, 1.2247, 0.0, 0.0, 0.0]),
            atol=2e-4,
            rtol=0,
        )


class CISPOLossTest(unittest.TestCase):
    def test_positive_and_negative_advantages_push_log_probability_in_opposite_directions(self):
        current_logps = torch.zeros((2, 2), requires_grad=True)
        old_logps = torch.zeros_like(current_logps)
        reference_logps = torch.zeros_like(current_logps)
        advantages = torch.tensor([1.0, -1.0])
        completion_mask = torch.ones_like(current_logps)

        loss = cispo_loss(
            current_logps,
            old_logps,
            reference_logps,
            advantages,
            completion_mask,
            beta=0.1,
            epsilon_high=5.0,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.all(current_logps.grad[0] < 0).item())
        self.assertTrue(torch.all(current_logps.grad[1] > 0).item())

    def test_padding_tokens_do_not_change_the_loss(self):
        current_logps = torch.tensor([[-0.5, -10.0]], requires_grad=True)
        old_logps = torch.zeros_like(current_logps)
        reference_logps = torch.zeros_like(current_logps)
        advantages = torch.tensor([1.0])

        with_padding = cispo_loss(
            current_logps,
            old_logps,
            reference_logps,
            advantages,
            torch.tensor([[1, 0]]),
        )
        single_token_loss = cispo_loss(
            torch.tensor([[-0.5]]),
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            advantages,
            torch.tensor([[1]]),
        )

        torch.testing.assert_close(with_padding, single_token_loss)
        self.assertTrue(torch.isfinite(with_padding).item())


if __name__ == "__main__":
    unittest.main()
