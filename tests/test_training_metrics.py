import math
import unittest

import torch

from trainer.metrics import TrainingMetrics


class TrainingMetricsTest(unittest.TestCase):
    def test_metrics_distinguish_raw_valid_and_target_tokens(self):
        metrics = TrainingMetrics(start_step=2000, window_updates=2, start_time=100.0)
        input_ids = torch.zeros((2, 4), dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        labels = torch.tensor([[-100, 5, 6, -100], [-100, 7, -100, -100]])

        metrics.record_batch(input_ids, attention_mask, labels)
        snapshot = metrics.record_update(step=2001, total_steps=2010, now=102.0)

        self.assertEqual(snapshot.raw_tokens_per_second, 4.0)
        self.assertEqual(snapshot.valid_tokens_per_second, 2.5)
        self.assertEqual(snapshot.target_tokens_per_second, 1.5)
        self.assertAlmostEqual(snapshot.padding_ratio, 3 / 8)
        self.assertAlmostEqual(snapshot.target_token_ratio, 3 / 5)
        self.assertEqual(snapshot.updates_per_second, 0.5)
        self.assertEqual(snapshot.eta_seconds, 18.0)

    def test_resume_rate_uses_only_current_run_updates(self):
        metrics = TrainingMetrics(start_step=46320, window_updates=100, start_time=0.0)
        input_ids = torch.zeros((1, 4), dtype=torch.long)
        mask = torch.ones_like(input_ids)

        metrics.record_batch(input_ids, mask, input_ids)
        snapshot = metrics.record_update(step=46321, total_steps=50000, now=2.0)

        self.assertEqual(snapshot.updates_per_second, 0.5)
        self.assertTrue(math.isfinite(snapshot.eta_seconds))
        self.assertGreater(snapshot.eta_seconds, 1000)

    def test_overflow_counter_is_reported(self):
        metrics = TrainingMetrics(start_step=0, start_time=0.0)
        metrics.record_fp16_overflow()
        metrics.record_fp16_overflow()
        metrics.record_batch(torch.zeros((1, 2), dtype=torch.long), torch.ones((1, 2)), torch.ones((1, 2)))

        snapshot = metrics.record_update(step=1, total_steps=2, now=1.0)

        self.assertEqual(snapshot.fp16_overflow_total, 2)


if __name__ == "__main__":
    unittest.main()
