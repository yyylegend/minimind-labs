import unittest

import torch

from minimind_lab import FeedForward, SwiGLU


class SwiGLUTest(unittest.TestCase):
    def test_keeps_batch_and_sequence_shape(self) -> None:
        ffn = SwiGLU(hidden_size=8, intermediate_size=16)
        x = torch.randn(2, 3, 8)

        y = ffn(x)

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

    def test_backward_produces_finite_gradients(self) -> None:
        ffn = FeedForward(hidden_size=8, intermediate_size=16)
        x = torch.randn(2, 3, 8, requires_grad=True)

        ffn(x).square().mean().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for parameter in ffn.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_gate_branch_can_close_the_output(self) -> None:
        ffn = SwiGLU(hidden_size=8, intermediate_size=16)
        x = torch.randn(2, 3, 8)

        with torch.no_grad():
            ffn.gate_proj.weight.zero_()
        y = ffn(x)

        self.assertTrue(torch.allclose(y, torch.zeros_like(y)))


if __name__ == "__main__":
    unittest.main()
