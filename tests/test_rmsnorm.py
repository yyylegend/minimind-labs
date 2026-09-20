import unittest

import torch

from minimind_lab import RMSNorm


class RMSNormTest(unittest.TestCase):
    def test_keeps_shape_and_normalizes_last_dimension(self) -> None:
        norm = RMSNorm(dim=4)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]])

        y = norm(x)

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.allclose(y.pow(2).mean(dim=-1), torch.ones(2), atol=1e-4))

    def test_backward_produces_finite_gradients(self) -> None:
        norm = RMSNorm(dim=4)
        x = torch.randn(2, 3, 4, requires_grad=True)

        loss = norm(x).square().mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(norm.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
