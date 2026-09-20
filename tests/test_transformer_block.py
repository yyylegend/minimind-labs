import unittest

import torch

from minimind_lab import TransformerBlock, precompute_freqs_cis


class TransformerBlockTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.hidden_size = 8
        self.sequence_length = 4
        self.num_heads = 4
        self.head_dim = self.hidden_size // self.num_heads
        self.cos, self.sin = precompute_freqs_cis(self.head_dim, self.sequence_length)

    def test_keeps_shape_and_backward_is_finite(self) -> None:
        block = TransformerBlock(
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=16,
        )
        x = torch.randn(2, self.sequence_length, self.hidden_size, requires_grad=True)

        y, cache = block(x, (self.cos, self.sin), use_cache=True)
        y.square().mean().backward()

        self.assertEqual(y.shape, x.shape)
        self.assertEqual(cache[0].shape, (2, self.sequence_length, 2, self.head_dim))
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_residual_paths_preserve_input_when_submodules_are_zero(self) -> None:
        block = TransformerBlock(
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=16,
        )
        x = torch.randn(1, self.sequence_length, self.hidden_size)

        with torch.no_grad():
            for parameter in block.parameters():
                parameter.zero_()
        y, _ = block(x, (self.cos, self.sin))

        self.assertTrue(torch.allclose(y, x))

    def test_attention_and_ffn_use_different_norms(self) -> None:
        block = TransformerBlock(
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=16,
        )

        self.assertIsNot(block.input_layernorm, block.post_attention_layernorm)


if __name__ == "__main__":
    unittest.main()
