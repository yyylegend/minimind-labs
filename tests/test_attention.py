import unittest

import torch

from minimind_lab import Attention, precompute_freqs_cis, repeat_kv


class AttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.hidden_size = 8
        self.sequence_length = 4
        self.cos, self.sin = precompute_freqs_cis(dim=2, end=self.sequence_length)

    def test_repeat_kv_expands_only_the_head_dimension(self) -> None:
        x = torch.randn(2, 3, 2, 4)

        y = repeat_kv(x, n_rep=2)

        self.assertEqual(y.shape, (2, 3, 4, 4))
        self.assertTrue(torch.equal(y[:, :, 0], x[:, :, 0]))
        self.assertTrue(torch.equal(y[:, :, 1], x[:, :, 0]))
        self.assertTrue(torch.equal(y[:, :, 2], x[:, :, 1]))

    def test_output_shape_and_gradients(self) -> None:
        attention = Attention(hidden_size=8, num_attention_heads=4, num_key_value_heads=2)
        x = torch.randn(2, self.sequence_length, self.hidden_size, requires_grad=True)

        output, cache = attention(x, (self.cos, self.sin), use_cache=True)
        output.square().mean().backward()

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(cache[0].shape, (2, self.sequence_length, 2, 2))
        self.assertEqual(cache[1].shape, (2, self.sequence_length, 2, 2))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_causal_mask_blocks_future_tokens(self) -> None:
        attention = Attention(hidden_size=8, num_attention_heads=4, num_key_value_heads=2)
        attention.eval()
        x = torch.randn(1, self.sequence_length, self.hidden_size)
        changed_future = x.clone()
        changed_future[:, -1] += 100.0

        output, _ = attention(x, (self.cos, self.sin))
        changed_output, _ = attention(changed_future, (self.cos, self.sin))

        self.assertTrue(torch.allclose(output[:, :-1], changed_output[:, :-1], atol=1e-5))

    def test_incremental_kv_cache_matches_full_forward(self) -> None:
        attention = Attention(hidden_size=8, num_attention_heads=4, num_key_value_heads=2)
        attention.eval()
        x = torch.randn(1, self.sequence_length, self.hidden_size)

        full_output, _ = attention(x, (self.cos, self.sin))
        _, cache = attention(x[:, :2], (self.cos[:2], self.sin[:2]), use_cache=True)
        next_output, next_cache = attention(
            x[:, 2:],
            (self.cos[2:], self.sin[2:]),
            past_key_value=cache,
            use_cache=True,
        )

        self.assertTrue(torch.allclose(next_output, full_output[:, 2:], atol=1e-5))
        self.assertEqual(next_cache[0].shape[1], self.sequence_length)

    def test_fused_attention_matches_manual_attention(self) -> None:
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            self.skipTest("当前 PyTorch 没有 scaled_dot_product_attention")

        manual = Attention(
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            flash_attn=False,
        )
        fused = Attention(
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            flash_attn=True,
        )
        fused.load_state_dict(manual.state_dict())
        manual.eval()
        fused.eval()
        x = torch.randn(1, self.sequence_length, self.hidden_size)

        manual_output, _ = manual(x, (self.cos, self.sin))
        fused_output, _ = fused(x, (self.cos, self.sin))

        self.assertTrue(torch.allclose(manual_output, fused_output, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
