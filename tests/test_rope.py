import unittest

import torch

from minimind_lab import apply_rotary_pos_emb, precompute_freqs_cis


class RoPETest(unittest.TestCase):
    def test_precomputed_table_has_expected_values(self) -> None:
        cos, sin = precompute_freqs_cis(dim=4, end=6)

        self.assertEqual(cos.shape, (6, 4))
        self.assertEqual(sin.shape, (6, 4))
        self.assertTrue(torch.allclose(cos[0], torch.ones(4)))
        self.assertTrue(torch.allclose(sin[0], torch.zeros(4)))
        self.assertTrue(torch.allclose(cos.square() + sin.square(), torch.ones_like(cos), atol=1e-6))

    def test_rotation_keeps_shape_and_vector_norm(self) -> None:
        cos, sin = precompute_freqs_cis(dim=4, end=3)
        q = torch.randn(2, 3, 2, 4)
        k = torch.randn(2, 3, 1, 4)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
        self.assertTrue(torch.isfinite(q_rot).all())
        self.assertTrue(torch.isfinite(k_rot).all())
        self.assertTrue(torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5))
        self.assertTrue(torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-5))

    def test_attention_score_depends_on_relative_distance(self) -> None:
        cos, sin = precompute_freqs_cis(dim=4, end=4, rope_base=100.0)
        q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]] * 4])
        k = torch.tensor([[[[2.0, 1.0, 4.0, 3.0]]] * 4])

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        score_distance_2_from_0 = (q_rot[0, 0, 0] * k_rot[0, 2, 0]).sum()
        score_distance_2_from_1 = (q_rot[0, 1, 0] * k_rot[0, 3, 0]).sum()

        self.assertTrue(torch.allclose(score_distance_2_from_0, score_distance_2_from_1, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
