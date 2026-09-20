import unittest

import torch

from minimind_lab import MiniMindConfig, MiniMindForCausalLM


class CausalLMTest(unittest.TestCase):
    def make_model(self) -> MiniMindForCausalLM:
        config = MiniMindConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=16,
            max_position_embeddings=16,
            flash_attn=False,
        )
        return MiniMindForCausalLM(config)

    def test_forward_returns_logits_loss_and_finite_gradients(self) -> None:
        torch.manual_seed(0)
        model = self.make_model()
        input_ids = torch.tensor([[1, 4, 7, 2], [3, 5, 6, 8]])

        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()

        self.assertEqual(outputs.logits.shape, (2, 4, 32))
        self.assertEqual(outputs.loss.ndim, 0)
        self.assertTrue(torch.isfinite(outputs.loss))
        self.assertTrue(torch.isfinite(model.model.embed_tokens.weight.grad).all())

    def test_lm_head_can_share_embedding_weights(self) -> None:
        model = self.make_model()

        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

    def test_generation_appends_tokens_with_kv_cache(self) -> None:
        torch.manual_seed(0)
        model = self.make_model().eval()
        input_ids = torch.tensor([[1, 4, 7]])

        generated = model.generate(
            input_ids,
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=None,
        )

        self.assertEqual(generated.shape, (1, 6))
        self.assertTrue(torch.isfinite(generated.float()).all())


if __name__ == "__main__":
    unittest.main()
