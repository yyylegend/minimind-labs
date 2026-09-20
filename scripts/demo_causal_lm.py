import torch

from minimind_lab import MiniMindConfig, MiniMindForCausalLM


def main() -> None:
    torch.manual_seed(0)
    config = MiniMindConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=16,
        max_position_embeddings=32,
        flash_attn=False,
    )
    model = MiniMindForCausalLM(config)
    input_ids = torch.tensor([[1, 4, 7]])
    outputs = model(input_ids, labels=input_ids)
    generated = model.generate(input_ids, max_new_tokens=2, do_sample=False, eos_token_id=None)

    print(f"logits: {outputs.logits.shape}")
    print(f"loss: {outputs.loss.item():.4f}")
    print(f"generated ids: {generated.tolist()}")


if __name__ == "__main__":
    main()
