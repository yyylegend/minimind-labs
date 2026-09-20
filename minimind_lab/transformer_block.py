"""TransformerBlock：把 Attention 和 FFN 组成一个完整的 Decoder block。"""

import torch
from torch import nn

from .attention import Attention
from .rmsnorm import RMSNorm
from .swiglu import FeedForward


class TransformerBlock(nn.Module):
    """MiniMind 风格的 Pre-Norm Transformer Block。"""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        num_key_value_heads: int | None = None,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-5,
        flash_attn: bool = True,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            dropout=dropout,
            rms_norm_eps=rms_norm_eps,
            flash_attn=flash_attn,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = FeedForward(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Pre-Norm：先归一化，再进入 Attention；最后加回原输入。
        residual = hidden_states
        attention_output, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + attention_output

        # 第二个子层同样是 Pre-Norm：先归一化，再进入 FFN，再做残差相加。
        residual = hidden_states
        ffn_output = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = residual + ffn_output
        return hidden_states, present_key_value
