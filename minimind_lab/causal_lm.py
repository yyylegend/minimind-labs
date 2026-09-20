"""MiniMind 学习版 Causal Language Model。"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import MiniMindConfig
from .rmsnorm import RMSNorm
from .rope import precompute_freqs_cis
from .transformer_block import TransformerBlock


@dataclass
class CausalLMOutput:
    """比 HuggingFace 输出对象更简单的学习版输出容器。"""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
    hidden_states: torch.Tensor | None = None


class MiniMindModel(nn.Module):
    """Embedding、TransformerBlock 堆叠和最终 RMSNorm。"""

    def __init__(self, config: MiniMindConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    intermediate_size=config.intermediate_size,
                    dropout=config.dropout,
                    rms_norm_eps=config.rms_norm_eps,
                    flash_attn=config.flash_attn,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE 表不是模型参数，但需要跟着模型一起移动到 CPU/GPU。
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None] | None]:
        batch_size, sequence_length = input_ids.shape
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        if len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values 数量必须和 TransformerBlock 数量一致")

        past_length = 0
        if past_key_values[0] is not None:
            past_length = past_key_values[0][0].shape[1]
        if past_length + sequence_length > self.config.max_position_embeddings:
            raise ValueError("输入序列超过 max_position_embeddings")

        if attention_mask is not None:
            attention_mask = attention_mask.to(device=input_ids.device)

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        position_embeddings = (
            self.freqs_cos[past_length : past_length + sequence_length],
            self.freqs_sin[past_length : past_length + sequence_length],
        )

        presents = [] if use_cache else None
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            if use_cache:
                presents.append(present)

        return self.norm(hidden_states), presents


class MiniMindForCausalLM(nn.Module):
    """带词表投影、loss 和生成能力的 Decoder-only 语言模型。"""

    def __init__(self, config: MiniMindConfig | None = None) -> None:
        super().__init__()
        self.config = config or MiniMindConfig()
        self.model = MiniMindModel(self.config)
        self.model.apply(self._init_weights)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self._init_weights(self.lm_head)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        """使用 MiniMind/HuggingFace 风格的小方差初始化，避免初始 logits 过大。"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        use_cache: bool = False,
        labels: torch.Tensor | None = None,
        logits_to_keep: int = 0,
    ) -> CausalLMOutput:
        hidden_states, presents = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # 有 labels 时保留完整序列，保证 next-token loss 能对齐。
        if labels is None and logits_to_keep > 0:
            hidden_for_logits = hidden_states[:, -logits_to_keep:]
        else:
            hidden_for_logits = hidden_states
        logits = self.lm_head(hidden_for_logits)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(
            logits=logits,
            loss=loss,
            # 当前学习版没有 MoE 路由损失，先提供与 MiniMind 训练器一致的字段。
            aux_loss=logits.new_zeros(()),
            past_key_values=presents,
            hidden_states=hidden_states,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """用 KV Cache 逐 token 生成；先支持最常用的采样参数。"""
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        if not 0 < top_p <= 1:
            raise ValueError("top_p 必须在 (0, 1] 范围内")

        generated = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(generated)
        else:
            attention_mask = attention_mask.to(device=generated.device)
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id
        past_key_values = None
        finished = torch.zeros(generated.size(0), dtype=torch.bool, device=generated.device)

        for _ in range(max_new_tokens):
            model_input = generated if past_key_values is None else generated[:, -1:]
            outputs = self(
                model_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            logits = outputs.logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                for row in range(generated.size(0)):
                    seen_tokens = torch.unique(generated[row])
                    seen_scores = logits[row, seen_tokens]
                    logits[row, seen_tokens] = torch.where(
                        seen_scores > 0,
                        seen_scores / repetition_penalty,
                        seen_scores * repetition_penalty,
                    )

            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cumulative_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_indices, sorted_logits)

            if do_sample:
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )
                finished = finished | next_token.squeeze(-1).eq(eos_token_id)

            generated = torch.cat((generated, next_token), dim=-1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(next_token)), dim=-1
            )
            past_key_values = outputs.past_key_values if use_cache else None

            if finished.all():
                break

        return generated


CausalLM = MiniMindForCausalLM
