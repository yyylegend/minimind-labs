"""MiniMind 学习版配置。"""

from dataclasses import dataclass
import math


@dataclass
class MiniMindConfig:
    """把模型结构参数集中放在一个小配置对象里。"""

    vocab_size: int = 6400
    hidden_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int | None = 4
    intermediate_size: int | None = None
    max_position_embeddings: int = 32768
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    flash_attn: bool = True
    tie_word_embeddings: bool = True
    use_moe: bool = False
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads 必须能被 num_key_value_heads 整除")
        if self.intermediate_size is None:
            # 对齐 MiniMind 当前默认的中间层宽度计算方式。
            self.intermediate_size = math.ceil(self.hidden_size * math.pi / 64) * 64

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
