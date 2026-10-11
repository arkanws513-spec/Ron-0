"""Scalable subword decoder for new Ron checkpoints; legacy RonCausalLM is unchanged."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import torch
from torch import Tensor, nn
import torch.nn.functional as F

@dataclass(frozen=True)
class ScalableRonConfig:
    vocab_size: int = 16000
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    max_sequence_length: int = 512
    intermediate_size: int = 0
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.hidden_size, self.num_layers, self.num_heads,
               self.num_kv_heads, self.max_sequence_length) < 1:
            raise ValueError("model dimensions must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if (self.hidden_size // self.num_heads) % 2:
            raise ValueError("head dimension must be even for RoPE")
        if self.intermediate_size < 0 or self.dropout < 0 or self.dropout >= 1:
            raise ValueError("invalid intermediate_size or dropout")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def ff_size(self) -> int:
        return self.intermediate_size or int(8 * self.hidden_size / 3 + 63) // 64 * 64

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ScalableRonConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normed = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return normed * self.weight

def apply_rope(x: Tensor, positions: Tensor, theta: float) -> Tensor:
    # x: [batch, heads, sequence, head_dim]
    dim = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    angles = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    cos, sin = angles.cos().to(x.dtype)[None, None], angles.sin().to(x.dtype)[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ScalableRonConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, seq, _ = x.shape
        c = self.config
        q = self.q_proj(x).view(b, seq, c.num_heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, seq, c.num_kv_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, seq, c.num_kv_heads, c.head_dim).transpose(1, 2)
        positions = torch.arange(seq, device=x.device)
        q, k = apply_rope(q, positions, c.rope_theta), apply_rope(k, positions, c.rope_theta)
        if c.num_kv_heads != c.num_heads:
            repeats = c.num_heads // c.num_kv_heads
            k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=c.dropout if self.training else 0.0)
        return self.o_proj(y.transpose(1, 2).contiguous().view(b, seq, c.hidden_size))

class SwiGLU(nn.Module):
    def __init__(self, config: ScalableRonConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.ff_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.ff_size, bias=False)
        self.down = nn.Linear(config.ff_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))

class ScalableBlock(nn.Module):
    def __init__(self, config: ScalableRonConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))

class ScalableRonLM(nn.Module):
    """Decoder-only LM for subword IDs; does not load legacy character checkpoints."""
    def __init__(self, config: ScalableRonConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(ScalableBlock(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self) -> None:
        """Restore shared input/output embeddings after checkpoint assignment loading."""
        if self.config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict[str, Tensor | None]:
        if input_ids.ndim != 2 or input_ids.shape[1] < 1 or input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("input_ids must be [batch, sequence] within configured sequence limits")
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids shape")
            if not bool(labels.ne(-100).any()):
                raise ValueError("labels must contain at least one non-masked token")
            loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                                   labels[:, 1:].contiguous().view(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}

    @torch.inference_mode()
    def generate(self, input_ids: Tensor, max_new_tokens: int = 128, *,
                 eos_token_id: int | None = None, temperature: float = 0.0,
                 valid_vocab_size: int | None = None,
                 suppress_token_ids: list[int] | tuple[int, ...] | None = None,
                 min_new_tokens: int = 0) -> Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must be a non-empty [batch, sequence] tensor")
        if max_new_tokens < 0 or temperature < 0:
            raise ValueError("max_new_tokens and temperature must be non-negative")
        if min_new_tokens < 0 or min_new_tokens > max_new_tokens:
            raise ValueError("min_new_tokens must be between zero and max_new_tokens")
        if valid_vocab_size is not None and not 1 <= valid_vocab_size <= self.config.vocab_size:
            raise ValueError("valid_vocab_size must be within the model vocabulary")
        if eos_token_id is not None and not 0 <= eos_token_id < self.config.vocab_size:
            raise ValueError("eos_token_id is outside the vocabulary")
        suppressed = tuple(suppress_token_ids or ())
        if any(token_id < 0 or token_id >= self.config.vocab_size for token_id in suppressed):
            raise ValueError("a suppressed token ID is outside the vocabulary")
        if eos_token_id is not None and eos_token_id in suppressed:
            raise ValueError("EOS must not be included in suppress_token_ids")
        result = input_ids
        finished = torch.zeros(result.shape[0], dtype=torch.bool, device=result.device)
        for step in range(max_new_tokens):
            context = result[:, -self.config.max_sequence_length:]
            logits = self(context)["logits"][:, -1, :].float()
            # Ron's fixed-size embedding table may be larger than its learned tokenizer.
            # Never sample untrained padding rows: they decode to empty text and can
            # dominate logits despite never appearing as targets during training.
            if valid_vocab_size is not None and valid_vocab_size < self.config.vocab_size:
                logits[:, valid_vocab_size:] = -torch.inf
            if suppressed:
                # Apply the suppression mask in one indexed operation; a Python
                # loop over hundreds of tokenizer pieces is costly on a 1-CPU host.
                logits[:, list(suppressed)] = -torch.inf
            # A weak/early checkpoint may predict EOS immediately. Require a chance
            # to emit content before accepting the end-of-turn token.
            if eos_token_id is not None and step < min_new_tokens:
                logits[:, eos_token_id] = -torch.inf
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)
            if eos_token_id is not None:
                next_token = torch.where(finished[:, None], torch.full_like(next_token, eos_token_id), next_token)
                finished = finished | next_token.squeeze(1).eq(eos_token_id)
            result = torch.cat((result, next_token), dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break
        return result

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
