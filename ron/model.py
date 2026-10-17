"""Ron-native decoder-only Transformer with optional autoregressive KV caching."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model_config import RonModelConfig

LayerKV: TypeAlias = tuple[Tensor, Tensor]
KVCache: TypeAlias = tuple[LayerKV, ...]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: RonModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.max_sequence_length = config.max_sequence_length

    def forward(
        self,
        x: Tensor,
        past_key_value: LayerKV | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[Tensor, LayerKV | None]:
        batch, seq, hidden = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        past_length = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_length = past_k.shape[-2]
            if past_k.shape[:2] != k.shape[:2] or past_v.shape != past_k.shape:
                raise ValueError("past_key_value shape is incompatible with current attention input")
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        if past_length + seq > self.max_sequence_length:
            raise ValueError("cached sequence exceeds model context length")

        if past_length == 0:
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # SDPA's is_causal=True uses upper-left alignment for non-square masks.
            # Build an explicit lower-right causal mask for cached decoding instead.
            query_positions = past_length + torch.arange(seq, device=x.device)
            key_positions = torch.arange(past_length + seq, device=x.device)
            allowed = key_positions[None, :] <= query_positions[:, None]
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)

        output = self.out(attended.transpose(1, 2).contiguous().view(batch, seq, hidden))
        present = (k, v) if use_cache else None
        return output, present


class TransformerBlock(nn.Module):
    def __init__(self, config: RonModelConfig) -> None:
        super().__init__()
        ff = 4 * config.hidden_size
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, ff), nn.GELU(),
            nn.Linear(ff, config.hidden_size),
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        past_key_value: LayerKV | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[Tensor, LayerKV | None]:
        attended, present = self.attention(self.norm1(x), past_key_value, use_cache=use_cache)
        x = x + self.residual_dropout(attended)
        x = x + self.residual_dropout(self.mlp(self.norm2(x)))
        return x, present


@dataclass(frozen=True)
class ModelOutput:
    logits: Tensor
    loss: Tensor | None = None
    past_key_values: KVCache | None = None


class RonCausalLM(nn.Module):
    def __init__(self, config: RonModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, seq = input_ids.shape
        if seq < 1:
            raise ValueError("sequence length must be positive")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must contain one key/value pair per Transformer block")
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        if past_length + seq > self.config.max_sequence_length:
            raise ValueError("sequence length is outside model limits")

        positions = torch.arange(past_length, past_length + seq, device=input_ids.device)
        x = self.embedding_dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        )
        presents: list[LayerKV] = []
        for index, block in enumerate(self.blocks):
            layer_past = None if past_key_values is None else past_key_values[index]
            x, present = block(x, layer_past, use_cache=use_cache)
            if use_cache:
                assert present is not None
                presents.append(present)

        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return ModelOutput(logits=logits, loss=loss, past_key_values=tuple(presents) if use_cache else None)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 32,
        *,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        use_cache: bool = True,
    ) -> Tensor:
        """Generate tokens, using a KV cache by default; set use_cache=False for reference decoding."""
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have shape [batch, non-empty sequence]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and not 1 <= top_k <= self.config.vocab_size:
            raise ValueError("top_k must be between 1 and vocab_size")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if eos_token_id is not None and not 0 <= eos_token_id < self.config.vocab_size:
            raise ValueError("eos_token_id must be between 0 and vocab_size - 1")

        result = input_ids
        finished = torch.zeros(result.shape[0], dtype=torch.bool, device=result.device)
        cache: KVCache | None = None
        next_input = result[:, -self.config.max_sequence_length:]
        for _ in range(max_new_tokens):
            if use_cache and cache is not None and cache[0][0].shape[-2] < self.config.max_sequence_length:
                model_input = next_input
                output = self(model_input, past_key_values=cache, use_cache=True)
            else:
                model_input = result[:, -self.config.max_sequence_length:]
                output = self(model_input, use_cache=use_cache)
            logits = output.logits[:, -1, :].float()
            cache = output.past_key_values if use_cache else None
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    threshold = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                if top_p < 1:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probs = sorted_probabilities.cumsum(dim=-1)
                    remove = cumulative_probs - sorted_probabilities >= top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    filtered = torch.full_like(logits, float("-inf"))
                    filtered.scatter_(1, sorted_indices, sorted_logits)
                    logits = filtered
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            if eos_token_id is not None:
                eos_fill = torch.full_like(next_token, eos_token_id)
                next_token = torch.where(finished[:, None], eos_fill, next_token)
                finished = finished | next_token.squeeze(1).eq(eos_token_id)
            result = torch.cat((result, next_token), dim=1)
            next_input = next_token
            if eos_token_id is not None and bool(finished.all()):
                break
        return result
