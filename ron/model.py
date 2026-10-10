""""Ron-native decoder-only Transformer model."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .model_config import RonModelConfig

class CausalSelfAttention(nn.Module):
    def __init__(self, config: RonModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        mask = torch.triu(torch.ones(config.max_sequence_length, config.max_sequence_length, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq, hidden = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=~self.causal_mask[:seq, :seq])
        return self.out(attended.transpose(1, 2).contiguous().view(batch, seq, hidden))

class TransformerBlock(nn.Module):
    def __init__(self, config: RonModelConfig) -> None:
        super().__init__()
        ff = 4 * config.hidden_size
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(nn.Linear(config.hidden_size, ff), nn.GELU(), nn.Linear(ff, config.hidden_size))
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.residual_dropout(self.attention(self.norm1(x)))
        return x + self.residual_dropout(self.mlp(self.norm2(x)))

@dataclass(frozen=True)
class ModelOutput:
    logits: Tensor
    loss: Tensor | None = None

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

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, seq = input_ids.shape
        if seq < 1 or seq > self.config.max_sequence_length:
            raise ValueError("sequence length is outside model limits")
        positions = torch.arange(seq, device=input_ids.device)
        x = self.embedding_dropout(self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :])
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return ModelOutput(logits=logits, loss=loss)

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
    ) -> Tensor:
        """Generate tokens greedily (temperature=0) or by filtered sampling.

        top_k and nucleus (top_p) filtering only affect sampled generation.
        Set eos_token_id to stop once every sequence in the batch has ended.
        """
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
        for _ in range(max_new_tokens):
            context = result[:, -self.config.max_sequence_length:]
            logits = self(context).logits[:, -1, :].float()
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    threshold = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                if top_p < 1:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
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
            if eos_token_id is not None and bool(finished.all()):
                break
        return result
"