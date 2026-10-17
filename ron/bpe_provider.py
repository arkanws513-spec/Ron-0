"""Inference adapter for native byte-level BPE checkpoints.

Legacy character checkpoints remain supported by NativeCheckpointProvider. A BPE
checkpoint must explicitly declare tokenizer_type='byte_bpe' and embed tokenizer_json.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import torch

from .contracts import ModelRequest, ModelResponse
from .generation_quality import guard_generated_text
from .model import RonCausalLM
from .model_config import RonModelConfig
from .native_provider import NativeCheckpointProvider
from .tokenizer import RonTokenizer


class BpeCheckpointProvider:
    """Run Ron's own Transformer with the exact BPE tokenizer saved in its checkpoint."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Ron BPE checkpoint not found: {self.checkpoint_path}")
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        if payload.get("tokenizer_type") != "byte_bpe" or not payload.get("tokenizer_json"):
            raise ValueError("Checkpoint is not a self-contained byte-BPE checkpoint")
        self.tokenizer = RonTokenizer.from_json(payload["tokenizer_json"])
        self.config = RonModelConfig(**payload["config"])
        if self.config.vocab_size != self.tokenizer.backend.get_vocab_size():
            raise ValueError("Checkpoint model vocabulary does not match its embedded tokenizer")
        self.model = RonCausalLM(self.config)
        use_best = "best_state_dict" in payload
        self.model.load_state_dict(payload.get("best_state_dict", payload["state_dict"]))
        self.model.eval()
        self.selected_step = int(payload.get("best_selected_step", payload.get("selected_step", 0)))
        self.final_training_step = int(payload.get("selected_step", self.selected_step))
        self.inference_checkpoint = "best_validation" if use_best else "final_training_state"
        self.training_steps = int(payload.get("training_steps_this_run", 0))
        self.training_steps_total = int(payload.get("training_steps_total", self.selected_step))
        self.optimizer_state_persisted = "optimizer_state_dict" in payload
        self.checkpoint_source = str(payload.get("checkpoint_source", "native_bpe_checkpoint"))

    # Reuse Ron's vetted deterministic and curated-retrieval layer without inheriting
    # the legacy character-checkpoint loader.
    _normalize_question = staticmethod(NativeCheckpointProvider._normalize_question)
    _retrieval_tokens = staticmethod(NativeCheckpointProvider._retrieval_tokens)
    _char_ngrams = staticmethod(NativeCheckpointProvider._char_ngrams)
    _retrieve_curated_answer = NativeCheckpointProvider._retrieve_curated_answer
    _known_core_answer = NativeCheckpointProvider._known_core_answer

    @torch.inference_mode()
    def generate(self, request: ModelRequest) -> ModelResponse:
        user_text = next((m.content for m in reversed(request.messages) if m.role == "user"), "").strip()
        if not user_text:
            return ModelResponse(content="أنا رون. اكتب لي ما تريد.", model="ron-native-bpe",
                                 metadata={"runtime": "ron-0", "provider": "native-bpe"})
        known = self._known_core_answer(request, user_text)
        if known is not None:
            return self._response(known, deterministic=True)
        retrieved = self._retrieve_curated_answer(user_text)
        if retrieved is not None:
            answer, score = retrieved
            return self._response(answer, curated_retrieval=True, retrieval_similarity=round(score, 4))

        # Encode the same role-labelled context used during training; BPE IDs are
        # never inferred from characters or from a separately rebuilt vocabulary.
        has_arabic = any("\u0600" <= char <= "\u06ff" for char in user_text)
        user_label, assistant_label = ("المستخدم", "رون") if has_arabic else ("User", "Ron")
        lines = []
        for message in request.messages:
            if message.role not in {"user", "assistant"} or not message.content.strip():
                continue
            label = user_label if message.role == "user" else assistant_label
            lines.append(f"{label}: {message.content.strip()}")
        prompt = "\n".join(lines[-12:]) + f"\n{assistant_label}:"
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)[-self.config.max_sequence_length:]
        if not ids:
            ids = self.tokenizer.encode(f"{user_label}: {user_text}\n{assistant_label}:", add_special_tokens=False)
            ids = ids[-self.config.max_sequence_length:]
        result = torch.tensor([ids], dtype=torch.long)
        generated: list[int] = []
        seed_material = (self.checkpoint_path.name + ":" + prompt).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big") % (2**63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        eos_id = self.tokenizer.backend.token_to_id("<eos>")

        # Prefill once, then feed only the newly generated token while the cache
        # fits the context window. At rollover, rebuild the cache from the exact
        # sliding window so position embeddings retain the same semantics.
        context = result[:, -self.config.max_sequence_length:]
        output = self.model(context, use_cache=True)
        cache = output.past_key_values
        logits = output.logits[:, -1, :].float().squeeze(0)

        for _ in range(min(160, self.config.max_sequence_length)):
            sampling_logits = logits / 0.8
            probabilities = torch.softmax(sampling_logits, dim=-1)
            sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)
            cumulative = sorted_probs.cumsum(0)
            sorted_probs[cumulative - sorted_probs >= 0.90] = 0
            filtered = torch.zeros_like(probabilities)
            filtered.scatter_(0, sorted_ids, sorted_probs)
            total = filtered.sum()
            next_id = int(torch.argmax(sampling_logits).item()) if not torch.isfinite(total) or total.item() <= 0 else int(torch.multinomial(filtered / total, 1, generator=generator).item())
            if eos_id is not None and next_id == eos_id:
                break
            generated.append(next_id)
            result = torch.cat((result, torch.tensor([[next_id]], dtype=torch.long)), dim=1)
            decoded = self.tokenizer.decode(generated)
            if "\\n" in decoded:
                break

            assert cache is not None
            cached_length = cache[0][0].shape[-2]
            if cached_length < self.config.max_sequence_length:
                output = self.model(
                    result[:, -1:],
                    past_key_values=cache,
                    use_cache=True,
                )
            else:
                output = self.model(
                    result[:, -self.config.max_sequence_length:],
                    use_cache=True,
                )
            cache = output.past_key_values
            logits = output.logits[:, -1, :].float().squeeze(0)
        raw = self.tokenizer.decode(generated).split("\n", 1)[0].strip()
        answer, rejected = guard_generated_text(
            raw,
            "لم أتمكن من صياغة إجابة موثوقة من نموذجي المحلي الحالي. أحتاج إلى تحسين التدريب قبل الإجابة عن هذا السؤال.",
        )
        return self._response(answer, native_generated_answer_rejected=rejected,
                              generation_quality_fallback=rejected)

    def _response(self, content: str, **extra) -> ModelResponse:
        return ModelResponse(
            content=content,
            model="ron-native-bpe",
            metadata={
                "runtime": "ron-0",
                "provider": "native-bpe",
                "native_weights_loaded": True,
                "tokenizer_type": "byte_bpe",
                "selected_training_step": self.selected_step,
                "inference_checkpoint": self.inference_checkpoint,
                "final_training_step": self.final_training_step,
                "training_steps_this_run": self.training_steps,
                "training_steps_total": self.training_steps_total,
                "optimizer_state_persisted": self.optimizer_state_persisted,
                "checkpoint_source": self.checkpoint_source,
                "external_model_used": False,
                **extra,
            },
        )
