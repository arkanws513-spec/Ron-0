"""Optional local Hugging Face model adapter for Ron's central orchestration core.

Weights are downloaded to the host's Hugging Face cache on first startup and then
run locally. No hosted inference API or API key is used.
"""
from __future__ import annotations

import os
from typing import Any

from .contracts import ModelRequest, ModelResponse


class HuggingFaceLocalProvider:
    """A locally executed instruction model behind Ron's ModelProvider interface."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        *,
        max_new_tokens: int = 256,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face backend needs the optional dependencies: "
                "pip install -e '.[huggingface]'"
            ) from exc

        self.model_id = model_id
        self.max_new_tokens = max(16, min(int(max_new_tokens), 2048))
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Keep inference portable across CPU-only hosts and avoid requiring accelerate.
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.device = next(self.model.parameters()).device

    @classmethod
    def from_environment(cls) -> "HuggingFaceLocalProvider":
        return cls(
            model_id=os.getenv("RON_HF_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct"),
            max_new_tokens=int(os.getenv("RON_HF_MAX_NEW_TOKENS", "256")),
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        messages = [
            {
                "role": message.role if message.role in {"system", "user", "assistant"} else "user",
                "content": message.content,
            }
            for message in request.messages
        ]
        tokenizer = self.tokenizer
        if getattr(tokenizer, "chat_template", None):
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_ids = encoded.to(self.device)
            attention_mask = self._torch.ones_like(input_ids)
        else:
            prompt = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

        generation_args: dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if attention_mask is not None:
            generation_args["attention_mask"] = attention_mask
        with self._torch.inference_mode():
            output = self.model.generate(**generation_args)
        new_tokens = output[0, input_ids.shape[-1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not answer:
            answer = "لم أتمكن من صياغة إجابة واضحة. هل يمكنك إعادة السؤال بطريقة أخرى؟"
        return ModelResponse(
            content=answer,
            model=self.model_id,
            metadata={
                "provider": "huggingface-local",
                "model_id": self.model_id,
                "pretrained_model_used": True,
                "external_model_used": False,
                "inference_mode": "local",
                "parameter_count": self.parameter_count,
                "weights_location": "host Hugging Face cache",
            },
        )
