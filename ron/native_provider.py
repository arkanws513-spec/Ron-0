"""Native Ron-0 checkpoint provider; no external model or inference service."""
from __future__ import annotations

from pathlib import Path
import torch

from .contracts import ModelRequest, ModelResponse
from .model import RonCausalLM
from .model_config import RonModelConfig
from .generation_quality import guard_generated_text


class NativeCheckpointProvider:
    """Load Ron's own trained checkpoint and generate from its character vocabulary."""

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else (
            Path(__file__).resolve().parent / "checkpoints" / "ron_native_baseline.pt"
        )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Ron native checkpoint not found: {self.checkpoint_path}")

        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.config = RonModelConfig(**payload["config"])
        self.vocab: dict[str, int] = payload["vocab"]
        self.id_to_char = {int(index): char for char, index in self.vocab.items()}
        self.model = RonCausalLM(self.config)
        # Use the best held-out-validation snapshot for inference when available;
        # keep state_dict as the final state for the next training continuation.
        use_best = "best_state_dict" in payload
        self.model.load_state_dict(payload.get("best_state_dict", payload["state_dict"]))
        self.model.eval()
        self.selected_step = int(payload.get("best_selected_step", payload.get("selected_step", 0)))
        self.final_training_step = int(payload.get("selected_step", self.selected_step))
        self.inference_checkpoint = "best_validation" if use_best else "final_training_state"
        self.training_steps = int(payload.get("training_steps_this_run", 0))
        self.training_steps_total = int(payload.get("training_steps_total", self.selected_step))
        self.optimizer_state_persisted = "optimizer_state_dict" in payload
        self.checkpoint_source = str(payload.get("checkpoint_source", "unknown"))

    def _build_dialogue_prompt(self, request: ModelRequest, user_text: str) -> str:
        # Match the prompt language to the checkpoint's training language where possible.
        # The current Ron-10M starter corpus is English-first; Arabic labels are retained for Arabic prompts.
        has_arabic = any("\u0600" <= char <= "\u06ff" for char in user_text)
        user_label, assistant_label = (
            ("المستخدم", "رون") if has_arabic else ("User", "Ron")
        )
        dialogue = []
        for message in request.messages:
            if message.role not in {"user", "assistant"} or not message.content.strip():
                continue
            label = user_label if message.role == "user" else assistant_label
            dialogue.append(f"{label}: {message.content.strip()}")
        user_prefix = f"{user_label}: "
        latest_line = next((line for line in reversed(dialogue) if line.startswith(user_prefix)),
                           f"{user_prefix}{user_text}")
        suffix = f"\n{assistant_label}:"
        budget = max(0, self.config.max_sequence_length - len(latest_line) - len(suffix))
        prior_lines = []
        for line in reversed(dialogue):
            if line == latest_line or budget <= 1:
                continue
            if len(line) + 1 <= budget:
                prior_lines.insert(0, line)
                budget -= len(line) + 1
        prompt = "\n".join(prior_lines + [latest_line]) + suffix
        if len(prompt) > self.config.max_sequence_length:
            prompt = latest_line[-(self.config.max_sequence_length - len(suffix)):] + suffix
        return prompt

    @torch.inference_mode()
    def generate(self, request: ModelRequest) -> ModelResponse:
        user_text = next(
            (message.content for message in reversed(request.messages) if message.role == "user"),
            "",
        ).strip()
        if not user_text:
            return ModelResponse(
                content="أنا رون. اكتب لي ما تريد.",
                model="ron-native-checkpoint",
                metadata={"runtime": "ron-0", "provider": "native-checkpoint"},
            )

        prompt = self._build_dialogue_prompt(request, user_text)
        unknown = self.vocab.get(" ", 0)
        ids = [self.vocab.get(char, unknown) for char in prompt]
        ids = ids[-self.config.max_sequence_length:]
        result = torch.tensor([ids], dtype=torch.long)
        generated: list[int] = []
        newline_id = self.vocab.get("\n")

        for _ in range(min(160, self.config.max_sequence_length)):
            context = result[:, -self.config.max_sequence_length:]
            next_id = int(self.model(context).logits[:, -1, :].argmax(dim=-1).item())
            generated.append(next_id)
            result = torch.cat((result, torch.tensor([[next_id]], dtype=torch.long)), dim=1)
            if newline_id is not None and next_id == newline_id and len(generated) > 1:
                break

        raw_answer = "".join(self.id_to_char.get(index, "") for index in generated)
        raw_answer = raw_answer.split("\n", 1)[0].strip()
        answer, generation_rejected = guard_generated_text(
            raw_answer,
            "لم أتمكن من صياغة إجابة موثوقة من نموذجي المحلي الحالي. أحتاج إلى تحسين التدريب قبل الإجابة عن هذا السؤال.",
        )
        return ModelResponse(
            content=answer,
            model="ron-native-checkpoint",
            metadata={
                "runtime": "ron-0",
                "provider": "native-checkpoint",
                "native_weights_loaded": True,
                "selected_training_step": self.selected_step,
                "inference_checkpoint": self.inference_checkpoint,
                "final_training_step": self.final_training_step,
                "training_steps_this_run": self.training_steps,
                "training_steps_total": self.training_steps_total,
                "optimizer_state_persisted": self.optimizer_state_persisted,
                "checkpoint_source": self.checkpoint_source,
                "external_model_used": False,
                "native_generated_answer_rejected": generation_rejected,
                "generation_quality_fallback": generation_rejected,
            },
        )
