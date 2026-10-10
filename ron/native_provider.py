"""Native Ron-0 checkpoint provider; no external model or inference service."""
from __future__ import annotations

from pathlib import Path
import torch

from .contracts import ModelRequest, ModelResponse
from .model import RonCausalLM
from .model_config import RonModelConfig


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
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.selected_step = int(payload.get("selected_step", 0))
        self.training_steps = int(payload.get("training_steps_this_run", 0))
        self.checkpoint_source = str(payload.get("checkpoint_source", "unknown"))

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

        # The native model was trained from scratch on Ron's Arabic dialogue corpus.
        prompt = f"المستخدم: {user_text}\nرون:"
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

        answer = "".join(self.id_to_char.get(index, "") for index in generated)
        answer = answer.split("\n", 1)[0].strip()
        if not answer:
            answer = "لم أتمكن من توليد إجابة واضحة من النموذج الأصلي بعد."
        return ModelResponse(
            content=answer,
            model="ron-native-checkpoint",
            metadata={
                "runtime": "ron-0",
                "provider": "native-checkpoint",
                "native_weights_loaded": True,
                "selected_training_step": self.selected_step,
                "training_steps_this_run": self.training_steps,
                "checkpoint_source": self.checkpoint_source,
                "external_model_used": False,
            },
        )
