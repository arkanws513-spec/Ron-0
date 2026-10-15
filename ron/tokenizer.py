"""Byte-level BPE tokenizer for Ron-0's native language model.

Uses the open-source Hugging Face tokenizers runtime, not an external language model.
Byte-level pre-tokenization preserves whitespace and gives coverage for unseen Unicode,
including Arabic, diacritics, emoji, and mixed-language input.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


@dataclass
class Vocabulary:
    """Small compatibility container for callers that need a token/id mapping."""

    token_to_id: dict[str, int] = field(default_factory=dict)

    @property
    def id_to_token(self) -> dict[int, str]:
        return {index: token for token, index in self.token_to_id.items()}

    def add(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.token_to_id)
        return self.token_to_id[token]


class RonTokenizer:
    """Trainable byte-level BPE tokenizer with a stable JSON serialization format."""

    def __init__(self, backend=None):
        self._backend = backend

    @staticmethod
    def _runtime():
        try:
            from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        except ImportError as exc:
            raise RuntimeError(
                "RonTokenizer requires the 'tokenizers' package; install Ron-0 with .[model]."
            ) from exc
        return Tokenizer, decoders, models, pre_tokenizers

    @property
    def backend(self):
        if self._backend is None:
            Tokenizer, decoders, models, pre_tokenizers = self._runtime()
            self._backend = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
            self._backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self._backend.decoder = decoders.ByteLevel()
            self._backend.add_special_tokens(list(SPECIAL_TOKENS))
        return self._backend

    @property
    def vocabulary(self) -> Vocabulary:
        return Vocabulary(dict(self.backend.get_vocab()))

    def fit(self, texts: Iterable[str], vocab_size: int = 4096, min_frequency: int = 2):
        """Train BPE merges on text; all input text remains local to the caller."""
        if vocab_size < 300:
            raise ValueError("vocab_size must be at least 300 for byte-level coverage")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        Tokenizer, decoders, models, pre_tokenizers = self._runtime()
        backend = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        from tokenizers import trainers
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        backend.train_from_iterator((str(text) for text in texts), trainer=trainer)
        self._backend = backend
        return self

    def tokenize(self, text: str) -> list[str]:
        return self.backend.encode(str(text), add_special_tokens=False).tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        encoded = self.backend.encode(str(text), add_special_tokens=False).ids
        if not add_special_tokens:
            return encoded
        bos = self.backend.token_to_id("<bos>")
        eos = self.backend.token_to_id("<eos>")
        return ([bos] if bos is not None else []) + encoded + ([eos] if eos is not None else [])

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        values = [int(value) for value in ids]
        if skip_special_tokens:
            special_ids = {self.backend.token_to_id(token) for token in SPECIAL_TOKENS}
            values = [value for value in values if value not in special_ids]
        return self.backend.decode(values, skip_special_tokens=skip_special_tokens)

    def to_json(self) -> str:
        return self.backend.to_str()

    def to_dict(self) -> dict:
        return json.loads(self.to_json())

    @classmethod
    def from_json(cls, serialized: str) -> "RonTokenizer":
        Tokenizer, _, _, _ = cls._runtime()
        return cls(Tokenizer.from_str(serialized))

    @classmethod
    def from_dict(cls, data: dict) -> "RonTokenizer":
        return cls.from_json(json.dumps(data, ensure_ascii=False))
