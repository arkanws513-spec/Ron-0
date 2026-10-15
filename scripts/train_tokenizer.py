"""Build a local byte-level BPE tokenizer and report its compression metrics.

Run after preparing licensed training text:
    python scripts/train_tokenizer.py
or override paths/vocabulary size with RON_CORPUS_PATH and RON_TOKENIZER_VOCAB_SIZE.
This creates a tokenizer artifact only; it does not train or replace Ron's model weights.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ron.tokenizer import RonTokenizer

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = Path(os.environ.get("RON_CORPUS_PATH", ROOT / "training" / "corpus.txt"))
OUTPUT_PATH = ROOT / "artifacts" / "tokenizer" / "ron_bpe_tokenizer.json"
VOCAB_SIZE = int(os.environ.get("RON_TOKENIZER_VOCAB_SIZE", "4096"))
MIN_FREQUENCY = int(os.environ.get("RON_TOKENIZER_MIN_FREQUENCY", "2"))


def main() -> None:
    if not CORPUS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {CORPUS_PATH}. Prepare the licensed corpus before training a tokenizer."
        )
    corpus = CORPUS_PATH.read_text(encoding="utf-8")
    if len(corpus) < 10_000:
        raise ValueError("Tokenizer training corpus is too small; provide at least 10,000 characters.")
    # Keep the complete corpus as one iterator item to preserve all whitespace and boundaries.
    tokenizer = RonTokenizer().fit([corpus], vocab_size=VOCAB_SIZE, min_frequency=MIN_FREQUENCY)
    ids = tokenizer.encode(corpus, add_special_tokens=False)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(tokenizer.to_json() + "\n", encoding="utf-8")
    metrics = {
        "status": "tokenizer_candidate_created",
        "algorithm": "byte_level_bpe",
        "corpus_path": str(CORPUS_PATH.relative_to(ROOT)) if CORPUS_PATH.is_relative_to(ROOT) else str(CORPUS_PATH),
        "corpus_characters": len(corpus),
        "corpus_utf8_bytes": len(corpus.encode("utf-8")),
        "token_count": len(ids),
        "characters_per_token": round(len(corpus) / max(1, len(ids)), 4),
        "tokenizer_vocabulary_size": tokenizer.backend.get_vocab_size(),
        "requested_vocabulary_size": VOCAB_SIZE,
        "tokenizer_file": str(OUTPUT_PATH.relative_to(ROOT)),
        "weights_changed": False,
        "checkpoint_promoted": False,
        "notes": [
            "Tokenizer artifact only; model training and inference are not switched automatically.",
            "Compare token compression and blind conversation quality before migrating model checkpoints.",
        ],
    }
    metrics_path = OUTPUT_PATH.with_name("metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
