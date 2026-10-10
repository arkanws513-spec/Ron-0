"""Train a byte-level BPE tokenizer from an explicitly supplied local text corpus."""
from __future__ import annotations
import argparse
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="UTF-8 text corpus; use legally usable, reviewed data")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ron-tokenizer"))
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    if not args.corpus.is_file():
        parser.error(f"corpus file not found: {args.corpus}")
    if args.vocab_size < 256 or args.min_frequency < 1:
        parser.error("vocab-size must be >= 256 and min-frequency must be positive")
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as exc:
        raise SystemExit("Install tokenizer tooling with: pip install -e '.[distill]'") from exc
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train([str(args.corpus)], trainer)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output / "tokenizer.json"))
    (args.output / "tokenizer_manifest.json").write_text(
        '{\n  "format": "byte-level-bpe",\n  "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>"],\n  "corpus": "' + args.corpus.name.replace('\\', '\\\\').replace('"', '\\"') + '\",\n  "review_required": true\n}\n', encoding="utf-8")
    print(f"tokenizer={args.output / 'tokenizer.json'} vocab_size={tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    main()
