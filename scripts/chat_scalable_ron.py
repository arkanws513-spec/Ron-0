"""Chat with a locally trained scalable Ron checkpoint; no hosted inference is used."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from ron.scalable_model import ScalableRonConfig, ScalableRonLM

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/ron-scalable/checkpoint.pt"))
    p.add_argument("--config", type=Path, default=Path("artifacts/ron-scalable/config.json"))
    p.add_argument("--tokenizer", type=Path, default=Path("artifacts/ron-scalable/tokenizer.json"))
    p.add_argument("--prompt", help="Single prompt; omit for interactive terminal chat")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()
    if args.max_new_tokens < 0 or args.temperature < 0:
        p.error("generation limits must be non-negative")
    for path in (args.checkpoint, args.config, args.tokenizer):
        if not path.is_file():
            p.error(f"required file not found: {path}")
    from tokenizers import Tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ScalableRonConfig.load(args.config)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab = tokenizer.get_vocab()
    if tokenizer.get_vocab_size() != config.vocab_size:
        p.error("tokenizer vocabulary does not match checkpoint config")
    model = ScalableRonLM(config).to(device)
    try:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    bos, user, assistant, eos = (vocab.get("<bos>", 2), vocab.get("<user>", 4),
                                 vocab.get("<assistant>", 5), vocab.get("<eos>", 3))
    def answer(prompt: str) -> str:
        ids = [bos, user, *tokenizer.encode(prompt).ids, assistant]
        if len(ids) > config.max_sequence_length:
            ids = ids[-config.max_sequence_length:]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        output = model.generate(input_ids, args.max_new_tokens, eos_token_id=eos,
                                temperature=args.temperature)[0, len(ids):].tolist()
        return tokenizer.decode(output, skip_special_tokens=True).strip()
    if args.prompt is not None:
        print(answer(args.prompt))
        return
    print("Ron scalable local pilot. Type /exit to quit.")
    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"/exit", "/quit"}:
            break
        if not prompt:
            continue
        print("Ron> " + answer(prompt))

if __name__ == "__main__":
    main()
