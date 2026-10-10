"""Fetch reviewed English OASST1 conversation trees for Ron's dialogue-training stage.

Dataset: OpenAssistant/oasst1, Apache-2.0 according to its Hugging Face dataset card.
Source: https://huggingface.co/datasets/OpenAssistant/oasst1
This script filters non-English, deleted, explicitly rejected, and high-risk label-scored
messages. The resulting corpus still needs review; automated filters are not perfect.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "training" / "corpus" / "oasst1"
OUTPUT = DEST / "english_dialogues.txt"
MANIFEST = ROOT / "training" / "oasst1_manifest.json"
URL = (
    "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/"
    "2023-04-12_oasst_ready.trees.jsonl.gz?download=true"
)
MAX_TURNS = 8
MAX_FLAG_SCORE = 0.25
FILTER_LABELS = (
    "spam",
    "not_appropriate",
    "sexual_content",
    "toxicity",
    "severe_toxicity",
    "threat",
    "identity_attack",
)


def is_acceptable_message(node: dict) -> bool:
    if node.get("deleted") is True or node.get("review_result") is False:
        return False
    if node.get("lang") not in (None, "en"):
        return False
    labels = node.get("labels") or {}
    for name in FILTER_LABELS:
        label = labels.get(name)
        if isinstance(label, dict) and float(label.get("value", 0.0)) > MAX_FLAG_SCORE:
            return False
    role = node.get("role")
    if role not in {"prompter", "assistant"}:
        return False
    text = node.get("text")
    return isinstance(text, str) and bool(" ".join(text.split()))


def clean_message(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def iter_conversation_examples(tree: dict):
    """Yield English User/Ron prefixes ending in a reviewed assistant response."""
    def walk(node: dict, path: list[tuple[str, str]]):
        if not is_acceptable_message(node):
            return
        role = node["role"]
        text = clean_message(node["text"])
        label = "User" if role == "prompter" else "Ron"
        current = path + [(label, text)]
        if role == "assistant" and any(item[0] == "User" for item in current):
            rendered = "\n".join(f"{speaker}: {message}" for speaker, message in current[-MAX_TURNS:])
            if len(rendered) >= 80:
                yield rendered
        for child in node.get("replies") or []:
            if isinstance(child, dict):
                yield from walk(child, current)

    # OASST1 tree rows wrap the root prompt under a top-level "prompt" key.
    # Walking the wrapper itself sees no role and silently yields zero examples.
    root = tree.get("prompt", tree.get("tree", tree))
    if isinstance(root, dict):
        yield from walk(root, [])


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    request = Request(URL, headers={"User-Agent": "Ron-0 research corpus builder/1.0"})
    with urlopen(request, timeout=120) as response:
        compressed = response.read()
    digest = hashlib.sha256(compressed).hexdigest()
    examples: set[str] = set()
    tree_count = 0
    import io
    with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            tree = json.loads(line.decode("utf-8"))
            tree_count += 1
            examples.update(iter_conversation_examples(tree))

    if len(examples) < 100:
        raise RuntimeError(
            f"Only {len(examples)} acceptable English conversation examples were produced; refusing to write a tiny corpus."
        )
    rendered = "\n\n".join(sorted(examples)) + "\n"
    OUTPUT.write_text(rendered, encoding="utf-8")
    manifest = {
        "dataset": "OpenAssistant Conversations Dataset (OASST1), ready-for-export trees",
        "source_url": URL,
        "license": "Apache-2.0 (dataset card declaration; retain source/license records)",
        "compressed_source_sha256": digest,
        "tree_count": tree_count,
        "unique_english_conversation_prefixes": len(examples),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_characters": len(rendered),
        "filters": {
            "language": "English only",
            "deleted_or_explicitly_rejected": "excluded",
            "label_score_threshold": MAX_FLAG_SCORE,
            "label_categories": list(FILTER_LABELS),
            "max_turns": MAX_TURNS,
        },
        "notice": "Automated filters can miss unsafe or low-quality content. Review samples before production training.",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "trees": tree_count,
        "examples": len(examples),
        "characters": len(rendered),
        "manifest": str(MANIFEST.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
