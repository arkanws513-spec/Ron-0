"""Prepare a deduplicated plain-text corpus for Ron-10M training.

Place only public-domain or appropriately licensed .txt files under training/corpus/.
The script does not download data or grant rights to the source material.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "training" / "corpus"
OUTPUT = ROOT / "training" / "corpus.txt"
REPORT = ROOT / "training" / "corpus_manifest.json"
MIN_PARAGRAPH_CHARS = 80
MIN_TOTAL_CHARS = 100_000


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Missing {SOURCE_DIR}. Add public-domain or appropriately licensed .txt files first."
        )
    files = sorted(SOURCE_DIR.rglob("*.txt"))
    if not files:
        raise ValueError(f"No .txt files found under {SOURCE_DIR}")

    seen: set[str] = set()
    paragraphs: list[str] = []
    manifest_files = []
    for path in files:
        raw = path.read_text(encoding="utf-8-sig", errors="strict")
        normalized = unicodedata.normalize("NFC", raw).replace("\r\n", "\n").replace("\r", "\n")
        accepted = 0
        for part in normalized.split("\n\n"):
            paragraph = " ".join(line.strip() for line in part.splitlines() if line.strip()).strip()
            if len(paragraph) < MIN_PARAGRAPH_CHARS or paragraph in seen:
                continue
            seen.add(paragraph)
            paragraphs.append(paragraph)
            accepted += 1
        manifest_files.append({
            "path": str(path.relative_to(ROOT)),
            "source_characters": len(raw),
            "accepted_paragraphs": accepted,
            "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        })

    if len(paragraphs) < 10:
        raise ValueError(f"Only {len(paragraphs)} usable paragraphs; at least 10 are required.")
    corpus = "\n\n".join(paragraphs) + "\n"
    if len(corpus) < MIN_TOTAL_CHARS:
        raise ValueError(
            f"Prepared corpus has {len(corpus):,} characters; need at least {MIN_TOTAL_CHARS:,}. "
            "Add more reviewed, legally usable text and rerun."
        )

    OUTPUT.write_text(corpus, encoding="utf-8")
    report = {
        "format": "UTF-8 text, blank-line-separated paragraphs",
        "output": str(OUTPUT.relative_to(ROOT)),
        "characters": len(corpus),
        "unique_paragraphs": len(paragraphs),
        "source_files": manifest_files,
        "sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
        "notice": "This manifest records file hashes, not proof of factual quality or licensing. Keep source/license records.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
