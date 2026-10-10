"""Download a small English public-domain starter corpus from Project Gutenberg.

Check the legal status and Project Gutenberg terms for your jurisdiction before use.
The source texts are selected for their authors' age and historical public-domain status;
this script does not constitute legal advice or guarantee local reuse rights.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "training" / "corpus" / "gutenberg"
MANIFEST = ROOT / "training" / "gutenberg_sources.json"

# Original English-language works by long-deceased authors; IDs are Project Gutenberg ebook IDs.
BOOKS = [
    (11, "Alice's Adventures in Wonderland", "Lewis Carroll"),
    (84, "Frankenstein", "Mary Shelley"),
    (98, "A Tale of Two Cities", "Charles Dickens"),
    (120, "Robinson Crusoe", "Daniel Defoe"),
    (1342, "Pride and Prejudice", "Jane Austen"),
    (1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
    (2701, "Moby Dick", "Herman Melville"),
    (35, "The Time Machine", "H. G. Wells"),
    (36, "The War of the Worlds", "H. G. Wells"),
    (43, "The Strange Case of Dr Jekyll and Mr Hyde", "Robert Louis Stevenson"),
    (46, "A Christmas Carol", "Charles Dickens"),
    (55, "The Wonderful Wizard of Oz", "L. Frank Baum"),
    (74, "The Adventures of Tom Sawyer", "Mark Twain"),
    (76, "Adventures of Huckleberry Finn", "Mark Twain"),
    (174, "Great Expectations", "Charles Dickens"),
    (1260, "Jane Eyre", "Charlotte Bronte"),
    (2591, "Grimms' Fairy Tales", "Jacob and Wilhelm Grimm"),
    (345, "Dracula", "Bram Stoker"),
]


def clean_gutenberg_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    text = re.sub(r"(?s)^.*?\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", "", text, count=1, flags=re.I)
    text = re.sub(r"(?s)\n\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$", "", text, count=1, flags=re.I)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    for book_id, title, author in BOOKS:
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        request = Request(url, headers={"User-Agent": "Ron-0 educational research corpus builder/1.0"})
        try:
            with urlopen(request, timeout=45) as response:
                raw = response.read()
            text = clean_gutenberg_text(raw.decode("utf-8", errors="replace"))
            if len(text) < 10_000:
                raise ValueError(f"Downloaded text is unexpectedly short ({len(text)} characters)")
            path = DEST / f"pg{book_id}.txt"
            path.write_text(text, encoding="utf-8")
            records.append({
                "ebook_id": book_id,
                "title": title,
                "author": author,
                "url": url,
                "path": str(path.relative_to(ROOT)),
                "characters": len(text),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "source_note": "Project Gutenberg; verify local public-domain status and terms before reuse.",
            })
            print(f"downloaded {book_id}: {title} ({len(text):,} chars)", flush=True)
        except (HTTPError, URLError, TimeoutError, UnicodeError, ValueError) as exc:
            failures.append({"ebook_id": book_id, "title": title, "url": url, "error": str(exc)})
            print(f"warning: could not download {book_id} ({title}): {exc}", flush=True)
        time.sleep(0.2)

    if len(records) < 5:
        raise RuntimeError(f"Only {len(records)} books downloaded successfully; at least 5 are required. Failures: {failures}")

    manifest = {
        "dataset": "Ron-0 English public-domain literature starter corpus",
        "source": "Project Gutenberg",
        "downloaded_books": len(records),
        "characters_total_before_deduplication": sum(item["characters"] for item in records),
        "license_caution": "Public-domain status varies by jurisdiction. Verify terms and source status before use; this is not legal advice.",
        "books": records,
        "download_failures": failures,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "downloaded_books": len(records),
        "characters_total_before_deduplication": manifest["characters_total_before_deduplication"],
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "failures": len(failures),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
