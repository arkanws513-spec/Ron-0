"""Download a bounded, attributed sample of Arabic Wikipedia for language pretraining.

Wikipedia text is licensed under CC BY-SA; this script records source URLs and timestamps.
The sample is for language modeling, not a factual gold-standard dataset.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training" / "corpus" / "arabic_wikipedia_sample.txt"
MANIFEST = ROOT / "training" / "arabic_wikipedia_manifest.json"
API = "https://ar.wikipedia.org/w/api.php"
USER_AGENT = "Ron0ResearchTraining/1.0 (open-source language-model research)"
BATCHES = 30
PAGES_PER_BATCH = 10
MIN_EXTRACT_CHARS = 250


def fetch_batch() -> list[dict]:
    query = urllib.parse.urlencode({
        "action": "query",
        "generator": "random",
        "grnnamespace": 0,
        "grnlimit": PAGES_PER_BATCH,
        "prop": "extracts|info",
        "explaintext": 1,
        "exsectionformat": "plain",
        "inprop": "url",
        "format": "json",
        "formatversion": 2,
    })
    request = urllib.request.Request(
        f"{API}?{query}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    pages = payload.get("query", {}).get("pages", [])
    if isinstance(pages, dict):
        pages = list(pages.values())
    return pages


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    articles: dict[str, dict] = {}
    errors = []
    for batch_index in range(BATCHES):
        success = False
        for attempt in range(3):
            try:
                for page in fetch_batch():
                    title = str(page.get("title", "")).strip()
                    extract = str(page.get("extract", "")).strip()
                    if not title or len(extract) < MIN_EXTRACT_CHARS:
                        continue
                    articles[title] = {
                        "title": title,
                        "url": page.get("fullurl") or (
                            "https://ar.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
                        ),
                        "extract": extract,
                    }
                success = True
                break
            except Exception as exc:
                if attempt == 2:
                    errors.append({"batch": batch_index + 1, "error": str(exc)})
                else:
                    time.sleep(1.5 * (attempt + 1))
        if not success:
            continue
        print(f"Arabic Wikipedia batch {batch_index + 1}/{BATCHES}: {len(articles)} unique articles", flush=True)
        time.sleep(0.25)

    if len(articles) < 100:
        raise RuntimeError(
            f"Only {len(articles)} usable Arabic Wikipedia articles were retrieved; need at least 100. "
            f"Request errors: {len(errors)}"
        )

    chunks = []
    for article in articles.values():
        # Keep the article title as context for its first paragraph.
        paragraphs = [part.strip() for part in article["extract"].split("\n\n") if part.strip()]
        if paragraphs:
            paragraphs[0] = article["title"] + "\n" + paragraphs[0]
        chunks.extend(paragraphs)

    corpus = "\n\n".join(chunks) + "\n"
    if len(corpus) < 100_000:
        raise RuntimeError(f"Arabic corpus is too small ({len(corpus)} characters); refusing to train on a weak sample.")
    OUT.write_text(corpus, encoding="utf-8")
    manifest = {
        "source": "Arabic Wikipedia API",
        "license": "CC BY-SA; attribution and share-alike terms apply",
        "api": API,
        "unique_articles": len(articles),
        "characters": len(corpus),
        "sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
        "article_sources": [{"title": a["title"], "url": a["url"]} for a in articles.values()],
        "request_errors": errors,
        "notice": "This is an automatically sampled language-modeling corpus, not a verified factual-answer dataset. Review source licensing and dataset quality before redistribution.",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "article_sources"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
