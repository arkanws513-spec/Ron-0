# Ron-0 Native Training

This directory contains the reviewed corpus used to train Ron-0's own character-level Transformer. This training path does not download or fine-tune an external base model.

## Training data

- `seed_corpus.txt` contains reviewed user/assistant examples.
- The corpus keeps Arabic examples and adds English examples as a controlled starting curriculum.
- Blind evaluation prompts are maintained separately in `scripts/surprise_eval.py` and must not be copied into the training corpus.

## Run training locally

```bash
python -m pip install -e ".[model]" pytest
python -m scripts.train_native
python -m scripts.surprise_eval
```

The native training workflow also runs when relevant model, training, or corpus files change on `main`.

## Checkpoint policy

The trainer records a hash of the corpus. When the corpus changes, it starts from the best saved inference weights and resets optimizer state rather than comparing losses from different datasets or resuming from overfit final weights. During training, the best held-out validation snapshot is retained and training stops after repeated validation checks fail to improve.

The blind evaluation measures basic generation robustness and obvious degeneration; it does not establish factual correctness or general intelligence. Candidate weights are promoted only if the anti-degeneration gate passes. A failed candidate must not replace the live checkpoint.

## Independence

Ron-0's model architecture, training loop, tokenizer/vocabulary handling, and inference provider are maintained in this repository. No Qwen, OpenAI, Google, external model weights, or external inference service is required.
