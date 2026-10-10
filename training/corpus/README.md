# Ron-10M training workflow

This workflow is the dedicated larger-model path. It does not overwrite the existing small baseline checkpoint.

## Model profile

- Decoder-only causal Transformer, trained from scratch unless the Ron-10M checkpoint exists.
- Hidden size: 384
- Transformer blocks: 6
- Attention heads: 6
- Context length: 256 characters
- Parameter count depends on the actual character vocabulary; the target configuration is approximately 10.7 million parameters for a typical text vocabulary.

This is still a character-level research prototype, not a subword-tokenized general-purpose assistant. The larger parameter count is a capacity experiment, not proof of higher intelligence.

## Prepare data

1. Put UTF-8 .txt files that you have the right to use under training/corpus/.
2. Keep a record of each dataset's source, license, language, and collection date. Do not include secrets or private conversations without valid permission.
3. Separate documents or paragraphs with blank lines. The preparation script normalizes Unicode, drops short paragraphs, removes exact duplicate paragraphs, and creates a manifest with hashes.
4. Run:

   python scripts/prepare_corpus.py

The script writes training/corpus.txt and training/corpus_manifest.json. It requires at least 100,000 characters and at least 10 usable paragraphs. This is a minimum gate for running the experiment, not a claim that the corpus is large enough to train a capable language model. For meaningful language modeling, plan for millions of high-quality characters/tokens and broad coverage.

## Train

Install the model extra, then run:

    pip install -e ".[model]"
    python scripts/train_native_10m.py

Optional environment settings:

- RON_STEPS (default 1800)
- RON_BATCH_SIZE (default 2)
- RON_SEQUENCE_LENGTH (default 256; changing this also changes checkpoint compatibility)

Outputs go to artifacts/native-10m/; the resumable checkpoint is ron/checkpoints/ron_native_10m.pt. The training script stops if the prepared corpus is missing or below the minimum size. It records training/validation losses, parameter count, corpus hash, and checkpoint provenance.

## Important limitations

- Do not commit large raw datasets or model checkpoints to Git unless that is an intentional storage decision; use appropriate artifact storage.
- A character-level model spends capacity on individual characters and generally needs much more context and data than a good subword tokenizer.
- Training/validation splitting by paragraph reduces direct leakage but is not an independent benchmark.
- Review corpus licenses, factual quality, duplication, and language balance before training.
- Compare against held-out prompts and the previous checkpoint before deciding whether to replace Ron's active model.

## Automated experiment

The `Train Ron-10M` GitHub Actions workflow runs on reviewed changes to the training recipe and can also be started manually from the Actions tab. It stores the candidate checkpoint and metrics as an Actions artifact rather than committing multi-megabyte weights or downloaded books to Git. The live Ron-10M checkpoint is promoted only if held-out validation loss improves; otherwise the previous live checkpoint is preserved. Review the artifact's metrics and generation behavior before deploying a candidate.
