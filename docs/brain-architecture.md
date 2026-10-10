# Ron-0 Native Model Architecture

Ron-0 is built around its own local reasoning core and native character-level Transformer. It is not a wrapper around Qwen or another ready-made language model, and it is separate from Ron-1.

## Local request path

User → Ron UI → Ron Core → local rules / memory / tools → Ron native model → generation-quality guard → response

Deterministic operations, including clear arithmetic expressions, should be resolved by Ron's own code before natural-language generation. A language model should not be asked to approximate calculations that a local parser can evaluate exactly.

## Learning and training

1. Maintain a reviewed corpus in `training/seed_corpus.txt`.
2. Split complete conversation pairs reproducibly into training and validation data.
3. Train Ron's native model defined in `ron/model.py`.
4. Track train and validation loss and stop when held-out validation stops improving.
5. Test the selected checkpoint against blind prompts that are not included in the training corpus.
6. Promote weights only when the quality gate passes; otherwise preserve the last validated checkpoint.

The corpus includes English examples as a controlled starting curriculum while retaining Arabic examples. English does not replace the Arabic UI or the goal of Arabic support.

## Memory is not training

Local memory stores retrievable facts and user-approved instructions. Training changes the model's numerical weights. A conversation is not automatically a reliable training example, and a saved checkpoint is not automatically a better checkpoint.

## Limits

The current native model is a small character-level prototype. It can learn basic patterns but should not be described as a capable general-purpose assistant until it demonstrates consistent results on independent tests. No external model, external weights, or cloud inference API is required by this architecture.
