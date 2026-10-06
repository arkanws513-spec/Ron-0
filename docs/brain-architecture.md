# Ron local brain architecture

Ron owns its identity, memory, tools, learning rules, and source code. A local
language model is only a replaceable **brain component**.

`User → Ron Core → Memory/Tools/Learning → Local Brain`

No cloud AI provider is required.

## Ready-made brain

Ron already exposes a local OpenAI-compatible adapter in `ron/providers.py`.
Any compatible model running on the same machine can be selected through
`RON_MODEL_ENDPOINT`, `RON_MODEL_NAME`, and `RON_MODEL_TIMEOUT`.

## Qwen3 teacher

Ron can also use a locally running Qwen3 model as an **explicit teacher**.
The default teacher target is `qwen3:1.7b` through a local OpenAI-compatible
endpoint. Configure it with `RON_TEACHER_ENDPOINT`, `RON_TEACHER_MODEL`, and
`RON_TEACHER_TIMEOUT`. The teacher generates examples; Ron stores the examples
with provenance and decides what can be promoted. Qwen3 does not become Ron,
and Ron does not depend on a Qwen cloud service.

Model weights are deliberately not committed to Git history.

## Knowledge transfer

`ron/brain.py` records teacher responses as Ron-owned examples with provenance.
This is knowledge transfer, not magical copying of another model's internal
weights.

The next training layer can use these examples for evaluation/fine-tuning. Any
weight update must pass Ron's evaluation gate before promotion.

## Ownership boundary

Ron remains GitHub-first and user-controlled. A third-party model, if used,
remains subject to its own license; it does not become the owner of Ron.
Provider selection is explicit and cloud providers are not hard-coded.
