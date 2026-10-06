# Ron model training

The pipeline is:

1. Qwen responses become candidates.
2. Ron approves only curated examples.
3. Approved examples become JSONL.
4. Qwen3 is fine-tuned with LoRA.
5. The resulting adapter is evaluated before promotion.
6. A merged model can later run without a live Qwen service.

Qwen's official documentation supports LoRA and Q-LoRA fine-tuning. The project starts with LoRA because it keeps the base weights separate and is easier to validate. Official guidance is available in the Qwen training documentation.

Run:
python -m training.finetune --data training/data/approved.jsonl

The trained adapter is written under artifacts/ and model weights are never committed to Git.
