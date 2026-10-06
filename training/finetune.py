"""Train a Ron-specific adapter from approved Qwen examples."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="Qwen/Qwen3-1.7B")
    p.add_argument("--data",required=True)
    p.add_argument("--output",default="artifacts/ron-qwen-lora")
    p.add_argument("--epochs",type=float,default=1.0)
    p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--grad-accumulation",type=int,default=8)
    p.add_argument("--max-length",type=int,default=2048)
    a=p.parse_args()
    path=Path(a.data)
    if not path.exists(): raise SystemExit(f"training data not found: {path}")
    ds=load_dataset("json",data_files=str(path),split="train")
    if "messages" not in ds.column_names: raise SystemExit("dataset must contain messages")
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype="auto",device_map="auto",trust_remote_code=True)
    lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules="all-linear",task_type="CAUSAL_LM")
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    trainer=SFTTrainer(
        model=model,tokenizer=tok,train_dataset=ds,peft_config=lora,
        formatting_func=lambda row: tok.apply_chat_template(row["messages"],tokenize=False,add_generation_prompt=False),
        max_seq_length=a.max_length,
        args=TrainingArguments(output_dir=str(out),num_train_epochs=a.epochs,per_device_train_batch_size=a.batch_size,
            gradient_accumulation_steps=a.grad_accumulation,learning_rate=1e-4,logging_steps=5,save_steps=100,save_total_limit=2,report_to="none"))
    trainer.train(); trainer.save_model(str(out)); tok.save_pretrained(str(out))
    (out/"ron-model-manifest.json").write_text(json.dumps({"base_model":a.model,"method":"LoRA SFT","dataset":str(path),"independent_identity":True,"qwen_required_at_runtime":False},ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__": main()
