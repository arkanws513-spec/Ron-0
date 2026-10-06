"""Review exported Ron training candidates and write only approved examples."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--output",default="training/data/approved.jsonl")
    a=p.parse_args()
    source=Path(a.input)
    if not source.exists(): raise SystemExit(f"candidate file not found: {source}")
    approved=[]
    seen=set()
    for line_no,line in enumerate(source.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        try: item=json.loads(line)
        except json.JSONDecodeError as exc: raise SystemExit(f"invalid JSON on line {line_no}: {exc}") from exc
        if item.get("status")!="approved": continue
        messages=item.get("messages")
        if not isinstance(messages,list) or len(messages)<2: continue
        user=str(messages[0].get("content","")).strip()
        assistant=str(messages[-1].get("content","")).strip()
        if len(user)<2 or len(assistant)<2 or (user,assistant) in seen: continue
        seen.add((user,assistant))
        approved.append({"messages":[{"role":"user","content":user},{"role":"assistant","content":assistant}]})
    dest=Path(a.output); dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in approved),encoding="utf-8")
    print(f"approved examples: {len(approved)}")
    print(f"written: {dest}")

if __name__=="__main__": main()
