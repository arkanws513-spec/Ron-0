"""Deterministic Arabic-aware memory with phrase and token retrieval."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from .contracts import MemoryItem

_DIACRITICS=re.compile(r"[ًٌٍَُِّْـ]")
_SPACES=re.compile(r"\s+")
_PUNCT=re.compile(r"[؟?!.,،؛;:()\[\]{}]")

def normalize_arabic(text:str)->str:
    text=_DIACRITICS.sub("",str(text))
    text=text.replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ى","ي")
    return _SPACES.sub(" ",text.lower()).strip()

def _tokens(text:str)->list[str]:
    return [x for x in _PUNCT.sub(" ",normalize_arabic(text)).split() if len(x)>=2]

def _phrases(tokens:list[str],max_size:int=6)->set[str]:
    result=set()
    for size in range(2,min(max_size,len(tokens))+1):
        result.update(" ".join(tokens[i:i+size]) for i in range(len(tokens)-size+1))
    return result

@dataclass
class InMemoryStore:
    items:dict[str,MemoryItem]=field(default_factory=dict)

    def remember(self,item:MemoryItem)->None:
        if not item.key.strip(): raise ValueError("memory key cannot be empty")
        if not item.content.strip(): raise ValueError("memory content cannot be empty")
        self.items[item.key]=item

    def recall(self,query:str,limit:int=5)->list[MemoryItem]:
        if limit<1:return []
        q=normalize_arabic(query)
        qt=_tokens(q)
        if not qt:return []
        qp=_phrases(qt)
        ranked=[]
        for position,item in enumerate(self.items.values()):
            text=normalize_arabic(f"{item.key} {item.content}")
            tt=_tokens(text)
            tp=_phrases(tt)
            score=0.0
            if text==q: score+=30
            if q and len(q)>=4 and q in text: score+=20
            score+=len(set(qt)&set(tt))*1.5
            score+=sum(len(p)*1.5 for p in qp&tp)
            if item.key.lower() in q: score+=8
            if score: ranked.append((score,position,item))
        ranked.sort(key=lambda x:(x[0],x[1]),reverse=True)
        return [item for _,_,item in ranked[:limit]]
