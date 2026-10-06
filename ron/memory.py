"""Deterministic Arabic memory with short/long tiers and lightweight relationships."""
from __future__ import annotations
import re
from dataclasses import dataclass,field
from .contracts import MemoryItem
_DIACRITICS=re.compile(r"[ًٌٍَُِّْـ]"); _SPACES=re.compile(r"\s+"); _PUNCT=re.compile(r"[؟?!.,،؛;:()\[\]{}]")
def normalize_arabic(text:str)->str:
    text=_DIACRITICS.sub("",str(text)).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ى","ي")
    return _SPACES.sub(" ",text.lower()).strip()
def _tokens(text:str)->list[str]: return [x for x in _PUNCT.sub(" ",normalize_arabic(text)).split() if len(x)>=2]
def _phrases(tokens:list[str],max_size:int=6)->set[str]:
    out=set()
    for size in range(2,min(max_size,len(tokens))+1): out.update(" ".join(tokens[i:i+size]) for i in range(len(tokens)-size+1))
    return out
@dataclass
class InMemoryStore:
    items:dict[str,MemoryItem]=field(default_factory=dict)
    short_limit:int=12
    long_limit:int=500
    def remember(self,item:MemoryItem)->None:
        if not item.key.strip(): raise ValueError("memory key cannot be empty")
        if not item.content.strip(): raise ValueError("memory content cannot be empty")
        self.items[item.key]=item
    def _rank(self,query:str):
        q=normalize_arabic(query); qt=_tokens(q); qp=_phrases(qt); ranked=[]
        for pos,item in enumerate(self.items.values()):
            text=normalize_arabic(f"{item.key} {item.content}"); tt=_tokens(text); tp=_phrases(tt); score=0.0
            if text==q: score+=30
            if len(q)>=4 and q in text: score+=20
            score+=len(set(qt)&set(tt))*1.5+sum(len(p)*1.5 for p in qp&tp)
            if item.key.lower() in q: score+=8
            if score: ranked.append((score,pos,item))
        return sorted(ranked,key=lambda x:(x[0],x[1]),reverse=True)
    def recall(self,query:str,limit:int=5)->list[MemoryItem]: return [x[2] for x in self._rank(query)[:max(1,limit)]] if limit>0 else []
    def related(self,query:str,limit:int=8)->list[MemoryItem]:
        ranked=self._rank(query); return [x[2] for x in ranked[:max(1,limit)]] if limit>0 else []
    def tiers(self)->dict[str,list[MemoryItem]]:
        values=list(self.items.values()); return {"short":values[-self.short_limit:],"long":values[:-self.short_limit][-self.long_limit:]}
