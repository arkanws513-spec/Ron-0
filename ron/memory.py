"""Deterministic Arabic memory with short/long tiers and lightweight relationships."""
from __future__ import annotations
import re
import json
import sqlite3
from pathlib import Path
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


class SQLiteMemoryStore(InMemoryStore):
    """SQLite-backed local memory that survives process restarts.

    The store mirrors records into the inherited in-memory index for fast retrieval.
    Configure it explicitly with RON_MEMORY_DB_PATH; the default remains in-memory
    so tests and isolated sessions never share personal data accidentally.
    """

    def __init__(self, path: str) -> None:
        self._json = json
        self._sqlite3 = sqlite3
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__()
        with self._sqlite3.connect(self.path) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS ron_memory (
                    key TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )"""
            )
            rows = connection.execute(
                "SELECT key, content, metadata_json FROM ron_memory ORDER BY rowid"
            ).fetchall()
        for key, content, metadata_json in rows:
            try:
                metadata = self._json.loads(metadata_json)
                if not isinstance(metadata, dict):
                    metadata = {}
            except (TypeError, ValueError):
                metadata = {}
            self.items[key] = MemoryItem(key=key, content=content, metadata=metadata)

    def remember(self, item: MemoryItem) -> None:
        # Validate before touching persistent state.
        if not item.key.strip():
            raise ValueError("memory key cannot be empty")
        if not item.content.strip():
            raise ValueError("memory content cannot be empty")
        metadata_json = self._json.dumps(item.metadata, ensure_ascii=False, sort_keys=True)
        with self._sqlite3.connect(self.path) as connection:
            connection.execute(
                """INSERT INTO ron_memory(key, content, metadata_json)
                   VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                     content = excluded.content,
                     metadata_json = excluded.metadata_json""",
                (item.key, item.content, metadata_json),
            )
        self.items[item.key] = item
