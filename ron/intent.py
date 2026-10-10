"""Arabic intent classifier owned by Ron, independent of model vendors."""
from __future__ import annotations
import re
from dataclasses import dataclass
_ARABIC_DIACRITICS=re.compile(r"[ًٌٍَُِّْـ]")
_SPACES=re.compile(r"\s+")
def normalize_arabic(text:str)->str:
    text=_ARABIC_DIACRITICS.sub("",str(text)).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ى","ي")
    return _SPACES.sub(" ",text.lower()).strip()
@dataclass(frozen=True)
class Intent:
    name:str
    confidence:float
    topic:str|None=None
    is_follow_up:bool=False
_FOLLOWUPS={"طيب","طب","وبعدين","وماذا عنه","وماذا عنها","وهل","وضح","اشرح اكثر","كمل","تابع","ماذا تقصد","ليه","لماذا","ازاي","كيف ذلك","لماذا ذلك","ما السبب","ما معنى ذلك","هل هذا صحيح"}
def detect_intent(text:str,topic:str|None=None)->Intent|None:
    value=normalize_arabic(text)
    if not value:return None
    if value.rstrip("؟?!.، ") in _FOLLOWUPS:return Intent("follow_up",.99,topic,True)
    if value in {"مرحبا","اهلا","السلام عليكم","سلام عليكم","هاي"}:return Intent("greeting",1.0,topic)
    if any(p in value for p in ("ما اسمك","ايه اسمك","من انت","ما هو اسمك")):return Intent("assistant_identity",.98,topic)
    if any(p in value for p in ("ما اسمي","ايه اسمي","ما هو اسمي","هل تتذكر اسمي")):return Intent("user_name",.99,topic)
    if any(p in value for p in ("ماذا احب","ما الذي احبه","ايه اللي بحبه","هل تتذكر ما احب")):return Intent("user_preferences",.98,topic)
    if re.search(r"[؟?]",value) or re.match(r"^(ليه|لماذا|ازاي|كيف|ماذا|ما هو|ما هي|هل|هل يمكن|ممكن|عايز|اريد)\b",value):return Intent("question",.9,topic)
    if value.startswith(("تعلم ","علّم رون","احفظ ","سجل ")):return Intent("learning",.9,topic)
    return Intent("statement",.7,topic)
