"""Conversation/session context owned by Ron, independent from model vendors."""
from __future__ import annotations
from dataclasses import dataclass,field
from .contracts import Message

@dataclass
class Conversation:
    session_id:str
    messages:list[Message]=field(default_factory=list)
    short_limit:int=12
    def add(self,role:str,content:str)->None:
        if not content.strip(): raise ValueError("message content cannot be empty")
        self.messages.append(Message(role=role,content=content))
    def history(self)->tuple[Message,...]: return tuple(self.messages)
    def short_history(self)->tuple[Message,...]: return tuple(self.messages[-self.short_limit:])
    def last_user_message(self)->str|None:
        for message in reversed(self.messages):
            if message.role=="user" and message.content.strip(): return message.content
        return None
    def clear(self)->None: self.messages.clear()
