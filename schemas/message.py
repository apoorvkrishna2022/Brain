from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from uuid import uuid4

class Message(BaseModel):
    """Basic message model for agent communication"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: Literal["user", "assistant", "system", "tool"] 
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class MessageThread(BaseModel):
    """A thread of messages between agents or users and agents"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    messages: List[Message] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    def add_message(self, role: str, content: str, **kwargs) -> Message:
        """Add a message to the thread"""
        message = Message(role=role, content=content, **kwargs)
        self.messages.append(message)
        return message
    
    def get_formatted_messages(self) -> List[Dict[str, str]]:
        """Get messages formatted for LLM input"""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages] 