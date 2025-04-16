from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import Field, validator
from datetime import datetime

from .base import BaseSchema

class Message(BaseSchema):
    """Message model for agent communication"""
    role: Literal["user", "assistant", "system", "tool", "function"]
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    
    @validator("role")
    def validate_role(cls, role):
        """Validate role field"""
        valid_roles = ["user", "assistant", "system", "tool", "function"]
        if role not in valid_roles:
            raise ValueError(f"Role must be one of {valid_roles}")
        return role

class MessageThread(BaseSchema):
    """A thread of messages between agents or users and agents"""
    messages: List[Message] = Field(default_factory=list)
    
    def add_user_message(self, content: str) -> Message:
        """Add a user message to the thread"""
        message = Message(role="user", content=content)
        self.messages.append(message)
        self.update_timestamp()
        return message
    
    def add_assistant_message(self, content: str) -> Message:
        """Add an assistant message to the thread"""
        message = Message(role="assistant", content=content)
        self.messages.append(message)
        self.update_timestamp()
        return message
        
    def add_system_message(self, content: str) -> Message:
        """Add a system message to the thread"""
        message = Message(role="system", content=content)
        self.messages.append(message)
        self.update_timestamp()
        return message
    
    def add_tool_message(self, content: str, name: str, tool_call_id: str) -> Message:
        """Add a tool message to the thread"""
        message = Message(
            role="tool", 
            content=content, 
            name=name,
            tool_call_id=tool_call_id
        )
        self.messages.append(message)
        self.update_timestamp()
        return message
    
    def get_formatted_messages(self) -> List[Dict[str, Any]]:
        """Get messages formatted for LLM input"""
        formatted = []
        for msg in self.messages:
            message_dict = {"role": msg.role, "content": msg.content}
            
            if msg.name:
                message_dict["name"] = msg.name
                
            if msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id
                
            if msg.tool_calls:
                message_dict["tool_calls"] = msg.tool_calls
                
            formatted.append(message_dict)
            
        return formatted
    
    def clear(self) -> None:
        """Clear all messages in the thread"""
        self.messages = []
        self.update_timestamp()