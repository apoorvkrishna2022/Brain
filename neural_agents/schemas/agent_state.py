from typing import Dict, List, Any, Optional, TypeVar, Generic
from pydantic import Field
from datetime import datetime

from .base import BaseSchema
from .message import Message, MessageThread

T = TypeVar('T')

class NodeOutput(BaseSchema, Generic[T]):
    """Output from a LangGraph node"""
    node_name: str
    output: T
    status: str = "completed"
    error: Optional[str] = None

class AgentState(BaseSchema):
    """Base state for agent graphs in LangGraph"""
    messages: MessageThread = Field(default_factory=MessageThread)
    current_node: Optional[str] = None
    next_node: Optional[str] = None
    node_outputs: Dict[str, NodeOutput] = Field(default_factory=dict)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    
    def add_node_output(self, node_name: str, output: Any, status: str = "completed", error: Optional[str] = None) -> None:
        """Add output from a node"""
        node_output = NodeOutput(
            node_name=node_name,
            output=output,
            status=status,
            error=error
        )
        self.node_outputs[node_name] = node_output
        self.update_timestamp()
        
    def add_error(self, node_name: str, error_message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Add an error to the state"""
        error = {
            "node": node_name,
            "message": error_message,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.errors.append(error)
        self.update_timestamp()
        
    def set_next_node(self, node_name: str) -> None:
        """Set the next node to execute"""
        self.current_node = self.next_node
        self.next_node = node_name
        self.update_timestamp()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary"""
        return {
            "messages": [m.dict() for m in self.messages.messages],
            "current_node": self.current_node,
            "next_node": self.next_node,
            "node_outputs": {k: v.dict() for k, v in self.node_outputs.items()},
            "errors": self.errors
        } 