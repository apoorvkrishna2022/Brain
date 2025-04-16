from typing import Dict, Any, Optional, Type, List, Union, Callable
from pydantic import BaseModel, Field, validator
from abc import ABC, abstractmethod
import inspect

class ToolInput(BaseModel):
    """Base model for tool inputs"""
    class Config:
        extra = "forbid"

class ToolOutput(BaseModel):
    """Base model for tool outputs"""
    result: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BaseTool(ABC):
    """Base class for all tools"""
    name: str
    description: str
    input_schema: Type[ToolInput]
    
    def __init__(self, name: Optional[str] = None, description: Optional[str] = None):
        if name:
            self.name = name
        if description:
            self.description = description
    
    @abstractmethod
    def _run(self, **kwargs) -> Any:
        """Implementation of the tool logic"""
        pass
    
    async def _arun(self, **kwargs) -> Any:
        """Async implementation of the tool logic"""
        return self._run(**kwargs)
    
    def run(self, **kwargs) -> ToolOutput:
        """Run the tool with the provided inputs"""
        try:
            # Validate inputs using the schema
            validated_inputs = self.input_schema(**kwargs)
            
            # Run the tool
            result = self._run(**validated_inputs.dict())
            
            return ToolOutput(result=result)
        except Exception as e:
            return ToolOutput(result=None, error=str(e))
            
    async def arun(self, **kwargs) -> ToolOutput:
        """Run the tool asynchronously"""
        try:
            # Validate inputs using the schema
            validated_inputs = self.input_schema(**kwargs)
            
            # Run the tool asynchronously
            result = await self._arun(**validated_inputs.dict())
            
            return ToolOutput(result=result)
        except Exception as e:
            return ToolOutput(result=None, error=str(e))
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's schema for LLM consumption"""
        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema.schema()
        }
        return schema 