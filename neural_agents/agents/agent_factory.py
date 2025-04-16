from typing import Dict, Any, List, Optional, Union
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph

from ..config import settings
from .researcher import create_researcher_agent
from .executor import create_executor_agent

def create_agent(agent_type: str, **kwargs) -> Union[ChatOpenAI, StateGraph]:
    """
    Factory function to create different types of agents.
    
    Args:
        agent_type: Type of agent to create ('researcher', 'executor', 'llm')
        **kwargs: Additional arguments to pass to the agent constructor
        
    Returns:
        The created agent
    """
    if agent_type == "researcher":
        return create_researcher_agent(**kwargs)
    elif agent_type == "executor":
        return create_executor_agent(**kwargs)
    elif agent_type == "llm":
        return create_llm_agent(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
def create_llm_agent(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> ChatOpenAI:
    """
    Create a basic LLM agent using OpenAI.
    
    Args:
        model: The model to use (defaults to config setting)
        temperature: The temperature to use (defaults to config setting)
        api_key: The API key to use (defaults to config setting)
        **kwargs: Additional arguments to pass to the ChatOpenAI constructor
        
    Returns:
        A ChatOpenAI instance
    """
    # Use config values if not provided
    model = model or settings.llm.model
    temperature = temperature if temperature is not None else settings.llm.temperature
    api_key = api_key or settings.llm.api_key
    
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        **kwargs
    )