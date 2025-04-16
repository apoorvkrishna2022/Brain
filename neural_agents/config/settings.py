import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# Load environment variables
load_dotenv()

class LLMConfig(BaseModel):
    """Configuration for language models"""
    model: str = os.getenv("LLM_MODEL", "gpt-4")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_tokens: Optional[int] = Field(
        default=int(os.getenv("LLM_MAX_TOKENS", "1000")) if os.getenv("LLM_MAX_TOKENS") else None
    )
    api_key: str = os.getenv("OPENAI_API_KEY", "")

class AgentConfig(BaseModel):
    """Configuration for agents"""
    max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
    debug_mode: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    memory_type: str = os.getenv("MEMORY_TYPE", "buffer")
    memory_size: int = int(os.getenv("MEMORY_SIZE", "5"))
    
class ToolConfig(BaseModel):
    """Configuration for tools"""
    search_engine: str = os.getenv("SEARCH_ENGINE", "duckduckgo")
    search_api_key: Optional[str] = os.getenv("SEARCH_API_KEY", None)
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

class VisualizationConfig(BaseModel):
    """Configuration for visualizations"""
    graph_layout: str = os.getenv("GRAPH_LAYOUT", "dot")
    show_state_details: bool = os.getenv("SHOW_STATE_DETAILS", "True").lower() == "true"
    
class ServiceConfig(BaseModel):
    """Configuration for web service"""
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("SERVICE_DEBUG", "False").lower() == "true"

class Settings(BaseModel):
    """Main settings container"""
    llm: LLMConfig = LLMConfig()
    agent: AgentConfig = AgentConfig()
    tool: ToolConfig = ToolConfig()
    viz: VisualizationConfig = VisualizationConfig()
    service: ServiceConfig = ServiceConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Dependency injection for FastAPI"""
    return settings 