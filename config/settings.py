import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

# Load environment variables
load_dotenv()

class AgentConfig(BaseModel):
    debug_mode: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))

class APIConfig(BaseModel):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

class ServiceConfig(BaseModel):
    port: int = int(os.getenv("PORT", "8000"))
    host: str = os.getenv("HOST", "0.0.0.0")

class Settings(BaseModel):
    agent: AgentConfig = AgentConfig()
    api: APIConfig = APIConfig()
    service: ServiceConfig = ServiceConfig()

# Create global settings instance
settings = Settings()

def get_settings() -> Settings:
    return settings 