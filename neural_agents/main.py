import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union

from config import settings
from schemas.message import Message, MessageThread
from schemas.agent_state import AgentState
from agents import create_agent
from utils.visualization import visualize_graph
from utils.logger import get_logger

logger = get_logger("main")

app = FastAPI(
    title="Neural Agent System",
    description="A powerful neural agent system built with LangGraph and LangChain",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class QueryRequest(BaseModel):
    """Model for query requests"""
    query: str
    agent_type: str = Field(default="researcher", description="Type of agent to use (researcher, executor)")
    context: Optional[str] = Field(default=None, description="Additional context for the agent")

class AgentResponse(BaseModel):
    """Model for agent responses"""
    result: str
    details: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to the Neural Agent System"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/query", response_model=AgentResponse)
async def process_query(request: QueryRequest = Body(...)):
    """
    Process a query using the specified agent type
    """
    try:
        logger.info(f"Processing query with agent type: {request.agent_type}")
        
        # Create agent graph
        agent_graph = create_agent(request.agent_type)
        
        # Initialize state with user message
        if request.agent_type == "researcher":
            from agents.researcher import ResearcherState
            state = ResearcherState()
        elif request.agent_type == "executor":
            from agents.executor import ExecutorState
            state = ExecutorState()
        else:
            raise ValueError(f"Unknown agent type: {request.agent_type}")
            
        # Add user message
        state.messages.add_user_message(request.query)
        
        # Add context if provided
        if request.context:
            state.messages.add_system_message(request.context)
        
        # Run the agent
        logger.info("Running agent workflow")
        final_state = agent_graph.invoke(state)
        
        # Extract result from messages
        assistant_messages = [msg for msg in final_state.messages.messages if msg.role == "assistant"]
        result = assistant_messages[-1].content if assistant_messages else "No response generated."
        
        # Create response
        response = AgentResponse(
            result=result,
            details={
                "agent_type": request.agent_type,
                "node_outputs": {k: v.output for k, v in final_state.node_outputs.items()},
                "errors": final_state.errors
            }
        )
        
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/visualize/{agent_type}")
async def visualize_agent(agent_type: str):
    """
    Visualize an agent's workflow graph
    """
    try:
        agent_graph = create_agent(agent_type)
        svg = visualize_graph(agent_graph)
        
        return {"svg": svg}
    except Exception as e:
        logger.error(f"Error visualizing agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error visualizing agent: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.service.host,
        port=settings.service.port,
        reload=settings.service.debug
    )