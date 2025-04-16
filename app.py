import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from config import get_settings, settings
from agents.researcher import researcher_graph, ResearcherState

app = FastAPI(title="Neural Agent System", description="A system of neural agents built with LangGraph")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    """Model for research requests"""
    query: str
    context: Optional[str] = None

class ResearchResponse(BaseModel):
    """Model for research responses"""
    result: str
    detailed_findings: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Neural Agent System API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest = Body(...)):
    """Conduct research on a topic using the researcher agent"""
    try:
        # Initialize state
        messages = [{"role": "user", "content": request.query}]
        
        if request.context:
            messages.insert(0, {"role": "system", "content": request.context})
        
        initial_state = ResearcherState(messages=messages)
        
        # Run the agent
        final_state = researcher_graph.invoke(initial_state)
        
        # Extract results
        assistant_messages = [msg["content"] for msg in final_state["messages"] if msg["role"] == "assistant"]
        result = assistant_messages[-1] if assistant_messages else "No results found."
        
        return ResearchResponse(
            result=result,
            detailed_findings=final_state["research_summary"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing research: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host=settings.service.host, 
        port=settings.service.port, 
        reload=True
    ) 