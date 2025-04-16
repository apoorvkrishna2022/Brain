from typing import Dict, List, Any, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from config import settings
from tools.web_search import WebSearchTool

# Initialize tools
web_search_tool = WebSearchTool()

class ResearcherState(dict):
    """State for the researcher agent graph"""
    @property
    def messages(self) -> List[Dict[str, str]]:
        return self.get("messages", [])
    
    @property
    def next_steps(self) -> List[str]:
        return self.get("next_steps", [])
    
    @property
    def research_summary(self) -> str:
        return self.get("research_summary", "")

def create_system_message() -> SystemMessage:
    """Create a system message for the researcher agent"""
    return SystemMessage(content="""
    You are a research assistant. Your job is to research topics thoroughly and provide well-structured information.
    When given a topic or question, break it down into subtopics or components that need to be researched. 
    When researching, be sure to use the web_search tool available to you to find information.
    After gathering information, synthesize it into a coherent summary.
    """)

def create_researcher_agent() -> ChatOpenAI:
    """Create a researcher agent using OpenAI"""
    return ChatOpenAI(
        model="gpt-4",
        temperature=settings.agent.temperature,
        api_key=settings.api.openai_api_key
    )

def research_task(state: ResearcherState) -> Dict:
    """Handle research task by determining next steps"""
    agent = create_researcher_agent()
    
    messages = [create_system_message()]
    messages.extend([
        HumanMessage(content=msg["content"]) if msg["role"] == "user" else
        AIMessage(content=msg["content"]) if msg["role"] == "assistant" else
        SystemMessage(content=msg["content"])
        for msg in state.messages
    ])
    
    # Ask agent to determine research steps
    response = agent.invoke(messages + [HumanMessage(content="What are the key aspects I should research about this topic? List 3-5 specific areas to focus on.")])
    
    # Parse response to get research steps
    next_steps = response.content.split("\n")
    next_steps = [step.strip() for step in next_steps if step.strip()]
    
    return {"messages": state.messages, "next_steps": next_steps, "research_summary": ""}

def execute_research(state: ResearcherState) -> Dict:
    """Execute research on each of the identified steps"""
    agent = create_researcher_agent()
    
    # Base messages for context
    messages = [create_system_message()]
    messages.extend([
        HumanMessage(content=msg["content"]) if msg["role"] == "user" else
        AIMessage(content=msg["content"]) if msg["role"] == "assistant" else
        SystemMessage(content=msg["content"])
        for msg in state.messages
    ])
    
    # Research each step
    findings = []
    for step in state.next_steps:
        # Search for information
        search_result = web_search_tool.run(step)
        
        # Ask agent to synthesize search results
        step_messages = messages + [
            HumanMessage(content=f"Research subtopic: {step}\n\nSearch results: {search_result}\n\nSynthesize this information into a concise paragraph.")
        ]
        
        response = agent.invoke(step_messages)
        findings.append(f"# {step}\n\n{response.content}")
    
    # Combine all findings
    all_findings = "\n\n".join(findings)
    
    return {"messages": state.messages, "next_steps": state.next_steps, "research_summary": all_findings}

def summarize_research(state: ResearcherState) -> Dict:
    """Create a final summary of all research"""
    agent = create_researcher_agent()
    
    # Ask the agent to create a final summary
    messages = [
        create_system_message(),
        HumanMessage(content=f"Based on all the research below, create a comprehensive summary:\n\n{state.research_summary}")
    ]
    
    response = agent.invoke(messages)
    
    # Add summary to messages for the user
    updated_messages = state.messages + [{"role": "assistant", "content": response.content}]
    
    return {"messages": updated_messages, "next_steps": [], "research_summary": state.research_summary}

def should_continue_research(state: ResearcherState) -> str:
    """Decide whether to continue with more research or finalize"""
    if not state.research_summary:
        return "execute_research"
    return "summarize"

def create_researcher_graph() -> StateGraph:
    """Create the researcher agent workflow graph"""
    workflow = StateGraph(ResearcherState)
    
    # Add nodes to the graph
    workflow.add_node("research", research_task)
    workflow.add_node("execute_research", execute_research)
    workflow.add_node("summarize", summarize_research)
    
    # Define edges
    workflow.add_edge("research", "execute_research")
    workflow.add_conditional_edges(
        "execute_research",
        should_continue_research,
        {
            "execute_research": "execute_research",
            "summarize": "summarize"
        }
    )
    workflow.add_edge("summarize", END)
    
    # Set entry point
    workflow.set_entry_point("research")
    
    return workflow

researcher_graph = create_researcher_graph().compile() 