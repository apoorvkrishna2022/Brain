from typing import Dict, List, Any, Optional, TypeVar, Union, Callable
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END

from ..config import settings
from ..schemas.agent_state import AgentState
from ..tools.web_search import WebSearchTool
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Initialize tools
web_search_tool = WebSearchTool()

class ResearcherState(AgentState):
    """State for the researcher agent workflow"""
    research_topics: List[str] = []
    research_findings: Dict[str, str] = {}
    summary: str = ""
    
    def add_research_topic(self, topic: str) -> None:
        """Add a research topic"""
        if topic not in self.research_topics:
            self.research_topics.append(topic)
            self.update_timestamp()
            
    def add_research_finding(self, topic: str, finding: str) -> None:
        """Add a research finding for a topic"""
        self.research_findings[topic] = finding
        self.update_timestamp()
        
    def set_summary(self, summary: str) -> None:
        """Set the research summary"""
        self.summary = summary
        self.update_timestamp()

def create_system_message() -> SystemMessage:
    """Create a system message for the researcher agent"""
    return SystemMessage(content="""
    You are an expert researcher. Your job is to gather information on topics and synthesize it into clear, concise summaries.
    
    When given a topic:
    1. Break it down into 3-5 subtopics or aspects to research
    2. For each subtopic, use the web_search tool to find information
    3. Synthesize the information into a concise summary for each subtopic
    4. Finally, create an overall summary that integrates all your findings
    
    Be thorough, accurate, and focus on factual information rather than opinions or speculation.
    """)

def identify_research_topics(state: ResearcherState) -> ResearcherState:
    """Identify research topics to explore"""
    logger.info("Identifying research topics")
    
    # Create LLM agent
    agent = ChatOpenAI(
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        api_key=settings.llm.api_key
    )
    
    # Get the user's query from messages
    user_messages = [msg for msg in state.messages.messages if msg.role == "user"]
    if not user_messages:
        state.add_error("identify_research_topics", "No user message found in state")
        return state
    
    query = user_messages[-1].content
    
    # Ask the agent to identify research topics
    messages = [
        create_system_message(),
        HumanMessage(content=f"I need to research the following topic: {query}\n\nWhat are 3-5 specific subtopics or aspects I should research about this? List each one on a separate line.")
    ]
    
    response = agent.invoke(messages)
    
    # Extract topics
    topics = [line.strip() for line in response.content.split('\n') if line.strip()]
    
    # Update state
    for topic in topics:
        state.add_research_topic(topic)
        
    state.add_node_output("identify_research_topics", topics)
    state.set_next_node("research_topics")
    
    return state

def research_topics(state: ResearcherState) -> ResearcherState:
    """Research each identified topic"""
    logger.info(f"Researching {len(state.research_topics)} topics")
    
    # Create LLM agent
    agent = ChatOpenAI(
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        api_key=settings.llm.api_key
    )
    
    # Research each topic
    for topic in state.research_topics:
        if topic in state.research_findings:
            continue  # Skip if already researched
            
        logger.info(f"Researching topic: {topic}")
        
        # Search for information
        search_results = web_search_tool.run(query=topic)
        
        # Synthesize the information
        messages = [
            create_system_message(),
            HumanMessage(content=f"Research subtopic: {topic}\n\nSearch results:\n{search_results}\n\nSynthesize this information into a concise paragraph.")
        ]
        
        response = agent.invoke(messages)
        
        # Store findings
        state.add_research_finding(topic, response.content)
    
    state.add_node_output("research_topics", list(state.research_findings.keys()))
    state.set_next_node("create_summary")
    
    return state

def create_summary(state: ResearcherState) -> ResearcherState:
    """Create a final summary of all research findings"""
    logger.info("Creating research summary")
    
    # Create LLM agent
    agent = ChatOpenAI(
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        api_key=settings.llm.api_key
    )
    
    # Format the research findings
    findings_text = ""
    for topic, finding in state.research_findings.items():
        findings_text += f"## {topic}\n\n{finding}\n\n"
    
    # Ask the agent to create a summary
    messages = [
        create_system_message(),
        HumanMessage(content=f"Based on the following research findings, create a comprehensive summary:\n\n{findings_text}")
    ]
    
    response = agent.invoke(messages)
    
    # Store the summary
    state.set_summary(response.content)
    
    # Add response to message thread
    state.messages.add_assistant_message(response.content)
    
    state.add_node_output("create_summary", response.content)
    
    return state

def decide_next_step(state: ResearcherState) -> str:
    """Decide the next step in the workflow"""
    if not state.research_topics:
        return "identify_research_topics"
        
    # If we have topics but not all findings, continue research
    if len(state.research_findings) < len(state.research_topics):
        return "research_topics"
        
    # If we have all findings but no summary, create summary
    if state.research_findings and not state.summary:
        return "create_summary"
        
    # Otherwise, we're done
    return "end"

def create_researcher_agent() -> StateGraph:
    """Create the researcher agent workflow"""
    # Create the graph
    workflow = StateGraph(ResearcherState)
    
    # Add nodes
    workflow.add_node("identify_research_topics", identify_research_topics)
    workflow.add_node("research_topics", research_topics)
    workflow.add_node("create_summary", create_summary)
    
    # Add edges based on the decision function
    workflow.add_conditional_edges(
        "",  # This is the starting node (empty string)
        decide_next_step,
        {
            "identify_research_topics": "identify_research_topics",
            "research_topics": "research_topics",
            "create_summary": "create_summary",
            "end": END
        }
    )
    
    # Add regular edges
    workflow.add_edge("identify_research_topics", "research_topics")
    workflow.add_edge("research_topics", "create_summary")
    workflow.add_edge("create_summary", END)
    
    # Compile the graph
    return workflow.compile() 