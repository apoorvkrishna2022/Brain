from typing import Dict, List, Any, Optional, Union, Callable
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END

from ..config import settings
from ..schemas.agent_state import AgentState
from ..tools.file_operations import FileReadTool, FileWriteTool
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Initialize tools
file_read_tool = FileReadTool()
file_write_tool = FileWriteTool()

class ExecutorState(AgentState):
    """State for the executor agent workflow"""
    tasks: List[Dict[str, Any]] = []
    current_task_index: int = 0
    completed_tasks: List[Dict[str, Any]] = []
    
    def add_task(self, task_description: str, task_type: str = "general") -> None:
        """Add a task to the execution queue"""
        task = {
            "description": task_description,
            "type": task_type,
            "status": "pending",
            "result": None
        }
        self.tasks.append(task)
        self.update_timestamp()
        
    def mark_current_task_complete(self, result: Any) -> None:
        """Mark the current task as complete"""
        if self.current_task_index < len(self.tasks):
            self.tasks[self.current_task_index]["status"] = "completed"
            self.tasks[self.current_task_index]["result"] = result
            self.completed_tasks.append(self.tasks[self.current_task_index])
            self.current_task_index += 1
            self.update_timestamp()
            
    def get_current_task(self) -> Optional[Dict[str, Any]]:
        """Get the current task in the queue"""
        if self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index]
        return None

def create_system_message() -> SystemMessage:
    """Create a system message for the executor agent"""
    return SystemMessage(content="""
    You are an executor agent that can perform tasks based on instructions.
    You have the following capabilities:
    1. Reading files
    2. Writing files
    3. Executing general tasks that involve reasoning and planning
    
    Follow instructions precisely and report back your results in a clear, structured format.
    """)

def parse_tasks(state: ExecutorState) -> ExecutorState:
    """Parse the user's request into specific tasks"""
    logger.info("Parsing tasks from user request")
    
    # Create LLM agent
    agent = ChatOpenAI(
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        api_key=settings.llm.api_key
    )
    
    # Get the user's request from messages
    user_messages = [msg for msg in state.messages.messages if msg.role == "user"]
    if not user_messages:
        state.add_error("parse_tasks", "No user message found in state")
        return state
    
    request = user_messages[-1].content
    
    # Ask the agent to break down the request into tasks
    messages = [
        create_system_message(),
        HumanMessage(content=f"I need you to execute the following:\n\n{request}\n\nBreak this down into a list of specific tasks that need to be performed. List each task on a separate line.")
    ]
    
    response = agent.invoke(messages)
    
    # Extract tasks
    tasks = [line.strip() for line in response.content.split('\n') if line.strip()]
    
    # Add tasks to state
    for task in tasks:
        state.add_task(task)
    
    state.add_node_output("parse_tasks", tasks)
    state.set_next_node("execute_tasks")
    
    return state

def execute_tasks(state: ExecutorState) -> ExecutorState:
    """Execute the current task in the queue"""
    # Get the current task
    task = state.get_current_task()
    
    if not task:
        logger.info("No tasks to execute")
        return state
    
    logger.info(f"Executing task: {task['description']}")
    
    # Create LLM agent
    agent = ChatOpenAI(
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        api_key=settings.llm.api_key
    )
    
    # Execute based on task type
    if "read" in task["description"].lower() and "file" in task["description"].lower():
        # This is likely a file read task
        messages = [
            create_system_message(),
            HumanMessage(content=f"I need to execute this task: {task['description']}\n\nWhat file path should I read from? Extract just the file path.")
        ]
        
        response = agent.invoke(messages)
        file_path = response.content.strip()
        
        # Read the file
        result = file_read_tool.run(file_path=file_path)
        
    elif "write" in task["description"].lower() and "file" in task["description"].lower():
        # This is likely a file write task
        messages = [
            create_system_message(),
            HumanMessage(content=f"I need to execute this task: {task['description']}\n\nProvide the file path and content I should write in this format:\nFILE PATH: <path>\nCONTENT:\n<content>")
        ]
        
        response = agent.invoke(messages)
        
        # Extract file path and content
        parts = response.content.split("CONTENT:", 1)
        if len(parts) == 2:
            file_path_part = parts[0]
            content = parts[1].strip()
            
            file_path = file_path_part.replace("FILE PATH:", "").strip()
            
            # Write to the file
            result = file_write_tool.run(file_path=file_path, content=content)
        else:
            result = "Failed to parse file path and content"
    else:
        # General task
        messages = [
            create_system_message(),
            HumanMessage(content=f"I need to execute this task: {task['description']}\n\nProvide a step-by-step approach to complete this task and then execute it. Report your result.")
        ]
        
        response = agent.invoke(messages)
        result = response.content
    
    # Mark the task as complete
    state.mark_current_task_complete(result)
    
    state.add_node_output("execute_tasks", {
        "task": task["description"],
        "result": result
    })
    
    return state

def final_report(state: ExecutorState) -> ExecutorState:
    """Create a final report of all executed tasks"""
    logger.info("Creating final report")
    
    # Create LLM agent
    agent = ChatOpenAI(
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        api_key=settings.llm.api_key
    )
    
    # Format the tasks and results
    tasks_text = ""
    for i, task in enumerate(state.completed_tasks):
        tasks_text += f"## Task {i+1}: {task['description']}\n\n"
        tasks_text += f"**Result:** {task['result']}\n\n"
    
    # Ask the agent to create a summary report
    messages = [
        create_system_message(),
        HumanMessage(content=f"I've completed the following tasks:\n\n{tasks_text}\n\nProvide a summary report of what was accomplished.")
    ]
    
    response = agent.invoke(messages)
    
    # Add response to message thread
    state.messages.add_assistant_message(response.content)
    
    state.add_node_output("final_report", response.content)
    
    return state

def decide_next_step(state: ExecutorState) -> str:
    """Decide the next step in the workflow"""
    if not state.tasks:
        return "parse_tasks"
        
    # If there are still tasks to execute
    if state.current_task_index < len(state.tasks):
        return "execute_tasks"
        
    # If all tasks are complete, generate a report
    return "final_report"

def create_executor_agent() -> StateGraph:
    """Create the executor agent workflow"""
    # Create the graph
    workflow = StateGraph(ExecutorState)
    
    # Add nodes
    workflow.add_node("parse_tasks", parse_tasks)
    workflow.add_node("execute_tasks", execute_tasks)
    workflow.add_node("final_report", final_report)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "",  # Starting node
        decide_next_step,
        {
            "parse_tasks": "parse_tasks",
            "execute_tasks": "execute_tasks",
            "final_report": "final_report"
        }
    )
    
    # Add regular edges
    workflow.add_edge("parse_tasks", "execute_tasks")
    workflow.add_edge("execute_tasks", decide_next_step)
    workflow.add_edge("final_report", END)
    
    # Compile the graph
    return workflow.compile() 