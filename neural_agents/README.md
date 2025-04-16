# Neural Agent System

A modular, extensible neural agent system built with LangGraph, LangChain, and FastAPI.

## Features

- 🧠 **LangGraph-based Agent Architecture**: Modular agent design with state management and workflow control
- 🔌 **Multiple Agent Types**: Specialized agents for research, execution, and more
- 🔧 **Rich Toolset**: Web search, file operations, and other tools for agents to use
- 📊 **Visualization**: Graph visualization of agent workflows
- 🌐 **FastAPI Backend**: High-performance REST API
- ⚙️ **Configurable**: Environment-based configuration with sensible defaults

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-agents.git
cd neural-agents

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Running the API

```bash
cd neural_agents
python main.py
```

The API will be available at http://localhost:8000.

### API Endpoints

- `GET /`: Welcome message
- `GET /health`: Health check
- `POST /query`: Submit a query to an agent
- `GET /visualize/{agent_type}`: Visualize an agent's workflow

### Example Query

```json
{
  "query": "What are the recent advances in neural networks?",
  "agent_type": "researcher",
  "context": "Focus on developments in the last 2 years"
}
```

## Project Structure

```
neural_agents/
├── agents/                # Agent implementations
│   ├── researcher.py      # Research agent
│   ├── executor.py        # Task execution agent
│   └── agent_factory.py   # Factory for creating agents
├── config/                # Configuration
│   └── settings.py        # Settings loaded from .env
├── schemas/               # Data models
│   ├── agent_state.py     # Agent state models
│   └── message.py         # Message models
├── tools/                 # Tools for agents
│   ├── web_search.py      # Web search tool
│   └── file_operations.py # File operations tools
├── utils/                 # Utilities
│   ├── logger.py          # Logging utilities
│   └── visualization.py   # Graph visualization
├── main.py                # FastAPI application
├── requirements.txt       # Dependencies
└── .env.example           # Example environment variables
```

## Extending the System

### Adding a New Agent

1. Create a new file in the `agents` directory
2. Define the agent's state model, node functions, and workflow
3. Add the agent to the `agent_factory.py` file
4. Use the agent through the API

### Adding a New Tool

1. Create a new file in the `tools` directory
2. Extend the `BaseTool` class with your implementation
3. Add the tool to the `tools/__init__.py` file
4. Use the tool in your agents

## License

MIT 