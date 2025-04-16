# Neural Agent System

A neural agent system built with LangGraph, LangChain, and FastAPI.

## Features

- 🧠 LangGraph-based neural agent architecture
- 🔍 Research agent that can search and synthesize information
- 🔄 Modular and extensible design
- 🌐 REST API with FastAPI

## Setup

1. Clone the repository:
```bash
git clone https://github.com/apoorvkrishna2022/Brain.git
cd Brain
```

2. Create and activate the virtual environment:
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file based on `.env.example` and add your API keys:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

Start the API server:
```bash
python app.py
```

The API will be available at http://localhost:8000

## API Endpoints

- `GET /`: Welcome message
- `GET /health`: Health check
- `POST /research`: Run research using the neural agent

### Example Research Request

```json
{
  "query": "What are the latest advancements in large language models?",
  "context": "Focus on developments in the last 6 months"
}
```

## Project Structure

- `agents/`: Neural agent implementations
- `tools/`: Tools used by agents
- `schemas/`: Data models and schemas
- `config/`: Configuration and settings
- `app.py`: Main application file

## Extending the System

To add new agents:
1. Create a new agent file in the `agents/` directory
2. Define the agent's state, nodes, and edges using LangGraph
3. Add API endpoints in `app.py` to interact with your agent 