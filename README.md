# EAG V3 Session 7: Agent System with LLM Gateway

A highly capable cognitive AI agent that breaks down tasks, manages persistent vector memory, and routes all LLM calls through a custom load-balancing LLM Gateway.

## 🧠 Architecture Overview

The system is split into two major components:

1. **The Agentic Loop (`agent7.py`)**: A four-layer cognitive loop that runs up to 20 iterations per query:
   - **Memory**: Vector (FAISS) + Keyword retrieval of past facts and tool outcomes.
   - **Perception**: Decomposes the query and memory into an invariant list of goals.
   - **Decision**: Analyzes the current goal and picks exactly one tool to call (or answers directly).
   - **Action**: Dispatches the tool call to an MCP Server and saves large results to an Artifact store.

2. **LLM Gateway V7 (`llm_gatewayV7/`)**: A local FastAPI server that provides unified, free-tier access to multiple LLM providers (Gemini, Groq, NVIDIA, Cerebras, OpenRouter, GitHub, Ollama). It handles:
   - **Tiered Routing**: A fast "router" LLM classifies prompts as TINY/LARGE and routes them to the best free provider.
   - **Automatic Failover**: If a provider rate-limits, it instantly fails over to the next one.
   - **Embeddings**: Provides a unified 768-dim embedding endpoint using local Ollama (fallback to Gemini).

## 🚀 Setup & Installation

### Prerequisites
- Python ≥ 3.11
- `uv` package manager
- [Ollama](https://ollama.com/) (for local embeddings and fallback models)

### 1. Environment Variables
Add your API keys to the `.env` file in the root directory (this is shared with the Gateway).
```env
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
NVIDIA_API_KEY=your_key
CEREBRAS_API_KEY=your_key
OPEN_ROUTER_API_KEY=your_key
GITHUB_ACCESS_TOKEN=your_key

OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:12b
```

### 2. Download Ollama Models
The gateway relies on Ollama for local embeddings and worker tasks:
```bash
ollama pull nomic-embed-text
ollama pull gemma4:12b
```

## 💻 Running the Agent

You do **not** need to start the gateway manually; the agent will auto-start it for you on port 8107.

Run the agent with your query:
```bash
uv run agent7.py "Fetch the latest news on AI and summarize the top 3 stories"
```

## 📂 Project Structure

- `agent7.py` - The main orchestrator and agent loop.
- `perception.py` & `decision.py` - The "brain" layers containing the core LLM prompts and logic.
- `memory.py` & `vector_index.py` - Handles long-term persistent memory and FAISS indexing.
- `artifacts.py` - Content-addressable blob store for large tool outputs (>4KB).
- `mcp_server.py` - Contains all 11 tools (web search, file ops, document indexing).
- `llm_gatewayV7/` - The standalone FastAPI LLM load balancer.
- `state/` - Where the persistent database (`memory.json`) and vector index (`index.faiss`) live.
- `sandbox/` - The isolated folder for file operations (`read_file`, `create_file`).

## 📚 Further Reading

For a deep dive into the code mechanics, data flows, and design decisions, please see the [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md).
