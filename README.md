# Agentic RAG System

A production-grade Agentic RAG system that ingests arXiv papers, indexes them for hybrid search (BM25 + Semantic), and answers questions using OpenRouter (Llama 3.1) and a LangGraph agent.

## Features
- **Hybrid Search**: Combines BM25 and vector search via OpenSearch.
- **Agentic Retrieval**: LangGraph agent with query validation, document grading, and fallback logic.
- **Automated Ingestion**: Apache Airflow DAG that automatically fetches and parses arXiv papers using Docling.
- **Gradio Chat**: A local browser-based chat UI.
- **Render Ready**: Comes with a `render.yaml` for easy deployment.

## Prerequisites
- Docker Desktop
- Python 3.12+ and `uv` package manager

## Quick Start

1. **Install Dependencies**
   ```bash
   uv sync
   ```

2. **Configure Environment**
   Copy `.env.example` to `.env` and fill in your keys:
   - **OpenRouter API Key** (for LLM generation)
   - **Neon Database URL** (Serverless Postgres)
   - **Upstash Redis URL** (for caching)
   - **Langfuse Cloud Keys** (for tracing/observability)
   - **Jina AI API Key** (for embeddings)

3. **Start Services**
   ```bash
   make start
   ```
   This spins up the FastAPI app, Apache Airflow, OpenSearch, and OpenSearch Dashboards locally.

4. **Verify and Access**
   - API Docs: http://localhost:8000/docs
   - Airflow UI: http://localhost:8080 (admin / admin)
   - OpenSearch Dashboards: http://localhost:5601
   - Chat UI: Run `uv run python gradio_launcher.py` and open http://localhost:7861

## Deployment to Render

The repository includes a `render.yaml` blueprint to easily deploy the entire stack to Render.

1. Connect your GitHub repository to Render.
2. Render will automatically detect the `render.yaml` file.
3. Configure your environment variables directly in the Render dashboard when prompted (e.g., API keys, database URLs).
4. The blueprint deploys:
   - **rag-api**: Your FastAPI application (Web Service)
   - **rag-airflow**: Apache Airflow for data ingestion (Background Worker)
   - **opensearch**: Your search engine (Private Service with a persistent disk)

## Development Commands

```bash
make format    # Format code with Ruff
make lint      # Run Ruff and MyPy
make test      # Run PyTest suite
make stop      # Stop Docker containers
```
