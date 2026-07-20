# Copilot Instructions for Agentic-RAG-project

- This repo is a production-grade multi-agent Retrieval-Augmented Generation system centered on FastAPI + LangGraph + OpenSearch/Pinecone.
- The API is wired in `src/main.py`; app state is lazy-initialized in the lifespan handler and many services may be `None` until background startup completes.
- Key service factories live under `src/services/*/factory.py`. Use those factories when adding or modifying service wiring.

## What a code change usually touches

- `src/config.py` defines environment settings with nested `__` delimiters. Example: `BEDROCK__MODEL_ID`, `OPENSEARCH__HOST`, `REDIS__URL`, `PINECONE__API_KEY`.
- `src/dependencies.py` exposes FastAPI deps and returns `503` when a background-initialized service is not ready.
- `src/routers` contains HTTP routes. Important endpoints:
  - `/api/v1/ask` and `/api/v1/stream` in `src/routers/ask.py`
  - `/api/v1/ask-agentic`, `/api/v1/feedback`, `/api/v1/export-pdf`, `/api/v1/ask-stream-logs` in `src/routers/agentic_ask.py`
- `src/services/agents/factory.py` selects the provider model and builds `AgenticRAGService` based on `settings.provider` and `settings.vector_db_provider`.

## Architecture and integration points

- `src/main.py` starts the API and initializes:
  - OpenSearch or Pinecone (`settings.vector_db_provider`)
  - OpenAI or AWS Bedrock (`settings.provider`)
  - Langfuse tracing
  - Bedrock guardrails
  - Agentic RAG service singleton used by both API and Telegram bot
- The ingestion pipeline is in `airflow/`, with the main production DAG at `airflow/dags/arxiv_paper_ingestion.py`.
- The project uses `app.state` to share service instances across routers and the MCP context.
- `src/routers/ask.py` is the classic cache-first RAG path; `src/routers/agentic_ask.py` is the multi-agent reasoning workflow.

## Developer workflows

- Install dependencies: `uv sync`
- Start local services: `make start` (runs `docker compose up --build -d`)
- Stop services: `make stop`
- Tail logs: `make logs`
- Health checks: `make health`
- Format: `make format`
- Lint/type-check: `make lint`
- Run tests: `make test`

## CI-specific behavior

- The GitHub CI uses `uv sync --frozen --dev` and tests with Python 3.12.
- `pyproject.toml` is the source of truth for dependency versions and targets Python 3.12.
- Tests run with `.env.test`; external observability is disabled via `LOGFIRE__ENABLED=false` and `LANGFUSE__ENABLED=false`.
- CI runs `uv run ruff check --no-fix src/ tests/`, `uv run ruff format --check src/ tests/`, `uv run mypy src/`, and `uv run pytest`.

## What not to assume

- Do not assume the app is fully ready immediately after Uvicorn binds ports. `src/main.py` uses background initialization, and several deps can still be pending.
- Do not hardcode service hosts or credentials; use `.env` values and the `Settings` hierarchy in `src/config.py`.
- Do not bypass `Factory` functions for service creation unless the change is intentionally low-level.

## Useful entrypoints

- `src/main.py` — API application and lifespan setup
- `src/config.py` — environment schema and nested settings
- `src/dependencies.py` — FastAPI dependency injection and startup readiness checks
- `src/services/agents/factory.py` — Agentic RAG service construction
- `src/routers/ask.py` / `src/routers/agentic_ask.py` — main RAG HTTP surface
- `airflow/dags/arxiv_paper_ingestion.py` — ingestion pipeline logic
- `Makefile` — local dev commands and Docker Compose orchestration
- `.github/workflows/ci.yml` — repo CI expectations

Please review this draft and tell me if any sections feel unclear or if you want the instructions to emphasize a different area of the codebase.