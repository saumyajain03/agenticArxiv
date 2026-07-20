# arXiv Agent: Production-Grade Multi-Agent RAG System

arXiv Agent is a state-of-the-art, production-grade **Multi-Agent Retrieval-Augmented Generation (RAG)** platform. It automates the process of fetching academic research papers from arXiv, performing hybrid indexing (combining semantic vector and lexical BM25 search), and generating high-fidelity literature reviews and answers using a LangGraph-coordinated multi-agent pipeline.

---

## 🏗️ System Architecture

The following diagram illustrates the complete data flow, from daily automated ingestion to real-time agentic query processing and tracing:

```mermaid
flowchart TD
    %% Ingestion Pipeline
    subgraph Ingestion ["1. Daily Ingestion Pipeline (Airflow)"]
        ARXIV_API[arXiv API] -->|Daily Fetch| AIRFLOW[Apache Airflow DAG]
        AIRFLOW -->|Download PDF| DOCLING[Docling PDF Parser]
        DOCLING -->|Layout-Aware Text| CHUNKER[Text Chunker]
        CHUNKER -->|Passages| JINA[Jina AI Embeddings v3]
        JINA -->|Vectors| OPENSEARCH[(OpenSearch Hybrid Index)]
        CHUNKER -->|Raw Text| OPENSEARCH
    end

    %% User Request Flow
    subgraph Server ["2. API & Agentic RAG Services (FastAPI + LangGraph)"]
        USER[User UI / Client] -->|Search Query| FASTAPI[FastAPI Gateway]
        FASTAPI -->|1. Exact/Semantic Check| REDIS{Upstash Redis Cache}
        
        REDIS -->|Cache Hit| USER
        REDIS -->|Cache Miss| LANGGRAPH[LangGraph Agentic RAG]
        
        subgraph Graph ["LangGraph Orchestrated Workflow"]
            START([Start]) --> GUARDRAIL{Guardrail Node}
            GUARDRAIL -->|Block / Out of Scope| REJECT[Out of Scope Node]
            GUARDRAIL -->|Approved| PLANNER[Supervisor Plan Node]
            PLANNER -->|Research Sections| RESEARCHER[Researcher Node]
            RESEARCHER -->|Parallel Drafts| WRITER[Section Writer Node]
            WRITER -->|Review Drafts| CRITIC[Peer Critic Node]
            CRITIC -->|Refine / Output| OUT_GUARDRAIL{Output Guardrail}
            OUT_GUARDRAIL -->|Approved| END([End & Synthesize])
        end
        
        LANGGRAPH -->|Execute Workflow| Graph
        END -->|Response + Sources| REDIS_SET[Store in Redis]
        REDIS_SET -->|Stream Logs & MD/PDF| USER
    end

    %% Observability
    subgraph Observability ["3. Telemetry & Monitoring"]
        LANGGRAPH -->|Spans & Observations| LANGFUSE[Langfuse Cloud]
        FASTAPI -->|Structured Logs| LOGFIRE[Pydantic Logfire]
    end
    
    style Ingestion fill:#f9f5ff,stroke:#7c3aed,stroke-width:2px
    style Server fill:#f0f7ff,stroke:#2563eb,stroke-width:2px
    style Graph fill:#f0fdf4,stroke:#16a34a,stroke-width:2px
    style Observability fill:#fff7ed,stroke:#ea580c,stroke-width:2px
```

### 🧠 LangGraph Workflow Routing
The agent workflow coordinates validation, research planning, parallel section drafting, and validation:

```mermaid
flowchart LR
    __start__(["__start__"]) --> guardrail["Guardrail Node<br/>(Local LLM / Bedrock)"]
    
    guardrail -->|Score < 60| out_of_scope["Out of Scope Node"]
    guardrail -->|Score >= 60| supervisor_plan["Supervisor Plan Node"]
    
    out_of_scope --> __end__(["__end__"])
    
    supervisor_plan --> retrieve["Retrieve Node<br/>(Hybrid OpenSearch)"]
    retrieve --> tool_retrieve["Tool Retrieve Node"]
    tool_retrieve --> grade_documents["Grade Documents Node"]
    
    grade_documents -->|Generate Answer| generate_answer["Generate Answer Node"]
    grade_documents -->|Re-plan / Rewrite| rewrite_query["Rewrite Query Node"]
    
    rewrite_query --> retrieve
    
    generate_answer --> output_guardrail["Output Guardrail Node"]
    output_guardrail --> __end__
```

---

## 🌟 Key Features

* **Multi-Agent LangGraph Orchestration**: Implements a complete multi-step RAG state machine with query validation, dynamic search re-planning, grade-based document filtering, and parallel writing agents.
* **Hybrid Search Engine**: Combines lexical (BM25) and semantic (k-NN) search via **OpenSearch** with Jina AI v3 1024-dimensional embeddings, unified using Reciprocal Rank Fusion (RRF).
* **Local & Cloud Guardrails**: Strictly monitors queries against Computer Science / AI / ML / NLP / CV research domains. Uses cloud-based AWS Bedrock Guardrails when available, falling back to a structured local LLM evaluator with strict caching-enabled routing.
* **Robust Ingestion Pipeline**: Powered by **Apache Airflow** DAGs that fetch papers daily, extract high-fidelity text layouts using **Docling**, and index them automatically.
* **Real-time Logging & Dashboard**: Real-time event streaming via EventSource (SSE) logs combined with a dashboard to preview processing states, export summaries, and print formatted PDF reports.
* **Semantic Caching & Telemetry**: Uses **Upstash Redis** for exact and semantic (vector similarity $\ge 0.92$) caching, and **Langfuse** + **Logfire** for request tracing, debugging, and quality monitoring.

---

## 📁 Project Structure

```
├── .github/workflows/      # GitHub Action CI pipelines
├── airflow/                # Apache Airflow DAGs and task setup
├── docker/                 # Service Dockerfiles (OpenSearch, etc.)
├── scripts/                # Utility and ingestion scripts
├── src/                    # Primary application codebase
│   ├── config.py           # Application settings and env validation
│   ├── dependencies.py     # FastAPI dependency injections
│   ├── main.py             # FastAPI entrypoint and routes
│   ├── mcp_server/         # Model Context Protocol (MCP) tool server
│   ├── models/             # Database ORM models (Postgres)
│   ├── repositories/       # DB query layer (Paper repositories)
│   ├── routers/            # FastAPI API endpoints
│   ├── schemas/            # Pydantic data schemas
│   ├── services/           # Business logic layer
│   │   ├── agents/         # LangGraph workflow, nodes, and supervisor
│   │   ├── arxiv/          # arXiv API retrieval client
│   │   ├── cache/          # Redis exact & semantic caching
│   │   ├── embeddings/     # Jina AI embedding service
│   │   ├── indexing/       # Text chunking and hybrid indexing
│   │   ├── pdf_generator/  # Report exporting (Markdown to PDF)
│   │   └── pdf_parser/     # Docling PDF parsing engine
│   └── static/             # Frontend Dashboard interface
├── tests/                  # PyTest suite (Unit, Integration, and Eval)
├── compose.yml             # Docker services orchestration
├── Makefile                # Fast command-line shorthands
└── pyproject.toml          # Project metadata and dependencies
```

---

## ⚡ Quick Start

### 1. Installation
Install the project dependencies using `uv`:
```bash
uv sync
```

### 2. Configure Environment
Copy `.env.example` to `.env` and configure your API credentials:
```bash
cp .env.example .env
```
*Fill in the keys for your LLM Provider, Neon PostgreSQL database, Jina AI, Upstash Redis, and Langfuse Cloud.*

### 3. Start Local Environment
Launch all local services (FastAPI App, Airflow, OpenSearch, and OpenSearch Dashboards) via Docker Compose:
```bash
make start
```

### 4. Application Endpoints
* **Deep Research Dashboard**: [http://localhost:8000](http://localhost:8000)
* **Instant Indexer**: [http://localhost:8000/indexer](http://localhost:8000/indexer)
* **API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Apache Airflow UI**: [http://localhost:8080](http://localhost:8080) *(Credentials: `admin` / `admin`)*
* **OpenSearch Dashboards**: [http://localhost:5601](http://localhost:5601)

---

## 🧪 Testing & Linting

Verify package integrity, types, and lints before shipping:

```bash
make format    # Ruff formatting
make lint      # Ruff lints & MyPy type check
make test      # Pytest execution
```
