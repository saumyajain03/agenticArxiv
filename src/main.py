import fcntl
import logging
import os
from contextlib import asynccontextmanager

import logfire
import uvicorn
from fastapi import FastAPI

from src.config import get_settings
from src.db.factory import make_database
from src.mcp_server.server import MCPContext, mcp, set_mcp_context
from src.routers import agentic_ask, hybrid_search, ping
from src.routers.a2a import router as a2a_router
from src.routers.ask import ask_router, stream_router
from src.routers.ingest import router as ingest_router
from src.routers.supervisor_ask import router as supervisor_router
from src.services.arxiv.factory import make_arxiv_client
from src.services.bedrock_guardrails.factory import make_bedrock_guardrails_service
from src.services.bedrock_llm.factory import make_bedrock_llm_client
from src.services.cache.factory import make_cache_client
from src.services.embeddings.factory import make_embeddings_service
from src.services.langfuse.factory import make_langfuse_tracer
from src.services.logfire.factory import configure_logfire
from src.services.openai_llm.factory import make_openai_llm_client
from src.services.opensearch.factory import make_opensearch_client
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.services.telegram.factory import make_telegram_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create MCP HTTP app once at module level.
# path="/" places the route at "/" inside the sub-app so it matches when mounted at /mcp.
_mcp_http_app = mcp.http_app(path="/", stateless_http=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan for the API."""
    async with _mcp_http_app.lifespan(app):
        logger.info("Starting RAG API...")

        # Initialize all attributes as None to prevent crashes during lazy background startup
        app.state.database = None
        app.state.pinecone_client = None
        app.state.opensearch_client = None
        app.state.arxiv_client = None
        app.state.pdf_parser = None
        app.state.embeddings_service = None
        app.state.llm_client = None
        app.state.guardrails_service = None
        app.state.langfuse_tracer = None
        app.state.cache_client = None
        app.state.agentic_rag_service = None
        app.state.supervisor_agent = None
        app.state.telegram_service = None
        app.state._telegram_lock_fd = None

        settings = get_settings()
        app.state.settings = settings

        import asyncio

        async def init_services_async():
            # Yield to event loop slightly so Uvicorn can finish starting and bind the port
            await asyncio.sleep(0.5)
            try:
                logger.info("Initializing services in background...")

                # Configure Logfire first — wires stdlib logging bridge + auto-instrumentation.
                configure_logfire(settings)
                if settings.logfire.enabled:
                    try:
                        logfire.instrument_fastapi(app, request_attributes_mapper=_skip_health)
                    except ValueError:
                        pass

                database = make_database()
                app.state.database = database
                logger.info("Database connected (background)")

                # Initialize search service (OpenSearch or Pinecone Cloud)
                if settings.vector_db_provider == "pinecone":
                    from src.services.embeddings.pinecone_client import PineconeClient

                    logger.info("Initializing Pinecone Cloud vector search client (background)...")
                    pinecone_client = PineconeClient(
                        api_key=settings.pinecone.api_key,
                        index_name=settings.pinecone.index_name,
                        environment=settings.pinecone.environment,
                    )
                    app.state.pinecone_client = pinecone_client
                    logger.info("✓ Pinecone Cloud client ready")
                else:
                    logger.info("Initializing OpenSearch search client (background)...")
                    opensearch_client = make_opensearch_client()
                    app.state.opensearch_client = opensearch_client

                    if opensearch_client.health_check():
                        logger.info("✓ OpenSearch connected successfully")
                        setup_results = opensearch_client.setup_indices(force=False)
                        if setup_results.get("hybrid_index"):
                            logger.info("Hybrid index created")
                        else:
                            logger.info("Hybrid index already exists")

                        try:
                            stats = opensearch_client.client.count(index=opensearch_client.index_name)
                            logger.info(f"OpenSearch ready: {stats['count']} documents indexed")
                        except Exception:
                            logger.info("OpenSearch index ready (stats unavailable)")
                    else:
                        logger.warning("OpenSearch connection failed - search features will be limited")

                # Initialize other services
                app.state.arxiv_client = make_arxiv_client()
                app.state.pdf_parser = make_pdf_parser_service()
                app.state.embeddings_service = make_embeddings_service()
                if settings.provider == "bedrock":
                    app.state.llm_client = make_bedrock_llm_client(settings)
                    logger.info(f"LLM provider: AWS Bedrock (model={settings.bedrock.model_id})")
                else:
                    app.state.llm_client = make_openai_llm_client()
                    logger.info(f"LLM provider: OpenAI (model={settings.openai_model})")

                app.state.guardrails_service = make_bedrock_guardrails_service(settings)
                guardrail_status = (
                    f"guardrail_id={settings.bedrock.guardrail_id}" if settings.bedrock.guardrail_id else "disabled (no guardrail_id)"
                )
                logger.info(f"Bedrock Guardrails: {guardrail_status}")

                app.state.langfuse_tracer = make_langfuse_tracer()
                app.state.cache_client = make_cache_client(settings)
                logger.info(
                    "Services initialized: arXiv API client, PDF parser, OpenSearch, Embeddings, LLM, Guardrails, Langfuse, Cache"
                )

                # Create shared agentic RAG service (used by both MCP and Telegram)
                from src.services.agents.factory import make_agentic_rag_service
                agentic_rag_service = make_agentic_rag_service(
                    opensearch_client=getattr(app.state, "opensearch_client", None),
                    pinecone_client=getattr(app.state, "pinecone_client", None),
                    llm_client=app.state.llm_client,
                    embeddings_client=app.state.embeddings_service,
                    langfuse_tracer=app.state.langfuse_tracer,
                    guardrails_service=app.state.guardrails_service,
                )
                app.state.agentic_rag_service = agentic_rag_service

                # Supervisor agent — reuses existing agentic_rag_service and context
                from src.services.agents import Context, SupervisorAgent

                supervisor_context = Context(
                    llm_client=app.state.llm_client,
                    opensearch_client=getattr(app.state, "opensearch_client", None),
                    pinecone_client=getattr(app.state, "pinecone_client", None),
                    embeddings_client=app.state.embeddings_service,
                    langfuse_tracer=app.state.langfuse_tracer,
                    guardrails_service=app.state.guardrails_service,
                    model_name=(settings.bedrock.model_id if settings.provider == "bedrock" else settings.openai_model),
                )
                app.state.supervisor_agent = SupervisorAgent(
                    context=supervisor_context,
                    agentic_rag_service=agentic_rag_service,
                )
                logger.info("SupervisorAgent initialized")

                # Wire MCP context so tools can reach all services
                if settings.mcp.enabled:
                    set_mcp_context(
                        MCPContext(
                            opensearch_client=getattr(app.state, "opensearch_client", None),
                            embeddings_client=app.state.embeddings_service,
                            llm_client=app.state.llm_client,
                            langfuse_tracer=app.state.langfuse_tracer,
                            agentic_rag_service=agentic_rag_service,
                            database=app.state.database,
                        )
                    )
                    logger.info(f"MCP server context (mounted at {settings.mcp.path})")

                # Initialize Telegram bot (Phase 7)
                _telegram_lock_fd = None
                _telegram_lock_acquired = False
                try:
                    _telegram_lock_fd = open("/tmp/telegram_bot.lock", "w")
                    fcntl.flock(_telegram_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    _telegram_lock_acquired = True
                    app.state._telegram_lock_fd = _telegram_lock_fd
                except IOError:
                    logger.info("Telegram bot lock held by another worker — skipping in this worker")

                if _telegram_lock_acquired:
                    telegram_service = make_telegram_service(
                        opensearch_client=getattr(app.state, "opensearch_client", None),
                        embeddings_client=app.state.embeddings_service,
                        llm_client=app.state.llm_client,
                        cache_client=app.state.cache_client,
                        langfuse_tracer=app.state.langfuse_tracer,
                        agentic_rag_service=agentic_rag_service,
                    )

                    if telegram_service:
                        app.state.telegram_service = telegram_service
                        try:
                            await telegram_service.start()
                            logger.info("Telegram bot started successfully")
                        except Exception as e:
                            logger.error(f"Failed to start Telegram bot: {e}")
                    else:
                        logger.info("Telegram bot not configured - skipping initialization")

                logger.info("Background initialization completed successfully. RAG API is fully ready.")
                app.state.init_error = None
            except Exception as e:
                import traceback
                app.state.init_error = e
                app.state.init_error_traceback = traceback.format_exc()
                logger.error(f"Critical error during background initialization: {e}\n{app.state.init_error_traceback}", exc_info=True)

        # Start the background task
        init_task = asyncio.create_task(init_services_async())
        app.state._init_task = init_task
        logger.info("Lifespan startup finished, yielding to Uvicorn port binding...")
        yield

        # Cleanup
        if getattr(app.state, "telegram_service", None):
            await app.state.telegram_service.stop()
            logger.info("Telegram bot stopped")

        lock_fd = getattr(app.state, "_telegram_lock_fd", None)
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass

        database = getattr(app.state, "database", None)
        if database:
            database.teardown()
        logger.info("API shutdown complete")


app = FastAPI(
    title="arXiv Paper Curator API",
    description="Personal arXiv CS.AI paper curator with RAG capabilities",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
)


def _skip_health(request, attributes):
    return {} if request.url.path == "/api/v1/health" else attributes


# Include routers
app.include_router(ping.router, prefix="/api/v1")
app.include_router(hybrid_search.router, prefix="/api/v1")
app.include_router(ask_router, prefix="/api/v1")
app.include_router(stream_router, prefix="/api/v1")
app.include_router(agentic_ask.router)
app.include_router(a2a_router)
app.include_router(supervisor_router)
app.include_router(ingest_router)

from pathlib import Path

from fastapi.responses import HTMLResponse


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    dashboard_path = Path(__file__).parent / "static" / "dashboard.html"
    if not dashboard_path.exists():
        return HTMLResponse(content="<h1>Dashboard file not found</h1>", status_code=404)
    with open(dashboard_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/indexer", response_class=HTMLResponse)
async def serve_indexer():
    indexer_path = Path(__file__).parent / "static" / "indexer.html"
    if not indexer_path.exists():
        return HTMLResponse(content="<h1>Indexer file not found</h1>", status_code=404)
    with open(indexer_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# Mount MCP sub-app (lifespan is composed inside the main lifespan above)
_mcp_settings = get_settings().mcp
if _mcp_settings.enabled:
    app.mount(_mcp_settings.path, _mcp_http_app)


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, port=port, host="0.0.0.0")
