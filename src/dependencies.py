from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, Generator, Optional

if TYPE_CHECKING:
    from fastapi import Depends, Request
    from sqlalchemy.orm import Session

    from src.services.agents.agentic_rag import AgenticRAGService
else:
    try:
        from fastapi import Depends, Request
        from sqlalchemy.orm import Session
    except ImportError:
        pass

from src.config import Settings
from src.db.interfaces.base import BaseDatabase
from src.services.arxiv.client import ArxivClient
from src.services.bedrock_guardrails.service import BedrockGuardrailsService
from src.services.cache.client import CacheClient
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.embeddings.pinecone_client import PineconeClient
from src.services.langfuse.client import LangfuseTracer
from src.services.llm_client_protocol import LLMClientProtocol
from src.services.opensearch.client import OpenSearchClient
from src.services.pdf_parser.parser import PDFParserService
from src.services.telegram.bot import TelegramBot


@lru_cache
def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


def get_request_settings(request: Request) -> Settings:
    """Get settings from the request state."""
    return request.app.state.settings


def get_database(request: Request) -> BaseDatabase:
    """Get database from the request state."""
    return request.app.state.database


def get_db_session(database: Annotated[BaseDatabase, Depends(get_database)]) -> Generator[Session, None, None]:
    """Get database session dependency."""
    if not database:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Database initializing, please try again in a few seconds.")
    with database.get_session() as session:
        yield session


def get_opensearch_client(request: Request) -> Optional[OpenSearchClient]:
    """Get OpenSearch client from the request state."""
    return getattr(request.app.state, "opensearch_client", None)


def get_pinecone_client(request: Request) -> Optional[PineconeClient]:
    """Get Pinecone client from the request state."""
    return getattr(request.app.state, "pinecone_client", None)


import logging
import traceback

logger = logging.getLogger(__name__)

def _check_initialized(val, name: str, request: Optional[Request] = None):
    logger.debug(f"ENTRY: checking dependency {name}")
    if val is None:
        init_err = getattr(request.app.state, "init_error", None) if request else None
        err_msg = f"Dependency '{name}' is unavailable."
        if init_err:
            err_msg += f" Startup error: {init_err}"
        logger.error(f"EXCEPTION: {err_msg}")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=err_msg)
    logger.debug(f"EXIT: dependency {name} is ready")
    return val

def get_arxiv_client(request: Request) -> ArxivClient:
    logger.debug("DI: get_arxiv_client")
    val = getattr(request.app.state, "arxiv_client", None)
    if val is None:
        from src.services.arxiv.factory import make_arxiv_client
        val = make_arxiv_client()
        request.app.state.arxiv_client = val
    return _check_initialized(val, "ArxivClient", request)

def get_pdf_parser(request: Request) -> PDFParserService:
    logger.debug("DI: get_pdf_parser")
    val = getattr(request.app.state, "pdf_parser", None)
    if val is None:
        from src.services.pdf_parser.factory import make_pdf_parser_service
        val = make_pdf_parser_service()
        request.app.state.pdf_parser = val
    return _check_initialized(val, "PDFParser", request)

def get_embeddings_service(request: Request) -> JinaEmbeddingsClient:
    logger.debug("DI: get_embeddings_service")
    val = getattr(request.app.state, "embeddings_service", None)
    if val is None:
        from src.services.embeddings.factory import make_embeddings_service
        val = make_embeddings_service()
        request.app.state.embeddings_service = val
    return _check_initialized(val, "EmbeddingsService", request)

def get_llm_client(request: Request) -> LLMClientProtocol:
    logger.debug("DI: get_llm_client")
    val = getattr(request.app.state, "llm_client", None)
    if val is None:
        settings = get_settings()
        if settings.provider == "bedrock":
            from src.services.bedrock_llm.factory import make_bedrock_llm_client
            val = make_bedrock_llm_client(settings)
        else:
            from src.services.openai_llm.factory import make_openai_llm_client
            val = make_openai_llm_client()
        request.app.state.llm_client = val
    return _check_initialized(val, "LLMClient", request)

def get_guardrails_service(request: Request) -> BedrockGuardrailsService:
    logger.debug("DI: get_guardrails_service")
    val = getattr(request.app.state, "guardrails_service", None)
    if val is None:
        from src.services.bedrock_guardrails.factory import make_bedrock_guardrails_service
        val = make_bedrock_guardrails_service(get_settings())
        request.app.state.guardrails_service = val
    return _check_initialized(val, "GuardrailsService", request)

def get_langfuse_tracer(request: Request) -> LangfuseTracer:
    logger.debug("DI: get_langfuse_tracer")
    val = getattr(request.app.state, "langfuse_tracer", None)
    if val is None:
        from src.services.langfuse.factory import make_langfuse_tracer
        val = make_langfuse_tracer()
        request.app.state.langfuse_tracer = val
    return _check_initialized(val, "LangfuseTracer", request)


def get_cache_client(request: Request) -> CacheClient | None:
    """Get cache client from the request state."""
    return getattr(request.app.state, "cache_client", None)


def get_telegram_service(request: Request) -> Optional[TelegramBot]:
    """Get Telegram service from the request state."""
    return getattr(request.app.state, "telegram_service", None)


# Dependency annotations
SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
OpenSearchDep = Annotated[Optional[OpenSearchClient], Depends(get_opensearch_client)]
PineconeDep = Annotated[Optional[PineconeClient], Depends(get_pinecone_client)]
ArxivDep = Annotated[ArxivClient, Depends(get_arxiv_client)]
PDFParserDep = Annotated[PDFParserService, Depends(get_pdf_parser)]
EmbeddingsDep = Annotated[JinaEmbeddingsClient, Depends(get_embeddings_service)]
LLMDep = Annotated[LLMClientProtocol, Depends(get_llm_client)]
GuardrailsDep = Annotated[BedrockGuardrailsService, Depends(get_guardrails_service)]
LangfuseDep = Annotated[LangfuseTracer, Depends(get_langfuse_tracer)]
CacheDep = Annotated[CacheClient | None, Depends(get_cache_client)]
TelegramDep = Annotated[Optional[TelegramBot], Depends(get_telegram_service)]


def get_agentic_rag_service(
    request: Request,
) -> "AgenticRAGService":
    """Get the shared agentic RAG service singleton from app state, with lazy initialization fallback."""
    service = getattr(request.app.state, "agentic_rag_service", None)
    if service is not None:
        return service

    logger.info("AgenticRAGService is None in app state. Attempting lazy on-demand initialization...")
    try:
        from src.services.agents.factory import make_agentic_rag_service
        
        # Ensure underlying components are loaded
        llm_client = getattr(request.app.state, "llm_client", None) or get_llm_client(request)
        embeddings_service = getattr(request.app.state, "embeddings_service", None) or get_embeddings_service(request)
        langfuse_tracer = getattr(request.app.state, "langfuse_tracer", None) or get_langfuse_tracer(request)
        guardrails_service = getattr(request.app.state, "guardrails_service", None) or get_guardrails_service(request)

        service = make_agentic_rag_service(
            opensearch_client=getattr(request.app.state, "opensearch_client", None),
            pinecone_client=getattr(request.app.state, "pinecone_client", None),
            llm_client=llm_client,
            embeddings_client=embeddings_service,
            langfuse_tracer=langfuse_tracer,
            guardrails_service=guardrails_service,
        )
        request.app.state.agentic_rag_service = service
        logger.info("✓ AgenticRAGService lazily initialized successfully")
        return service
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Lazy initialization of AgenticRAGService failed: {e}\n{tb}", exc_info=True)
        from fastapi import HTTPException
        init_err = getattr(request.app.state, "init_error", None)
        detail_msg = f"AgenticRAGService initialization failed: {e}"
        if init_err:
            detail_msg += f" (Background startup error: {init_err})"
        raise HTTPException(
            status_code=500,
            detail=detail_msg,
        )


AgenticRAGDep = Annotated["AgenticRAGService", Depends(get_agentic_rag_service)]

