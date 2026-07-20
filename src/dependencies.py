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


def _check_initialized(val, name: str):
    if val is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"Service '{name}' is still initializing. Please retry in a few seconds.")
    return val


def get_arxiv_client(request: Request) -> ArxivClient:
    """Get arXiv client from the request state."""
    return _check_initialized(request.app.state.arxiv_client, "ArxivClient")


def get_pdf_parser(request: Request) -> PDFParserService:
    """Get PDF parser service from the request state."""
    return _check_initialized(request.app.state.pdf_parser, "PDFParser")


def get_embeddings_service(request: Request) -> JinaEmbeddingsClient:
    """Get embeddings service from the request state."""
    return _check_initialized(request.app.state.embeddings_service, "EmbeddingsService")


def get_llm_client(request: Request) -> LLMClientProtocol:
    """Get LLM client from the request state (OpenAI or Bedrock depending on PROVIDER)."""
    return _check_initialized(request.app.state.llm_client, "LLMClient")


def get_guardrails_service(request: Request) -> BedrockGuardrailsService:
    """Get Bedrock Guardrails service from the request state."""
    return _check_initialized(request.app.state.guardrails_service, "GuardrailsService")


def get_langfuse_tracer(request: Request) -> LangfuseTracer:
    """Get Langfuse tracer from the request state."""
    return _check_initialized(request.app.state.langfuse_tracer, "LangfuseTracer")


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
    llm: LLMDep,
    embeddings: EmbeddingsDep,
    langfuse: LangfuseDep,
    guardrails: GuardrailsDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> "AgenticRAGService":
    """Get agentic RAG service."""
    from src.services.agents.factory import make_agentic_rag_service

    opensearch = getattr(request.app.state, "opensearch_client", None)
    pinecone = getattr(request.app.state, "pinecone_client", None)
    return make_agentic_rag_service(
        opensearch_client=opensearch,
        pinecone_client=pinecone,
        llm_client=llm,
        embeddings_client=embeddings,
        langfuse_tracer=langfuse,
        guardrails_service=guardrails,
        settings=settings,
    )


AgenticRAGDep = Annotated["AgenticRAGService", Depends(get_agentic_rag_service)]
