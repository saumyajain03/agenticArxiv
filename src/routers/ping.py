from fastapi import APIRouter, Request
from fastapi.responses import Response
from sqlalchemy import text

from ..dependencies import DatabaseDep, OpenSearchDep, SettingsDep
from ..schemas.api.health import HealthResponse, ServiceStatus
from ..services.openai_llm.client import OpenAILLMClient

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

router = APIRouter()


@router.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse, tags=["Health"])
async def health_check(request: Request, settings: SettingsDep) -> HealthResponse:
    """
    Health check endpoint.

    Returns 200 OK with status="starting" while background services are still
    initializing (prevents Render from killing the service during the slow startup window).
    Returns full service status once everything is ready.
    """
    logger.debug("ENTRY: /api/v1/health")

    # --- Fast path: return 200 "starting" during background initialization ---
    # Render's health checker will keep the service alive as long as it gets 200.
    # Returning 503/crashing here during the ~3-min init window causes Render to
    # restart the service in a loop, preventing it from ever becoming ready.
    agentic_ready = getattr(request.app.state, "agentic_rag_service", None) is not None
    if not agentic_ready:
        return HealthResponse(
            status="starting",
            version=settings.app_version,
            environment=settings.environment,
            service_name=settings.service_name,
            services={
                "startup": ServiceStatus(
                    status="starting",
                    message="Services are initializing in background. Please wait ~2-3 minutes on first boot.",
                )
            },
        )

    # --- Full health check once services are ready ---
    try:
        services = {}
        overall_status = "ok"

        def _check_service(name: str, check_func):
            nonlocal overall_status
            try:
                logger.debug(f"ENTRY: health check for {name}")
                result = check_func()
                services[name] = result
                if result.status != "healthy":
                    overall_status = "degraded"
                logger.debug(f"EXIT: health check for {name} - status: {result.status}")
            except Exception as e:
                logger.exception(f"EXCEPTION: health check for {name} failed")
                services[name] = ServiceStatus(status="unhealthy", message=str(e))
                overall_status = "degraded"

        database = getattr(request.app.state, "database", None)

        def _check_database():
            if not database:
                return ServiceStatus(status="unhealthy", message="Database client not initialized yet")
            with database.get_session() as session:
                session.execute(text("SELECT 1"))
            return ServiceStatus(status="healthy", message="Connected successfully")

        if settings.vector_db_provider == "pinecone":
            def _check_pinecone():
                pinecone_client = getattr(request.app.state, "pinecone_client", None)
                if not pinecone_client:
                    return ServiceStatus(status="unhealthy", message="Pinecone client not initialized yet")
                if not pinecone_client.health_check():
                    return ServiceStatus(status="unhealthy", message="Not responding")
                return ServiceStatus(
                    status="healthy",
                    message=f"Index '{pinecone_client.index_name}' connected successfully",
                )
            _check_service("pinecone", _check_pinecone)
        else:
            opensearch_client = getattr(request.app.state, "opensearch_client", None)

            def _check_opensearch():
                if not opensearch_client:
                    return ServiceStatus(status="unhealthy", message="OpenSearch client not initialized yet")
                if not opensearch_client.health_check():
                    return ServiceStatus(status="unhealthy", message="Not responding")
                stats = opensearch_client.get_index_stats()
                return ServiceStatus(
                    status="healthy",
                    message=f"Index '{stats.get('index_name', 'unknown')}' with {stats.get('document_count', 0)} documents",
                )
            _check_service("opensearch", _check_opensearch)

        _check_service("database", _check_database)

        try:
            logger.debug("ENTRY: health check for openai")
            llm_client = OpenAILLMClient(settings)
            openai_health = await llm_client.health_check()
            services["openai"] = ServiceStatus(status=openai_health["status"], message=openai_health["message"])
            if openai_health["status"] != "healthy":
                overall_status = "degraded"
            logger.debug(f"EXIT: health check for openai - status: {openai_health['status']}")
        except Exception as e:
            logger.exception("EXCEPTION: health check for openai failed")
            services["openai"] = ServiceStatus(status="unhealthy", message=str(e))
            overall_status = "degraded"

        logger.debug(f"EXIT: /api/v1/health with overall_status={overall_status}")
        return HealthResponse(
            status=overall_status,
            version=settings.app_version,
            environment=settings.environment,
            service_name=settings.service_name,
            services=services,
        )
    except Exception as e:
        logger.exception("CRITICAL EXCEPTION in /api/v1/health")
        raise
