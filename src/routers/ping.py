import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import Response
from sqlalchemy import text

from ..dependencies import SettingsDep
from ..schemas.api.health import HealthResponse, ServiceStatus
from ..services.openai_llm.client import OpenAILLMClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

router = APIRouter()


# ---------------------------------------------------------------------------
# /health/live  — pure liveness probe, no external calls
# Render can hit this every few seconds without risk of blocking on DB/LLM.
# ---------------------------------------------------------------------------

@router.get("/health/live", tags=["Health"])
async def liveness() -> dict:
    """Liveness probe — returns immediately without touching any external service."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Helper: run a synchronous health-check function safely with a timeout
# ---------------------------------------------------------------------------

async def safe_health_check(fn, timeout: float = 3.0) -> bool:
    """
    Run a synchronous health-check callable in a thread with a hard timeout.

    :param fn: Zero-argument callable that returns bool.
    :param timeout: Seconds before we give up and return False.
    :returns: True if the check passed within the timeout, False otherwise.
    """
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("Health check timed out after %.1fs", timeout)
        return False
    except Exception as e:
        logger.error("Health check error: %s", e)
        return False


# ---------------------------------------------------------------------------
# /health  — full readiness probe
# Returns 200 "starting" during the background init window so Render does not
# restart the service before it has had a chance to finish initialising.
# ---------------------------------------------------------------------------

@router.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse, tags=["Health"])
async def health_check(request: Request, settings: SettingsDep) -> HealthResponse:
    """
    Readiness health check.

    Returns 200 OK with status="starting" while background services are still
    initialising (prevents Render from killing the service during the slow
    startup window).  Returns full service status once everything is ready.
    """
    logger.debug("ENTRY: /api/v1/health")

    # Fast path: still booting up
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
                    message="Services are initialising in background. Please wait ~2-3 minutes on first boot.",
                )
            },
        )

    # Full check
    try:
        services: dict = {}
        overall_status = "ok"

        def _set(name: str, result: ServiceStatus) -> None:
            nonlocal overall_status
            services[name] = result
            if result.status != "healthy":
                overall_status = "degraded"

        # ── Database ─────────────────────────────────────────────────────────
        database = getattr(request.app.state, "database", None)

        def _db_fn() -> ServiceStatus:
            if not database:
                return ServiceStatus(status="unhealthy", message="Database client not initialised yet")
            with database.get_session() as session:
                session.execute(text("SELECT 1"))
            return ServiceStatus(status="healthy", message="Connected successfully")

        try:
            _set("database", await asyncio.wait_for(asyncio.to_thread(_db_fn), timeout=3.0))
        except asyncio.TimeoutError:
            _set("database", ServiceStatus(status="unhealthy", message="Timed out after 3s"))
        except Exception as e:
            logger.exception("Database health check failed")
            _set("database", ServiceStatus(status="unhealthy", message=str(e)))

        # ── Vector DB (Pinecone or OpenSearch) ───────────────────────────────
        if settings.vector_db_provider == "pinecone":
            pinecone_client = getattr(request.app.state, "pinecone_client", None)

            async def _check_pinecone() -> ServiceStatus:
                if not pinecone_client:
                    return ServiceStatus(status="unhealthy", message="Pinecone client not initialised yet")
                ok = await safe_health_check(pinecone_client.health_check, timeout=3.0)
                if not ok:
                    return ServiceStatus(status="unhealthy", message="Not responding or timed out")
                return ServiceStatus(
                    status="healthy",
                    message=f"Index '{pinecone_client.index_name}' connected successfully",
                )

            try:
                _set("pinecone", await _check_pinecone())
            except Exception as e:
                logger.exception("Pinecone health check failed")
                _set("pinecone", ServiceStatus(status="unhealthy", message=str(e)))

        else:
            opensearch_client = getattr(request.app.state, "opensearch_client", None)

            async def _check_opensearch() -> ServiceStatus:
                if not opensearch_client:
                    return ServiceStatus(status="unhealthy", message="OpenSearch client not initialised yet")
                ok = await safe_health_check(opensearch_client.health_check, timeout=3.0)
                if not ok:
                    return ServiceStatus(status="unhealthy", message="Not responding or timed out")
                stats = opensearch_client.get_index_stats()
                return ServiceStatus(
                    status="healthy",
                    message=f"Index '{stats.get('index_name', 'unknown')}' with {stats.get('document_count', 0)} documents",
                )

            try:
                _set("opensearch", await _check_opensearch())
            except Exception as e:
                logger.exception("OpenSearch health check failed")
                _set("opensearch", ServiceStatus(status="unhealthy", message=str(e)))

        # ── LLM ──────────────────────────────────────────────────────────────
        try:
            logger.debug("ENTRY: health check for openai")
            llm_client = OpenAILLMClient(settings)
            openai_health = await asyncio.wait_for(llm_client.health_check(), timeout=5.0)
            _set("openai", ServiceStatus(status=openai_health["status"], message=openai_health["message"]))
            logger.debug("EXIT: health check for openai — status: %s", openai_health["status"])
        except asyncio.TimeoutError:
            logger.warning("LLM health check timed out")
            _set("openai", ServiceStatus(status="unhealthy", message="Timed out after 5s"))
        except Exception as e:
            logger.exception("LLM health check failed")
            _set("openai", ServiceStatus(status="unhealthy", message=str(e)))

        logger.debug("EXIT: /api/v1/health — overall_status=%s", overall_status)
        return HealthResponse(
            status=overall_status,
            version=settings.app_version,
            environment=settings.environment,
            service_name=settings.service_name,
            services=services,
        )

    except Exception:
        logger.exception("CRITICAL EXCEPTION in /api/v1/health")
        raise
