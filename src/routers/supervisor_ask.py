from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from ..schemas.api.ask import AskRequest

if TYPE_CHECKING:
    from ..services.agents import SupervisorAgent, SupervisorResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["supervisor"])


@router.post("/ask-supervisor")
async def ask_supervisor(body: AskRequest, request: Request) -> dict:
    supervisor: "SupervisorAgent" = request.app.state.supervisor_agent
    if not supervisor:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Supervisor agent is still initializing. Please retry in a few seconds.")

    logger.info("Supervisor ask: query_len=%d", len(body.query))

    result = await supervisor.ask(query=body.query)

    return {
        "query": body.query,
        "answer": result.answer,
        "intent": result.intent,
        "routed_to": result.routed_to,
        "sources": result.sources,
    }
