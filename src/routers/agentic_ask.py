import uuid
import tempfile
import json
import asyncio
import logging
import time
import traceback
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from src.dependencies import AgenticRAGDep, LangfuseDep
from src.schemas.api.ask import AgenticAskResponse, AskRequest, FeedbackRequest, FeedbackResponse
from src.services.pdf_generator.generator import MarkdownPDFGenerator

logger = logging.getLogger(__name__)


def _safe_json_dumps(obj) -> str:
    """
    json.dumps with a fallback for non-serializable types.
    If any value cannot be serialised, it is replaced with str(value).
    A silent TypeError here would close the SSE stream without sending an
    error event, appearing to the browser as a plain connection drop.
    """
    def _default(o):
        try:
            # Pydantic v1/v2 models
            if hasattr(o, 'model_dump'):
                return o.model_dump()
            if hasattr(o, 'dict'):
                return o.dict()
        except Exception:
            pass
        return str(o)
    return json.dumps(obj, default=_default)

router = APIRouter(prefix="/api/v1", tags=["agentic-rag"])


class ExportPDFRequest(BaseModel):
    query: str
    answer: str


@router.post("/ask-agentic", response_model=AgenticAskResponse)
async def ask_agentic(
    request: AskRequest,
    agentic_rag: AgenticRAGDep,
) -> AgenticAskResponse:
    """
    Agentic RAG endpoint with intelligent retrieval and query refinement.

    Features:
    - Decides if retrieval is needed
    - Grades document relevance
    - Rewrites queries if needed
    - Provides reasoning transparency

    The agent will automatically:
    1. Determine if the question requires research paper retrieval
    2. If needed, search for relevant papers
    3. Grade retrieved documents for relevance
    4. Rewrite the query if documents aren't relevant
    5. Generate an answer with citations

    Args:
        request: Question and parameters
        agentic_rag: Injected agentic RAG service

    Returns:
        Answer with sources and reasoning steps

    Raises:
        HTTPException: If processing fails
    """
    try:
        result = await agentic_rag.ask(
            query=request.query,
            model=request.model,
        )

        return AgenticAskResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result.get("sources", []),
            chunks_used=request.top_k,
            search_mode="hybrid" if request.use_hybrid else "bm25",
            reasoning_steps=result.get("reasoning_steps", []),
            retrieval_attempts=result.get("retrieval_attempts", 0),
            rewritten_query=result.get("rewritten_query"),
            trace_id=result.get("trace_id"),
            guardrail_filter=result.get("guardrail_filter"),
            output_guardrail_filter=result.get("output_guardrail_filter"),
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    langfuse_tracer: LangfuseDep,
) -> FeedbackResponse:
    """
    Submit user feedback for an agentic RAG response.

    This endpoint allows users to rate the quality of answers and provide
    optional comments. Feedback is tracked in Langfuse for continuous improvement.

    Args:
        request: Feedback data including trace_id, score, and optional comment
        langfuse_tracer: Injected Langfuse tracer service

    Returns:
        FeedbackResponse indicating success or failure

    Raises:
        HTTPException: If feedback submission fails
    """
    try:
        if not langfuse_tracer:
            raise HTTPException(status_code=503, detail="Langfuse tracing is disabled. Cannot submit feedback.")

        success = langfuse_tracer.submit_feedback(
            trace_id=request.trace_id,
            score=request.score,
            comment=request.comment,
        )

        if success:
            # Flush to ensure feedback is sent immediately
            langfuse_tracer.flush()

            return FeedbackResponse(success=True, message="Feedback recorded successfully")
        else:
            raise HTTPException(status_code=500, detail="Failed to submit feedback to Langfuse")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")


@router.post("/export-pdf")
async def export_pdf(request: ExportPDFRequest) -> FileResponse:
    """Export RAG answer to a formatted PDF report."""
    try:
        generator = MarkdownPDFGenerator()
        temp_dir = Path(tempfile.gettempdir())
        pdf_path = temp_dir / f"arxiv_rag_report_{uuid.uuid4().hex}.pdf"

        generator.generate_pdf(
            query=request.query,
            answer_markdown=request.answer,
            output_path=pdf_path
        )

        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename="arxiv_rag_report.pdf"
        )
    except Exception as e:
        logger.error(f"Failed to generate PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")


# ---------------------------------------------------------------------------
# Fully instrumented streaming endpoint
# Every stage is logged with a [STAGE] prefix so you can grep Render logs
# and immediately see which stage is reached / missing / slow.
# ---------------------------------------------------------------------------

@router.get("/ask-stream-logs")
async def ask_stream_logs(
    query: str,
    http_request: Request,
    agentic_rag: AgenticRAGDep,
):
    """
    Execute Agentic RAG in the background and stream logs in real-time.

    SSE event shapes:
      {"log": "<message>"}              — pipeline stage log line
      {"result": {...}, "done": true}   — final answer payload
      {"error": "<msg>", "traceback": "..."}  — fatal error with full tb

    NOTE: FastAPI resolves all Depends() BEFORE this function body runs.
    If any dependency raises (e.g. AgenticRAGService still initialising),
    FastAPI returns a non-SSE JSON error response and log_generator() is
    never called — no [STAGE] ENTRY log appears.
    """
    # Log entry immediately so we know the route handler was reached at all.
    # If this line never appears in Render logs the request is dying during
    # dependency injection — before this function body executes.
    req_id = str(uuid.uuid4())[:8]
    logger.info("[STAGE] ENTRY ask_stream_logs (route body) | req=%s | query=%r", req_id, query[:80])

    async def log_generator():
        # ── STAGE 0: endpoint entered ────────────────────────────────────────
        t0 = time.perf_counter()
        logger.info("[STAGE] ENTRY ask_stream_logs | req=%s | query=%r", req_id, query[:80])
        yield f"data: {json.dumps({'log': f'[{req_id}] ENTRY ask_stream_logs'})}\n\n"

        queue: asyncio.Queue = asyncio.Queue()

        # ── Log capture: forward agent logs to SSE ───────────────────────────
        class QueueLogHandler(logging.Handler):
            def emit(self, record):
                try:
                    msg = record.getMessage()
                    # Always forward STAGE and PIPELINE lines
                    if "[STAGE]" in msg or "[PIPELINE]" in msg:
                        queue.put_nowait(self.format(record))
                        return
                    # Forward all agent-namespace logs
                    if "agents" not in record.name and "nodes" not in record.name:
                        return
                    # Drop noisy init chatter
                    if any(x in msg for x in [
                        "Initializing AgenticRAGService", "Model:", "Top-k:", "Hybrid search:",
                        "Max retrieval:", "Guardrail threshold:", "initialized successfully",
                        "Building LangGraph workflow", "Adding nodes to workflow",
                        "Configuring graph edges", "Compiling LangGraph workflow",
                        "Graph compilation successful", "CallbackHandler added",
                        "Generating graph", "==================",
                    ]):
                        return
                    queue.put_nowait(self.format(record))
                except Exception:
                    pass

        handler = QueueLogHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s.%(msecs)03d [%(name)s] %(levelname)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        handler.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        # ── STAGE 1: request validated ───────────────────────────────────────
        logger.info("[STAGE] Request validated | req=%s", req_id)
        yield f"data: {json.dumps({'log': f'[{req_id}] Request validated'})}\n\n"

        # ── STAGE 2: agent instance confirmed ───────────────────────────────
        logger.info("[STAGE] Agent initialized (singleton) | req=%s", req_id)
        yield f"data: {json.dumps({'log': f'[{req_id}] Agent initialized'})}\n\n"

        # ── STAGE 3: launch background task ─────────────────────────────────
        logger.info("[STAGE] Launching background pipeline task | req=%s", req_id)
        yield f"data: {json.dumps({'log': f'[{req_id}] Pipeline task launched'})}\n\n"

        task = asyncio.create_task(_run_instrumented_pipeline(agentic_rag, query, req_id))

        # ── Drain queue until task done; detect frontend disconnect ──────────
        backend_terminated = False
        try:
            while not task.done() or not queue.empty():
                # Detect frontend disconnect
                if await http_request.is_disconnected():
                    elapsed = (time.perf_counter() - t0) * 1000
                    logger.warning(
                        "[STAGE] CLIENT DISCONNECTED — stream terminated by FRONTEND | req=%s | elapsed=%.0fms",
                        req_id, elapsed,
                    )
                    task.cancel()
                    yield f"data: {json.dumps({'log': f'[{req_id}] ⚠ FRONTEND DISCONNECTED after {elapsed:.0f}ms'})}\n\n"
                    return

                try:
                    log_msg = await asyncio.wait_for(queue.get(), timeout=0.3)
                    yield f"data: {json.dumps({'log': log_msg})}\n\n"
                except asyncio.TimeoutError:
                    # SSE heartbeat comment — keeps Render's proxy from closing the TCP connection
                    yield ": heartbeat\n\n"
                    continue
                except Exception:
                    break

            # ── Task finished — emit result or exception ──────────────────────
            backend_terminated = True
            elapsed_ms = (time.perf_counter() - t0) * 1000

            try:
                result = task.result()
                logger.info(
                    "[STAGE] Stream completed by BACKEND | req=%s | total=%.0fms | answer_len=%d",
                    req_id, elapsed_ms, len(result.get("answer", "")),
                )
                yield f"data: {json.dumps({'log': f'[{req_id}] ✓ STREAM COMPLETE — {elapsed_ms:.0f}ms'})}\n\n"
                # Use _safe_json_dumps: if result contains a non-serialisable
                # object (Pydantic model, dataclass, datetime, etc.) a plain
                # json.dumps() call would raise TypeError inside the generator,
                # closing the SSE stream with no error event — the browser sees
                # only a connection drop and fires onerror.
                try:
                    payload = _safe_json_dumps({'result': result, 'done': True})
                except Exception as serial_exc:
                    tb = traceback.format_exc()
                    logger.error(
                        "[STAGE] SERIALISATION ERROR | req=%s | error=%s\n%s",
                        req_id, repr(serial_exc), tb,
                    )
                    payload = json.dumps({'error': f'Result serialisation failed: {serial_exc}', 'traceback': tb})
                yield f"data: {payload}\n\n"

            except asyncio.CancelledError:
                logger.warning("[STAGE] Task cancelled | req=%s", req_id)
                yield f"data: {json.dumps({'error': 'Task was cancelled'})}\n\n"

            except Exception as exc:
                tb = traceback.format_exc()
                logger.error(
                    "[STAGE] TASK EXCEPTION | req=%s | error=%s\n%s",
                    req_id, repr(exc), tb,
                )
                yield f"data: {json.dumps({'error': str(exc), 'traceback': tb})}\n\n"

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("[STAGE] GENERATOR EXCEPTION | req=%s\n%s", req_id, tb)
            yield f"data: {json.dumps({'error': f'Generator error: {str(exc)}', 'traceback': tb})}\n\n"

        finally:
            root_logger.removeHandler(handler)
            terminated_by = "BACKEND" if backend_terminated else "FRONTEND"
            logger.info(
                "[STAGE] Generator exited | req=%s | terminated_by=%s | elapsed=%.0fms",
                req_id, terminated_by, (time.perf_counter() - t0) * 1000,
            )

    return StreamingResponse(log_generator(), media_type="text/event-stream")


async def _run_instrumented_pipeline(agentic_rag, query: str, req_id: str) -> dict:
    """
    Wraps agentic_rag.ask() with per-node timing using a NodeWatcher that
    detects the 'NODE: <name>' sentinel log each node already emits.
    """
    node_timings: dict = {}
    _current: dict = {"name": None, "start": None}
    pipeline_start = time.perf_counter()

    class NodeWatcher(logging.Handler):
        NODE_PREFIX = "NODE: "

        def emit(self, record):
            msg = record.getMessage()
            if not msg.startswith(self.NODE_PREFIX):
                return
            node_name = msg[len(self.NODE_PREFIX):].strip()
            now = time.perf_counter()

            # Close the previous node
            prev = _current["name"]
            if prev and _current["start"]:
                ms = (now - _current["start"]) * 1000
                node_timings[prev] = ms
                logger.info(
                    "[STAGE] NODE %-25s FINISHED ✓ | stage=%.0fms | elapsed=%.0fms | req=%s",
                    prev.upper(), ms, (now - pipeline_start) * 1000, req_id,
                )

            # Open the new node
            _current["name"] = node_name
            _current["start"] = now
            logger.info(
                "[STAGE] NODE %-25s STARTED   | elapsed=%.0fms | req=%s",
                node_name.upper(), (now - pipeline_start) * 1000, req_id,
            )

    watcher = NodeWatcher()
    watcher.setLevel(logging.INFO)
    node_loggers = [
        "src.services.agents.nodes.guardrail_node",
        "src.services.agents.nodes.supervisor_node",
        "src.services.agents.nodes.researcher_node",
        "src.services.agents.nodes.writer_node",
        "src.services.agents.nodes.critic_node",
        "src.services.agents.nodes.generate_answer_node",
        "src.services.agents.nodes.output_guardrail_node",
        "src.services.agents.nodes.retrieve_node",
        "src.services.agents.nodes.rewrite_query_node",
    ]
    for mod in node_loggers:
        logging.getLogger(mod).addHandler(watcher)

    try:
        logger.info("[STAGE] Retriever started | req=%s", req_id)
        result = await agentic_rag.ask(query=query)
        logger.info("[STAGE] Pipeline finished | req=%s", req_id)

        # Close last node
        now = time.perf_counter()
        prev = _current["name"]
        if prev and _current["start"]:
            ms = (now - _current["start"]) * 1000
            node_timings[prev] = ms
            logger.info(
                "[STAGE] NODE %-25s FINISHED ✓ | stage=%.0fms | req=%s",
                prev.upper(), ms, req_id,
            )

        # Timing summary
        total_ms = (now - pipeline_start) * 1000
        logger.info("[STAGE] ══════ TIMING SUMMARY req=%s ══════", req_id)
        for node, ms in node_timings.items():
            pct = ms / total_ms * 100 if total_ms else 0
            logger.info("[STAGE]   %-30s %7.0f ms  (%4.1f%%)", node, ms, pct)
        logger.info("[STAGE]   %-30s %7.0f ms", "TOTAL", total_ms)
        logger.info("[STAGE] ═══════════════════════════════════════", )

        return result

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(
            "[STAGE] PIPELINE EXCEPTION | req=%s | error=%s\nFULL TRACEBACK:\n%s",
            req_id, repr(exc), tb,
        )
        raise

    finally:
        for mod in node_loggers:
            logging.getLogger(mod).removeHandler(watcher)
