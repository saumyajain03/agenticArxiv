import uuid
import tempfile
import json
import asyncio
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from src.dependencies import AgenticRAGDep, LangfuseDep
from src.schemas.api.ask import AgenticAskResponse, AskRequest, FeedbackRequest, FeedbackResponse
from src.services.pdf_generator.generator import MarkdownPDFGenerator

logger = logging.getLogger(__name__)

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


@router.get("/ask-stream-logs")
async def ask_stream_logs(
    query: str,
    agentic_rag: AgenticRAGDep,
):
    """Execute Agentic RAG in the background and stream logs in real-time."""
    async def log_generator():
        queue = asyncio.Queue()
        
        class QueueLogHandler(logging.Handler):
            def emit(self, record):
                try:
                    if 'agents' not in record.name:
                        return
                    msg = record.getMessage()
                    # Filter out verbose initialization/utility details
                    if any(x in msg for x in [
                        "Initializing AgenticRAGService", "Model:", "Top-k:", "Hybrid search:",
                        "Max retrieval:", "Guardrail threshold:", "initialized successfully",
                        "Building LangGraph workflow", "Adding nodes to workflow", "Configuring graph edges",
                        "Compiling LangGraph workflow", "Graph compilation successful", "Creating Langfuse trace",
                        "CallbackHandler added", "Generating graph", "Answer length:", "Sources found:",
                        "Retrieval attempts:", "Execution time:", "=================="
                    ]):
                        return
                    
                    formatted_msg = self.format(record)
                    queue.put_nowait(formatted_msg)
                except Exception:
                    pass

        # Format: timestamp - logger_name - LEVEL - message
        handler = QueueLogHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        root_logger = logging.getLogger()
        handler.setLevel(logging.INFO)
        root_logger.addHandler(handler)
        
        # Start agentic RAG task
        task = asyncio.create_task(agentic_rag.ask(query=query))
        
        try:
            yield f"data: {json.dumps({'log': 'System - INFO - Agent connection initialized. Starting research pipeline...'})}\n\n"
            
            while not task.done() or not queue.empty():
                try:
                    log_msg = await asyncio.wait_for(queue.get(), timeout=0.2)
                    yield f"data: {json.dumps({'log': log_msg})}\n\n"
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break

            result = await task
            yield f"data: {json.dumps({'result': result, 'done': True})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in agent log stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            root_logger.removeHandler(handler)

    return StreamingResponse(log_generator(), media_type="text/event-stream")


