import json
import logging
import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.dependencies import AgenticRAGDep, CacheDep, EmbeddingsDep, LangfuseDep, LLMDep, OpenSearchDep
from src.schemas.api.ask import AskRequest, AskResponse
from src.services.langfuse.tracer import RAGTracer

logger = logging.getLogger(__name__)

# Two separate routers - one for regular ask, one for streaming
ask_router = APIRouter(tags=["ask"])
stream_router = APIRouter(tags=["stream"])


async def _prepare_chunks_and_sources(
    request: AskRequest,
    opensearch_client,
    embeddings_service,
    rag_tracer: RAGTracer,
    trace=None,
) -> tuple[List[Dict], List[str], List[str]]:
    """Retrieve and prepare chunks for RAG with clean tracing."""

    # Handle embeddings for hybrid search
    query_embedding = None
    if request.use_hybrid:
        with rag_tracer.trace_embedding(trace, request.query) as embedding_span:
            try:
                query_embedding = await embeddings_service.embed_query(request.query)
                logger.info("Generated query embedding for hybrid search")
            except Exception as e:
                logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")
                if embedding_span:
                    rag_tracer.tracer.update_span(embedding_span, output={"success": False, "error": str(e)})

    # Search with tracing
    with rag_tracer.trace_search(trace, request.query, request.top_k) as search_span:
        search_results = opensearch_client.search_unified(
            query=request.query,
            query_embedding=query_embedding,
            size=request.top_k,
            from_=0,
            categories=request.categories,
            use_hybrid=request.use_hybrid and query_embedding is not None,
            min_score=0.0,
        )

        # Extract essential data for LLM
        chunks = []
        arxiv_ids = []
        sources_set = set()

        for hit in search_results.get("hits", []):
            arxiv_id = hit.get("arxiv_id", "")

            # Minimal chunk data for LLM
            chunks.append(
                {
                    "arxiv_id": arxiv_id,
                    "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
                }
            )

            if arxiv_id:
                arxiv_ids.append(arxiv_id)
                arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                sources_set.add(f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf")

        # End search span with essential metadata
        rag_tracer.end_search(search_span, chunks, arxiv_ids, search_results.get("total", 0))

    return chunks, list(sources_set), arxiv_ids


@ask_router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    http_request: Request,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    llm_client: LLMDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
    agentic_rag: AgenticRAGDep,
) -> AskResponse:
    """Clean RAG endpoint utilizing Agentic RAG service pipeline and caching."""
    start_time = time.time()

    try:
        # Check semantic/exact cache first
        if cache_client:
            try:
                cached_response = await cache_client.find_cached_response(request)
                if cached_response:
                    logger.info("Returning cached response for query match")
                    return cached_response
            except Exception as e:
                logger.warning(f"Cache check failed: {e}")

        # Execute Multi-Agent RAG pipeline
        result = await agentic_rag.ask(
            query=request.query,
            model=request.model,
        )

        answer = result["answer"]
        sources_raw = result.get("sources", [])

        # Format source URLs
        sources = []
        for src in sources_raw:
            arxiv_id = src.get("arxiv_id")
            if arxiv_id:
                sources.append(f"https://arxiv.org/pdf/{arxiv_id}.pdf")

        # Prepare response
        response = AskResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            chunks_used=len(sources),
            search_mode="agentic",
        )

        # Store response in cache
        if cache_client:
            try:
                await cache_client.store_response(request, response)
            except Exception as e:
                logger.warning(f"Failed to store response in cache: {e}")

        return response

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@stream_router.post("/stream")
async def ask_question_stream(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    llm_client: LLMDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
    agentic_rag: AgenticRAGDep,
) -> StreamingResponse:
    """Clean streaming RAG endpoint using Agentic RAG service pipeline and caching."""

    async def generate_stream():
        import asyncio

        start_time = time.time()

        try:
            # Check semantic/exact cache first
            if cache_client:
                try:
                    cached_response = await cache_client.find_cached_response(request)
                    if cached_response:
                        logger.info("Returning cached response for streaming query match")

                        # Send metadata first
                        metadata_response = {
                            "sources": cached_response.sources,
                            "chunks_used": cached_response.chunks_used,
                            "search_mode": cached_response.search_mode,
                        }
                        yield f"data: {json.dumps(metadata_response)}\n\n"

                        # Stream the cached response in chunks
                        for chunk in cached_response.answer.split():
                            yield f"data: {json.dumps({'chunk': chunk + ' '})}\n\n"
                            await asyncio.sleep(0.01)

                        # Send completion signal
                        yield f"data: {json.dumps({'answer': cached_response.answer, 'done': True})}\n\n"
                        return
                except Exception as e:
                    logger.warning(f"Cache check failed: {e}")

            # Execute Multi-Agent RAG pipeline with true streaming
            full_answer = ""
            try:
                logger.debug("ENTRY: agentic_rag.astream")
                async for event in agentic_rag.astream(
                    query=request.query,
                    model=request.model,
                ):
                    event_type = event.get("type")
                    if event_type == "heartbeat":
                        # Send an empty chunk as a keep-alive signal that the frontend can safely append
                        yield f"data: {json.dumps({'chunk': ''})}\n\n"
                    elif event_type == "chunk":
                        # Yield the actual generated text chunk
                        chunk_data = event.get("data", "")
                        full_answer += chunk_data
                        yield f"data: {json.dumps({'chunk': chunk_data})}\n\n"
                    elif event_type == "metadata":
                        # Yield the final metadata (sources, attempts)
                        metadata_data = event.get("data", {})
                        sources_raw = metadata_data.get("sources", [])
                        
                        sources = []
                        for src in sources_raw:
                            arxiv_id = src.get("arxiv_id")
                            if arxiv_id:
                                sources.append(f"https://arxiv.org/pdf/{arxiv_id}.pdf")
                        
                        metadata_response = {
                            "sources": sources,
                            "chunks_used": len(sources),
                            "search_mode": "agentic",
                        }
                        yield f"data: {json.dumps(metadata_response)}\n\n"
                    elif event_type == "error":
                        error_msg = event.get("data", "Unknown error")
                        logger.error(f"Error in stream: {error_msg}")
                        yield f"data: {json.dumps({'error': error_msg})}\n\n"
                        
                logger.debug("EXIT: agentic_rag.astream success")
            except Exception as e:
                logger.exception("EXCEPTION in agentic_rag.astream")
                raise e

            # Send completion signal
            yield f"data: {json.dumps({'answer': full_answer, 'done': True})}\n\n"

            # Store response in cache
            if cache_client and full_answer:
                try:
                    response_to_cache = AskResponse(
                        query=request.query,
                        answer=full_answer,
                        sources=sources,
                        chunks_used=len(sources),
                        search_mode="agentic",
                    )
                    await cache_client.store_response(request, response_to_cache)
                except Exception as e:
                    logger.warning(f"Failed to store streaming response in cache: {e}")

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(), media_type="text/plain", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
