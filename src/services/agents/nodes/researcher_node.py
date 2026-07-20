import asyncio
import logging
from typing import Any, Dict, List, Union

import logfire
from langgraph.runtime import Runtime

from src.services.embeddings.hyde import HydeService
from src.services.reranker.cohere_client import RerankerService

from ..context import Context
from ..state import AgentState

logger = logging.getLogger(__name__)


@logfire.instrument("node:researcher", extract_args=False)
async def ainvoke_researcher_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, Union[dict, list]]:
    """Section Researcher worker node.

    Executes parallel search queries for each section in the research plan.
    Incorporates HyDE, Parent-Child retrieval, and Cross-Encoder Reranking.

    :param state: Current agent state
    :param runtime: Runtime context containing dependencies
    :returns: Updated state dictionary with retrieved documents grouped by section
    """
    logger.info("NODE: researcher")
    research_plan = state.get("research_plan")

    if not research_plan:
        logger.warning("No research plan found in state. Skipping researcher node.")
        return {}

    # Initialize services
    hyde_service = HydeService(runtime.context.llm_client, runtime.context.embeddings_client)
    reranker_service = RerankerService(model_name="BAAI/bge-reranker-base", enabled=True)

    # Concurrency Semaphore to prevent rate limits
    semaphore = asyncio.Semaphore(3)

    async def search_query(query: str) -> List[Dict[str, Any]]:
        """Safely execute semantic retrieval for a single query using HyDE + Reranking."""
        async with semaphore:
            try:
                # 1. HyDE Embeddings
                query_embedding = await hyde_service.embed_query_hyde(query, runtime.context.model_name)

                # 2. Search (BM25 + Vector or Pinecone)
                search_size = runtime.context.top_k * 3
                if runtime.context.opensearch_client:
                    logger.debug(f"Searching OpenSearch with pool size: {search_size}")
                    search_results = runtime.context.opensearch_client.search_unified(
                        query=query,
                        query_embedding=query_embedding,
                        size=search_size,
                        use_hybrid=runtime.context.use_hybrid,
                    )
                    hits = search_results.get("hits", [])
                elif runtime.context.pinecone_client:
                    logger.debug(f"Searching Pinecone Cloud with pool size: {search_size}")
                    hits = runtime.context.pinecone_client.query_similarity(
                        vector=query_embedding,
                        top_k=search_size,
                    )
                else:
                    logger.warning("No search backend client configured in LangGraph Context.")
                    hits = []

                # 3. Rerank Chunks
                reranked_hits = reranker_service.rerank(query, hits, top_k=runtime.context.top_k)
                return reranked_hits
            except Exception as e:
                logger.error(f"Failed to execute search for query '{query}': {e}")
                return []

    async def research_section(section: Dict[str, Any]) -> Dict[str, Any]:
        """Perform search queries and aggregate results for a specific section."""
        title = section.get("title", "")
        queries = section.get("queries", [])
        logger.info(f"Researching section: '{title}' with queries: {queries}")

        # Run section queries concurrently
        tasks = [search_query(q) for q in queries]
        query_results = await asyncio.gather(*tasks)

        # Merge results, deduplicating by arxiv_id
        seen_arxiv_ids = set()
        merged_hits = []

        for hits in query_results:
            for hit in hits:
                arxiv_id = hit.get("arxiv_id")
                if arxiv_id and arxiv_id not in seen_arxiv_ids:
                    seen_arxiv_ids.add(arxiv_id)
                    merged_hits.append(hit)

        logger.info(f"Section '{title}' research complete: retrieved {len(merged_hits)} unique papers")
        return {"section_title": title, "hits": merged_hits}

    # Run research concurrently across all outline sections
    section_tasks = [research_section(sec) for sec in research_plan]
    section_results = await asyncio.gather(*section_tasks)

    # Convert results to state structure
    retrieved_documents = {}
    all_sources = []

    for res in section_results:
        title = res["section_title"]
        hits = res["hits"]
        retrieved_documents[title] = hits
        all_sources.extend(hits)

    # Deduplicate global sources list for metadata output
    seen_global = set()
    deduped_sources = []
    for hit in all_sources:
        arxiv_id = hit.get("arxiv_id")
        if arxiv_id and arxiv_id not in seen_global:
            seen_global.add(arxiv_id)
            # Format to SourceItem equivalent
            deduped_sources.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": hit.get("title", "Unknown"),
                    "authors": hit.get("authors", "Unknown"),
                    "section": hit.get("section_title", "General"),
                }
            )

    return {
        "metadata": {**state.get("metadata", {}), "retrieved_documents": retrieved_documents},
        "relevant_sources": deduped_sources,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
