import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.tools import tool

from src.services.embeddings.hyde import HydeService
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient
from src.services.reranker.cohere_client import RerankerService

logger = logging.getLogger(__name__)


def create_retriever_tool(
    opensearch_client: OpenSearchClient,
    embeddings_client: JinaEmbeddingsClient,
    top_k: int = 3,
    use_hybrid: bool = True,
    hyde_service: Optional[HydeService] = None,
    reranker_service: Optional[RerankerService] = None,
    model_name: str = "google/gemini-2.5-flash",
):
    """Create a retriever tool that wraps OpenSearch service with HyDE and Reranking support.

    :param opensearch_client: Existing OpenSearch service
    :param embeddings_client: Existing Jina embeddings service
    :param top_k: Number of chunks to retrieve
    :param use_hybrid: Use hybrid search (BM25 + vector)
    :param hyde_service: Optional HyDE embedding generator service
    :param reranker_service: Optional Cross-Encoder reranker service
    :param model_name: Name of the model for HyDE document generation
    :returns: LangChain tool for retrieving papers
    """

    @tool
    async def retrieve_papers(query: str) -> list[Document]:
        """Search and return relevant arXiv research papers.

        Use this tool when the user asks about:
        - Machine learning concepts or techniques
        - Deep learning architectures
        - Natural language processing
        - Computer vision methods
        - AI research topics
        - Specific algorithms or models

        :param query: The search query describing what papers to find
        :returns: List of relevant paper excerpts with metadata
        """
        logger.info(f"Retrieving papers for query: {query[:100]}...")

        # 1. Generate query embedding (using HyDE if enabled)
        if hyde_service:
            logger.info("Using HyDE for query embedding generation")
            query_embedding = await hyde_service.embed_query_hyde(query, model_name)
        else:
            logger.debug("Generating standard query embedding")
            query_embedding = await embeddings_client.embed_query(query)

        # 2. Search using OpenSearch
        # Retrieve a larger pool of documents to allow the reranker to choose the best ones
        search_size = top_k * 3 if reranker_service and reranker_service.enabled else top_k
        logger.debug(f"Search mode: {'hybrid' if use_hybrid else 'bm25'}, base pool size: {search_size}")

        search_results = opensearch_client.search_unified(
            query=query,
            query_embedding=query_embedding,
            size=search_size,
            use_hybrid=use_hybrid,
        )

        hits = search_results.get("hits", [])
        logger.info(f"Found {len(hits)} raw chunks from OpenSearch")

        # 3. Rerank retrieved chunks
        if reranker_service and reranker_service.enabled:
            logger.info(f"Applying Cross-Encoder reranking on {len(hits)} chunks")
            hits = reranker_service.rerank(query, hits, top_k=top_k)
        else:
            hits = hits[:top_k]

        # 4. Convert hits to LangChain Documents
        documents = []
        for hit in hits:
            # Parent-Child retrieval mapping:
            # If parent_text is present in the source index, return the full parent context.
            # Otherwise, fall back to the standard chunk_text.
            page_content = hit.get("parent_text") or hit.get("chunk_text") or ""
            is_parent = "parent_text" in hit and bool(hit["parent_text"])

            doc = Document(
                page_content=page_content,
                metadata={
                    "arxiv_id": hit["arxiv_id"],
                    "title": hit.get("title", ""),
                    "authors": hit.get("authors", ""),
                    "score": hit.get("score", 0.0),
                    "rerank_score": hit.get("rerank_score", 0.0),
                    "source": f"https://arxiv.org/pdf/{hit['arxiv_id']}.pdf",
                    "section": hit.get("section_title", hit.get("section_name", "")),
                    "search_mode": "hybrid" if use_hybrid else "bm25",
                    "is_parent_context": is_parent,
                    "top_k": top_k,
                },
            )
            documents.append(doc)

        logger.debug(f"Converted {len(documents)} hits to LangChain Documents")
        logger.info(f"✓ Retrieved {len(documents)} papers successfully")

        return documents

    return retrieve_papers
