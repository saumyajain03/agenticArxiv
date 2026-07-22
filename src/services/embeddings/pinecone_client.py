import logging
from typing import Any, Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)


class PineconeClient:
    """Client for Pinecone serverless vector database."""

    def __init__(self, api_key: str, index_name: str, environment: str = "us-east-1"):
        """Initialize Pinecone client and ensure the index exists.

        :param api_key: Pinecone API Key
        :param index_name: Name of the index
        :param environment: AWS Region (e.g. us-east-1)
        """
        if not api_key:
            raise ValueError("Pinecone API key is required but got empty string.")

        self.index_name = index_name
        self.pc = Pinecone(api_key=api_key)

        # Auto-create serverless index if it doesn't exist
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            if index_name not in existing_indexes:
                logger.info(f"Creating serverless Pinecone index '{index_name}' (dim=1024, metric=cosine)...")
                self.pc.create_index(
                    name=index_name, dimension=1024, metric="cosine", spec=ServerlessSpec(cloud="aws", region=environment)
                )
                logger.info(f"✓ Pinecone index '{index_name}' created successfully")
        except Exception as e:
            logger.error(f"Error checking/creating Pinecone index: {e}")

        self.index = self.pc.Index(index_name)
        logger.info(f"Pinecone client initialized for index: {index_name}")

    def health_check(self) -> bool:
        """Check if Pinecone connection is healthy."""
        try:
            # Running a simple describe index stats call
            self.index.describe_index_stats()
            return True
        except Exception as e:
            logger.error(f"Pinecone health check failed: {e}")
            return False

    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upsert embedded document chunks to Pinecone.

        :param vectors: List of dicts matching: {"id": "chunk_id", "values": [...], "metadata": {...}}
        :returns: Response from Pinecone
        """
        try:
            # Bulk upsert
            response = self.index.upsert(vectors=vectors)
            logger.info(f"Successfully upserted {len(vectors)} vectors to Pinecone")
            return response
        except Exception as e:
            logger.error(f"Error upserting vectors to Pinecone: {e}")
            raise

    def query_similarity(
        self,
        vector: List[float],
        top_k: int = 5,
        categories: Optional[List[str]] = None,
        namespace: str = "",
    ) -> List[Dict[str, Any]]:
        """Query Pinecone for top K most similar vectors.

        :param vector: The query embedding vector
        :param top_k: Number of results to return
        :param categories: Filter by arXiv categories
        :param namespace: Pinecone namespace (default: "")
        :returns: List of matching document hits with metadata (OpenSearch-compatible)
        """
        try:
            filter_dict = {}
            if categories:
                filter_dict["category"] = {"$in": categories}

            logger.info("[PINECONE QUERY PAYLOAD]")
            logger.info(f"  Index: {self.index_name}")
            logger.info(f"  Namespace: '{namespace}'")
            logger.info(f"  Top_k: {top_k}")
            logger.info(f"  Vector dim: {len(vector)}")
            logger.info(f"  Include metadata: True")
            logger.info(f"  Filter: {filter_dict if filter_dict else None}")

            response = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None,
                namespace=namespace,
            )

            # If filtering by category returned 0 hits, retry without filter to avoid total failure due to missing metadata key
            if not response.matches and filter_dict:
                logger.warning("[PINECONE] Filter returned 0 hits. Retrying query without category filter...")
                response = self.index.query(
                    vector=vector,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=namespace,
                )

            logger.info(f"[PINECONE RESPONSE] Matches count: {len(response.matches)}")
            for idx, match in enumerate(response.matches):
                meta = match.metadata or {}
                logger.info(
                    f"  Match {idx+1}: ID={match.id}, Score={match.score:.4f}, ArxivID={meta.get('arxiv_id') or meta.get('paper_id')}, Title={meta.get('title', '')[:50]}"
                )

            # Map matches into general RAG hit dictionaries for compatibility
            hits = []
            for match in response.matches:
                meta = match.metadata or {}
                arxiv_id = meta.get("arxiv_id") or meta.get("paper_id") or match.id.split("_chunk_")[0]
                hits.append(
                    {
                        "chunk_id": match.id,
                        "score": match.score,
                        "chunk_text": meta.get("text", "") or meta.get("chunk_text", ""),
                        "arxiv_id": arxiv_id,
                        "title": meta.get("title", ""),
                        "authors": meta.get("authors", ""),
                        "url": meta.get("url") or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                        "section_title": meta.get("section") or meta.get("section_title", ""),
                        "chunk_index": int(meta.get("chunk_index", 0)),
                        "parent_text": meta.get("parent_text", ""),
                    }
                )

            return hits
        except Exception as e:
            logger.error(f"Error querying similarity from Pinecone: {e}", exc_info=True)
            raise
