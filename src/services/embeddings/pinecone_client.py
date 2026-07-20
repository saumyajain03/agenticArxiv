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
        self, vector: List[float], top_k: int = 5, categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Query Pinecone for top K most similar vectors.

        :param vector: The query embedding vector
        :param top_k: Number of results to return
        :param categories: Filter by arXiv categories
        :returns: List of matching document hits with metadata (OpenSearch-compatible)
        """
        try:
            # Build filters if categories are specified
            filter_dict = {}
            if categories:
                # Pinecone filter format: {"category": {"$in": categories}}
                filter_dict["category"] = {"$in": categories}

            response = self.index.query(
                vector=vector, top_k=top_k, include_metadata=True, filter=filter_dict if filter_dict else None
            )

            # Map matches into general RAG hit dictionaries for compatibility
            hits = []
            for match in response.matches:
                meta = match.metadata or {}
                # Match OpenSearch flat hit structure for downstream parser compatibility
                hits.append(
                    {
                        "chunk_id": match.id,
                        "score": match.score,
                        "chunk_text": meta.get("text", ""),
                        "arxiv_id": meta.get("paper_id", ""),
                        "title": meta.get("title", ""),
                        "authors": meta.get("authors", ""),
                        "url": meta.get("url", ""),
                        "section_title": meta.get("section", ""),
                        "chunk_index": int(meta.get("chunk_index", 0)),
                        "parent_text": meta.get("parent_text", ""),
                    }
                )

            return hits
        except Exception as e:
            logger.error(f"Error querying similarity from Pinecone: {e}")
            raise
