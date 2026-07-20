import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RerankerService:
    """Service to rerank retrieved documents using a Cross-Encoder model.

    This ensures that the most semantically relevant documents are placed at the
    top of the context list before feeding them to the LLM.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", enabled: bool = True):
        """Initialize the reranking service.

        :param model_name: The local Cross-Encoder model to load
        :param enabled: Set to False to bypass reranking
        """
        self.enabled = enabled
        self._model = None
        self.model_name = model_name

        if self.enabled:
            logger.info(f"Initializing local Cross-Encoder reranker with model: {model_name}...")
            # Lazy loading of model to avoid slowing down startup time
        else:
            logger.info("Reranking is disabled. Documents will remain sorted by search index score.")

    def _get_model(self):
        """Lazy load the CrossEncoder model."""
        if self._model is None and self.enabled:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(f"Loading sentence-transformers CrossEncoder: {self.model_name}...")
                self._model = CrossEncoder(self.model_name)
                logger.info("✓ CrossEncoder model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load local CrossEncoder model: {e}. Reranker will be disabled.")
                self.enabled = False
        return self._model

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Rerank a list of retrieved document chunks against the search query.

        :param query: The search query string
        :param documents: List of document dictionaries (from OpenSearch hits)
        :param top_k: Number of highest-scored documents to return
        :returns: List of reranked and sliced document dictionaries
        """
        if not self.enabled or not documents:
            logger.info(f"Reranking skipped (enabled={self.enabled}, documents_count={len(documents)}). Returning top_k={top_k}")
            return documents[:top_k]

        model = self._get_model()
        if model is None:
            return documents[:top_k]

        logger.info(f"Reranking {len(documents)} documents for query: '{query[:50]}...'")

        try:
            # Prepare pairs of (query, document_text) for the cross-encoder
            # We fetch chunk_text or abstract as the document context
            pairs = []
            for doc in documents:
                doc_text = doc.get("chunk_text") or doc.get("abstract") or ""
                pairs.append([query, doc_text])

            # Predict relevance scores (higher is better)
            scores = model.predict(pairs)

            # Attach scores to the documents
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)

            # Sort documents by rerank score in descending order
            reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

            logger.info(f"✓ Reranking completed. Top score: {reranked_docs[0]['rerank_score']:.4f}")

            # Slice and return top_k
            return reranked_docs[:top_k]

        except Exception as e:
            logger.error(f"Error occurred during document reranking: {e}. Falling back to default order.")
            return documents[:top_k]
