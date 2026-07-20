import logging
from typing import List, Optional

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.llm_client_protocol import LLMClientProtocol

logger = logging.getLogger(__name__)


class HydeService:
    """Service to generate Hypothetical Document Embeddings (HyDE).

    This converts a user question into a hypothetical scientific abstract/passage
    using an LLM, and then embeds that synthetic document to match academic papers.
    """

    def __init__(self, llm_client: LLMClientProtocol, embeddings_client: JinaEmbeddingsClient):
        """Initialize the HyDE service.

        :param llm_client: Swappable LLM client supporting LLMClientProtocol
        :param embeddings_client: Jina AI embeddings client
        """
        self.llm = llm_client
        self.embeddings = embeddings_client

    async def generate_hypothetical_document(self, query: str, model_name: str) -> str:
        """Generate a hypothetical scientific abstract/passage that answers the query.

        :param query: User search query
        :param model_name: Name of the LLM model to use
        :returns: String containing the generated hypothetical abstract
        """
        logger.info(f"Generating HyDE hypothetical document for query: '{query[:50]}...'")

        prompt = (
            "You are a leading AI researcher writing a peer-reviewed scientific paper. "
            "Write a single, highly detailed, technically accurate paragraph (about 150-200 words) "
            "that directly answers the following search query:\n\n"
            f"Query: {query}\n\n"
            "Format the paragraph as a formal abstract excerpt. Do not write introductory words, "
            "do not say 'Here is an abstract', and do not output any meta-commentary. "
            "Write only the technical snippet containing formal scientific assertions, equations, or terminology."
        )

        try:
            model = self.llm.get_langchain_model(model_name, temperature=0.0)
            response = await model.ainvoke(prompt)
            hypothetical_doc = response.content
            logger.info(f"Successfully generated hypothetical document ({len(hypothetical_doc)} chars)")
            logger.debug(f"Generated text: {hypothetical_doc}")
            return hypothetical_doc
        except Exception as e:
            logger.error(f"Failed to generate hypothetical document: {e}. Falling back to raw query.")
            return query

    async def embed_query_hyde(self, query: str, model_name: str) -> List[float]:
        """Generate a HyDE embedding vector for a search query.

        Generates a hypothetical document, then embeds it using Jina embeddings task 'retrieval.query'.

        :param query: User search query
        :param model_name: Name of the LLM model to use for HyDE generation
        :returns: 1024-dimensional embedding vector
        """
        hypothetical_doc = await self.generate_hypothetical_document(query, model_name)

        # Embed the generated abstract using the same retrieve.query task
        try:
            vector = await self.embeddings.embed_query(hypothetical_doc)
            logger.info("Successfully generated HyDE embedding vector")
            return vector
        except Exception as e:
            logger.error(f"HyDE embedding failed, falling back to raw query embedding: {e}")
            return await self.embeddings.embed_query(query)
