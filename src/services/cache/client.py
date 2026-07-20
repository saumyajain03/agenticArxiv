import hashlib
import json
import logging
from datetime import timedelta
from typing import Optional

import redis
from src.config import RedisSettings
from src.schemas.api.ask import AskRequest, AskResponse
from src.services.embeddings.jina_client import JinaEmbeddingsClient

logger = logging.getLogger(__name__)


class CacheClient:
    """Redis-based semantic and exact match cache for RAG queries."""

    def __init__(
        self, redis_client: redis.Redis, settings: RedisSettings, embeddings_client: Optional[JinaEmbeddingsClient] = None
    ):
        """Initialize cache client.

        :param redis_client: Connected Redis instance
        :param settings: Redis configurations
        :param embeddings_client: Optional JinaEmbeddingsClient to enable semantic caching
        """
        self.redis = redis_client
        self.settings = settings
        self.embeddings = embeddings_client
        self.ttl = timedelta(hours=settings.ttl_hours)
        self.similarity_threshold = 0.92
        self.index_key = "semantic_cache_index"

    def _generate_cache_key(self, request: AskRequest) -> str:
        """Generate exact cache key based on request parameters."""
        key_data = {
            "query": request.query,
            "model": request.model,
            "top_k": request.top_k,
            "use_hybrid": request.use_hybrid,
            "categories": sorted(request.categories) if request.categories else [],
        }
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        return f"exact_cache:{key_hash}"

    async def find_cached_response(self, request: AskRequest) -> Optional[AskResponse]:
        """Find cached response (uses semantic match if embeddings are active, otherwise exact match)."""
        try:
            # 1. Exact Match Cache Check (fast O(1) query)
            exact_key = self._generate_cache_key(request)
            cached_response = self.redis.get(exact_key)
            if cached_response:
                try:
                    response_data = json.loads(cached_response)
                    logger.info("Cache hit for exact query match")
                    return AskResponse(**response_data)
                except Exception as e:
                    logger.warning(f"Failed to deserialize cached exact response: {e}")

            # 2. Semantic Cache Check (local cosine similarity)
            if self.embeddings:
                logger.info(f"Checking semantic cache for query: '{request.query[:50]}...'")

                # Fetch all keys currently indexed in the semantic cache
                cache_keys = self.redis.smembers(self.index_key)
                if not cache_keys:
                    return None

                # Generate query embedding
                query_vector = await self.embeddings.embed_query(request.query)

                # Fetch all cached query data using pipeline (batch GET)
                pipeline = self.redis.pipeline()
                keys_list = list(cache_keys)
                for key in keys_list:
                    pipeline.get(key)
                results = pipeline.execute()

                best_similarity = -1.0
                best_response = None

                # Precompute query magnitude
                q_magnitude = sum(x * x for x in query_vector) ** 0.5

                for key, result_str in zip(keys_list, results):
                    if not result_str:
                        # Clean up stale reference in index if key expired in Redis
                        self.redis.srem(self.index_key, key)
                        continue

                    try:
                        data = json.loads(result_str)
                        cached_vector = data.get("embedding")

                        if not cached_vector or len(cached_vector) != len(query_vector):
                            continue

                        # Pure Python Cosine Similarity Calculation
                        dot_product = sum(q * c for q, c in zip(query_vector, cached_vector))
                        c_magnitude = sum(c * c for c in cached_vector) ** 0.5

                        if q_magnitude > 0 and c_magnitude > 0:
                            similarity = dot_product / (q_magnitude * c_magnitude)
                        else:
                            similarity = 0.0

                        if similarity > best_similarity:
                            best_similarity = similarity
                            if similarity >= self.similarity_threshold:
                                best_response = AskResponse(**data["response"])
                    except Exception as e:
                        logger.warning(f"Error parsing cache entry: {e}")

                if best_response:
                    logger.info(
                        f"✓ Semantic cache hit! Similarity: {best_similarity:.4f} (threshold: {self.similarity_threshold})"
                    )
                    return best_response
                else:
                    logger.info(f"Semantic cache miss. Best similarity found: {best_similarity:.4f}")

            return None

        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return None

    async def store_response(self, request: AskRequest, response: AskResponse) -> bool:
        """Store response in cache (both exact match and semantic match)."""
        try:
            # 1. Store in exact match cache
            exact_key = self._generate_cache_key(request)
            exact_success = self.redis.set(exact_key, response.model_dump_json(), ex=self.ttl)
            if exact_success:
                logger.info(f"Stored response in exact cache with key {exact_key[:16]}...")

            # 2. If embeddings are active, store in semantic cache
            if self.embeddings:
                query_hash = hashlib.sha256(request.query.encode()).hexdigest()[:16]
                semantic_key = f"semantic_cache:{query_hash}"

                # Fetch query embedding
                query_vector = await self.embeddings.embed_query(request.query)

                cache_data = {
                    "query": request.query,
                    "embedding": query_vector,
                    "response": response.model_dump(),
                }

                # Store cache entry
                semantic_success = self.redis.set(semantic_key, json.dumps(cache_data), ex=self.ttl)
                if semantic_success:
                    # Index the key in the set
                    self.redis.sadd(self.index_key, semantic_key)
                    logger.info(f"Stored response in semantic cache with key {semantic_key[:16]}...")

                    # Manage cache size (keep only top 500 keys for performance)
                    current_size = self.redis.scard(self.index_key)
                    if current_size > 500:
                        all_keys = list(self.redis.smembers(self.index_key))
                        to_remove = all_keys[: current_size - 500]
                        for key in to_remove:
                            self.redis.srem(self.index_key, key)
                            self.redis.delete(key)

            return True

        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            return False
