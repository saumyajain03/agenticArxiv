import logging

import redis
from src.config import Settings
from src.services.cache.client import CacheClient
from src.services.embeddings.factory import make_embeddings_client

logger = logging.getLogger(__name__)


def make_redis_client(settings: Settings) -> redis.Redis | None:
    """Create Redis client from a URL (supports local redis:// and Upstash rediss://).

    Returns None if connection fails to prevent crashing background startup on Render.
    """
    url = settings.redis.url
    try:
        client = redis.from_url(
            url,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=False,
            retry_on_error=[redis.ConnectionError, redis.TimeoutError],
        )
        client.ping()
        logger.info(f"Connected to Redis at {url.split('@')[-1] if '@' in url else url}")
        return client
    except Exception as e:
        logger.warning(f"Redis connection unavailable at {url.split('@')[-1] if '@' in url else url}: {e}. Proceeding without Redis.")
        return None


def make_cache_client(settings: Settings) -> CacheClient | None:
    """Create semantic and exact match cache client.

    Returns None if Redis is unavailable.
    """
    try:
        redis_client = make_redis_client(settings)
        if redis_client is None:
            return None
        # Create embeddings client for semantic cache matching
        embeddings_client = make_embeddings_client(settings)
        cache_client = CacheClient(redis_client, settings.redis, embeddings_client)
        logger.info("Semantic cache client created successfully")
        return cache_client
    except Exception as e:
        logger.warning(f"Failed to create cache client: {e}. Proceeding with cache disabled.")
        return None

