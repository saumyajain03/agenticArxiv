import logging

import redis
from src.config import Settings
from src.services.cache.client import CacheClient
from src.services.embeddings.factory import make_embeddings_client

logger = logging.getLogger(__name__)


def make_redis_client(settings: Settings) -> redis.Redis:
    """Create Redis client from a URL (supports local redis:// and Upstash rediss://)."""
    url = settings.redis.url
    try:
        client = redis.from_url(
            url,
            decode_responses=True,
            socket_timeout=30,
            socket_connect_timeout=30,
            retry_on_timeout=True,
            retry_on_error=[redis.ConnectionError, redis.TimeoutError],
        )
        client.ping()
        logger.info(f"Connected to Redis at {url.split('@')[-1] if '@' in url else url}")
        return client
    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating Redis client: {e}")
        raise


def make_cache_client(settings: Settings) -> CacheClient:
    """Create semantic and exact match cache client."""
    try:
        redis_client = make_redis_client(settings)
        # Create embeddings client for semantic cache matching
        embeddings_client = make_embeddings_client(settings)
        cache_client = CacheClient(redis_client, settings.redis, embeddings_client)
        logger.info("Semantic cache client created successfully")
        return cache_client
    except Exception as e:
        logger.error(f"Failed to create cache client: {e}")
        raise
