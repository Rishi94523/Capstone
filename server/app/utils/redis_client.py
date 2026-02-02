"""
Redis client utilities.
"""

import logging
from typing import Optional

import redis.asyncio as redis

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Global Redis connection pool
_redis_pool: Optional[redis.Redis] = None


async def init_redis() -> None:
    """Initialize Redis connection pool."""
    global _redis_pool

    _redis_pool = redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=False,
    )

    # Test connection
    try:
        await _redis_pool.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise


async def close_redis() -> None:
    """Close Redis connection pool."""
    global _redis_pool

    if _redis_pool:
        await _redis_pool.close()
        _redis_pool = None
        logger.info("Redis connection closed")


async def get_redis() -> redis.Redis:
    """Get Redis connection for dependency injection."""
    if _redis_pool is None:
        await init_redis()

    return _redis_pool


class RedisSessionStore:
    """Session storage using Redis."""

    def __init__(self, redis_client: redis.Redis, prefix: str = "session"):
        self.redis = redis_client
        self.prefix = prefix

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}:{session_id}"

    async def set(
        self, session_id: str, data: dict, ttl_seconds: int = 300
    ) -> None:
        """Store session data."""
        import json

        key = self._key(session_id)
        await self.redis.setex(key, ttl_seconds, json.dumps(data))

    async def get(self, session_id: str) -> Optional[dict]:
        """Retrieve session data."""
        import json

        key = self._key(session_id)
        data = await self.redis.get(key)

        if data:
            return json.loads(data)
        return None

    async def delete(self, session_id: str) -> None:
        """Delete session data."""
        key = self._key(session_id)
        await self.redis.delete(key)

    async def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        key = self._key(session_id)
        return await self.redis.exists(key) > 0

    async def extend(self, session_id: str, ttl_seconds: int) -> bool:
        """Extend session TTL."""
        key = self._key(session_id)
        return await self.redis.expire(key, ttl_seconds)


class RedisRateLimiter:
    """Rate limiting using Redis."""

    def __init__(
        self,
        redis_client: redis.Redis,
        prefix: str = "ratelimit",
        max_requests: int = 60,
        window_seconds: int = 60,
    ):
        self.redis = redis_client
        self.prefix = prefix
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def _key(self, identifier: str) -> str:
        return f"{self.prefix}:{identifier}"

    async def is_allowed(self, identifier: str) -> tuple:
        """
        Check if request is allowed.

        Returns:
            Tuple of (is_allowed, remaining_requests, reset_time)
        """
        key = self._key(identifier)

        # Use pipeline for atomic operations
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.ttl(key)
        results = await pipe.execute()

        current_count = results[0]
        ttl = results[1]

        # Set expiry if this is the first request
        if ttl == -1:
            await self.redis.expire(key, self.window_seconds)
            ttl = self.window_seconds

        remaining = max(0, self.max_requests - current_count)
        is_allowed = current_count <= self.max_requests

        return is_allowed, remaining, ttl

    async def reset(self, identifier: str) -> None:
        """Reset rate limit for an identifier."""
        key = self._key(identifier)
        await self.redis.delete(key)
