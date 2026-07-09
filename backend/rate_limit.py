import time
import logging
import json
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import config
from redis_store import get_redis

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # We only apply rate limiting to RAG and Chat API routes
        if not request.url.path.startswith("/api/chats/ask") and not request.url.path.startswith("/api/ask"):
            return await call_next(request)
            
        redis_client = get_redis()
        if not redis_client:
            # Fallback: if Redis is down, we allow request rather than breaking service
            return await call_next(request)
            
        rate_limit = getattr(config, "RATE_LIMIT_PER_MINUTE", 60)
        if rate_limit <= 0:
            return await call_next(request)
            
        session_id = None
        
        # Check X-Session-ID header
        session_id = request.headers.get("x-session-id")
        
        # Check query parameters
        if not session_id:
            session_id = request.query_params.get("session_id")
            
        # Fallback to client IP address if session_id is not specified
        identifier = session_id or (request.client.host if request.client else "unknown")
        
        now = time.time()
        key = f"ratelimit:{identifier}"
        
        try:
            # Sliding window using Redis sorted sets (timestamps as both score and member)
            pipe = redis_client.pipeline()
            # 1. Clean up elements older than 60 seconds
            pipe.zremrangebyscore(key, 0, now - 60)
            # 2. Add current request timestamp
            pipe.zadd(key, {str(now): now})
            # 3. Get count of requests in the window
            pipe.zcard(key)
            # 4. Refresh key expiration so it auto-cleans up
            pipe.expire(key, 60)
            
            _, _, count, _ = pipe.execute()
            
            if count > rate_limit:
                logger.warning(f"Rate limit exceeded for user/session '{identifier}': {count} requests/min (limit is {rate_limit})")
                redis_client.incr("metrics:rate_limit_rejections")
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests. Please try again later."}
                )
                
        except Exception as e:
            logger.error(f"Error checking rate limits in Redis: {e}")
            
        return await call_next(request)
