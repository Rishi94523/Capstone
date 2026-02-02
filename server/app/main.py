"""
PoUW CAPTCHA Server - Main Application

FastAPI application entry point with middleware, routes, and lifecycle management.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.api import captcha, verification, federated
from app.models import init_db, close_db
from app.utils.redis_client import init_redis, close_redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("Starting PoUW CAPTCHA Server...")

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Initialize Redis
    await init_redis()
    logger.info("Redis initialized")

    yield

    # Shutdown
    logger.info("Shutting down PoUW CAPTCHA Server...")

    await close_db()
    await close_redis()

    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="PoUW CAPTCHA API",
    description="Proof-of-Useful-Work CAPTCHA System API",
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
        },
    )


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for load balancers."""
    return {"status": "healthy", "version": "0.1.0"}


# Ready check endpoint
@app.get("/ready", tags=["Health"])
async def ready_check():
    """Readiness check for Kubernetes."""
    # TODO: Add database and Redis connectivity checks
    return {"status": "ready"}


# Include API routers
app.include_router(captcha.router, prefix="/api/v1", tags=["CAPTCHA"])
app.include_router(verification.router, prefix="/api/v1", tags=["Verification"])
app.include_router(federated.router, prefix="/api/v1", tags=["Federated Learning"])


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "name": "PoUW CAPTCHA API",
        "version": "0.1.0",
        "docs": "/docs" if settings.debug else None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
