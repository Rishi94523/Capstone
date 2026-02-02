"""
Test configuration and fixtures.
"""

import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.main import app
from app.models.base import Base
from app.config import get_settings

settings = get_settings()

# Test database URL (use SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create a test HTTP client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_init_request() -> dict:
    """Sample CAPTCHA init request."""
    return {
        "site_key": "pk_test_1234567890",
        "client_metadata": {
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "language": "en-US",
            "timezone": "America/New_York",
            "screen_width": 1920,
            "screen_height": 1080,
        },
    }


@pytest.fixture
def sample_prediction() -> dict:
    """Sample prediction data."""
    return {
        "label": "cat",
        "confidence": 0.95,
        "top_k": [
            {"label": "cat", "confidence": 0.95},
            {"label": "dog", "confidence": 0.03},
            {"label": "bird", "confidence": 0.02},
        ],
    }


@pytest.fixture
def sample_pow() -> dict:
    """Sample proof of work data."""
    return {
        "hash": "0" + "a" * 63,  # Starts with 0 for validity
        "nonce": 12345,
        "model_checksum": "abc123",
        "input_hash": "def456",
        "output_hash": "ghi789",
    }


@pytest.fixture
def sample_timing() -> dict:
    """Sample timing data."""
    return {
        "model_load_ms": 200,
        "inference_ms": 350,
        "total_ms": 550,
        "started_at": 1700000000000,
        "completed_at": 1700000000550,
    }
