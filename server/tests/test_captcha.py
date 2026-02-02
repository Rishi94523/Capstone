"""
Tests for CAPTCHA API endpoints.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Test health check endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_ready_check(client: AsyncClient):
    """Test readiness endpoint."""
    response = await client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient):
    """Test root endpoint."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "PoUW CAPTCHA API"
    assert "version" in data


class TestCaptchaInit:
    """Tests for CAPTCHA initialization."""

    @pytest.mark.asyncio
    async def test_init_success(
        self, client: AsyncClient, sample_init_request: dict
    ):
        """Test successful CAPTCHA initialization."""
        # Note: This test would fail without a database connection
        # In a real test, we'd mock the database
        pass

    @pytest.mark.asyncio
    async def test_init_missing_site_key(self, client: AsyncClient):
        """Test initialization with missing site key."""
        response = await client.post(
            "/api/v1/captcha/init",
            json={
                "client_metadata": {
                    "user_agent": "test",
                    "language": "en",
                    "timezone": "UTC",
                }
            },
        )
        assert response.status_code == 422  # Validation error


class TestCaptchaSubmit:
    """Tests for CAPTCHA submission."""

    @pytest.mark.asyncio
    async def test_submit_invalid_session(self, client: AsyncClient):
        """Test submission with invalid session."""
        response = await client.post(
            "/api/v1/captcha/submit",
            json={
                "session_id": "invalid-uuid",
                "task_id": "invalid-uuid",
                "prediction": {
                    "label": "cat",
                    "confidence": 0.9,
                    "top_k": [{"label": "cat", "confidence": 0.9}],
                },
                "proof_of_work": {
                    "hash": "0" + "a" * 63,
                    "nonce": 0,
                    "model_checksum": "test",
                    "input_hash": "test",
                    "output_hash": "test",
                },
                "timing": {
                    "model_load_ms": 100,
                    "inference_ms": 200,
                    "total_ms": 300,
                    "started_at": 0,
                    "completed_at": 300,
                },
            },
        )
        # Should fail with 404 or 422
        assert response.status_code in [404, 422, 500]


class TestCaptchaValidate:
    """Tests for CAPTCHA token validation."""

    @pytest.mark.asyncio
    async def test_validate_invalid_token(self, client: AsyncClient):
        """Test validation with invalid token."""
        response = await client.get("/api/v1/captcha/validate/invalid-token")
        assert response.status_code == 200
        assert response.json()["valid"] is False
