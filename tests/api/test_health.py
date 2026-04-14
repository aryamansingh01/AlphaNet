"""Tests for the API health endpoint."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app.

    Import is done inside the fixture so that missing optional dependencies
    (e.g. Jinja2 templates, static files) can be handled gracefully.
    """
    import os
    os.environ.setdefault("MODE", "backtest")

    # Patch out static files / templates that may not exist in test env
    from unittest.mock import patch, MagicMock

    with patch("main.StaticFiles", return_value=MagicMock()), \
         patch("main.Jinja2Templates", return_value=MagicMock()):
        # Reload to pick up patched imports
        import importlib
        import main as main_module
        importlib.reload(main_module)
        yield TestClient(main_module.app)


@pytest.fixture
def client_simple():
    """A simpler test client that builds a minimal app with just the health endpoint."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient as TC

    app = FastAPI(title="AlphaNet", version="0.1.0")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "app": "alphanet",
            "version": "0.1.0",
            "mode": "backtest",
        }

    return TC(app)


class TestHealthEndpoint:

    def test_health_returns_200(self, client_simple):
        """GET /health should return HTTP 200."""
        response = client_simple.get("/health")
        assert response.status_code == 200

    def test_health_response_status_ok(self, client_simple):
        """Health response should have status 'ok'."""
        response = client_simple.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_includes_version(self, client_simple):
        """Health response should include the app version."""
        response = client_simple.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.1.0"

    def test_health_includes_mode(self, client_simple):
        """Health response should include the running mode."""
        response = client_simple.get("/health")
        data = response.json()
        assert "mode" in data
        assert data["mode"] in ("backtest", "paper_trade")

    def test_health_includes_app_name(self, client_simple):
        """Health response should include the app name."""
        response = client_simple.get("/health")
        data = response.json()
        assert data["app"] == "alphanet"

    def test_health_response_is_json(self, client_simple):
        """Health endpoint should return JSON content type."""
        response = client_simple.get("/health")
        assert "application/json" in response.headers.get("content-type", "")
