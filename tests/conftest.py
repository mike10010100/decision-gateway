"""
Pytest Fixtures and Mock Infrastructure for Decision Gateway E2E Tests.
Provides offline hermetic mocks for upstream Ollaya (ports 11435 and 11436)
using respx, httpx.AsyncClient (ASGITransport), and FastAPI TestClient.
"""

import sys
import os
import json
import pytest
import pytest_asyncio
import respx
import httpx

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from app.main import app
from app.config import OLLAYA_URL, OLLAYA_MCP_URL

# ---------------------------------------------------------------------------
# Client Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sync_client():
    """Synchronous FastAPI TestClient."""
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest_asyncio.fixture
async def async_client():
    """Asynchronous httpx client running directly against FastAPI app via ASGITransport."""
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ---------------------------------------------------------------------------
# Standard Sample Payloads
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_payloads():
    """Standard, well-formed decision payloads for testing."""
    return {
        "choice": {
            "state": "The user reported receiving HTTP 500 error when clicking submit.",
            "questions": {
                "category": {
                    "type": "choice",
                    "criteria": {
                        "frontend_error": "User interface or display bug",
                        "backend_error": "Server error, 500 status code, or database failure",
                        "auth_issue": "Login or authentication problem",
                    },
                }
            },
        },
        "score": {
            "state": "The entire production cluster is unreachable across multiple regions.",
            "questions": {
                "urgency": {
                    "type": "score",
                    "criteria": ["1 - minor", "5 - critical production outage"],
                    "description": (
                        "Severity scale from 1 (minor) to 5 (critical production outage)"
                    ),
                }
            },
        },
        "noul": {
            "state": "Payment processed successfully and order invoice generated.",
            "questions": {
                "requires_human_escalation": {
                    "type": "noul",
                    "description": "True if an agent must manually inspect this transaction",
                }
            },
        },
        "multi": {
            "state": (
                "Customer account balance shows negative $400 after unauthorized overseas charge."
            ),
            "questions": {
                "triage_category": {
                    "type": "choice",
                    "criteria": {
                        "fraud": "Suspected fraud or compromised credentials",
                        "billing_dispute": "User dispute on valid charge",
                        "inquiry": "General account inquiry",
                    },
                },
                "fraud_risk_score": {
                    "type": "score",
                    "criteria": ["0 - low risk", "100 - critical fraud risk"],
                    "description": "Likelihood that transaction is fraudulent",
                },
                "freeze_account_immediately": {
                    "type": "noul",
                    "description": "Whether to apply an immediate security lock on the account",
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Upstream Mock Fixtures (Ollaya Inference on 11435)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ollaya_inference(respx_mock):
    """
    Mocks standard Ollaya inference responses on POST /v1/systemone.
    Dynamically mirrors back the model requested with high-fidelity mock answers.
    """

    def inference_callback(request):
        try:
            req_body = json.loads(request.content.decode("utf-8"))
        except Exception:
            req_body = {}

        model = req_body.get("model", "decider")
        questions = req_body.get("questions", {})

        answers = {}
        for q_id, q_spec in questions.items():
            q_type = q_spec.get("type", "choice") if isinstance(q_spec, dict) else "choice"
            if q_type == "choice":
                criteria = q_spec.get("criteria", {}) if isinstance(q_spec, dict) else {}
                first_key = next(iter(criteria.keys())) if criteria else "default"
                answers[q_id] = first_key
            elif q_type == "score":
                answers[q_id] = 4
            elif q_type == "noul":
                answers[q_id] = True
            else:
                answers[q_id] = "processed"

        return httpx.Response(
            status_code=200,
            json={
                "answers": answers,
                "model": model,
                "usage": {"prompt_tokens": 42, "completion_tokens": 12, "total_tokens": 54},
            },
        )

    route = respx_mock.post(f"{OLLAYA_URL}/v1/systemone").mock(side_effect=inference_callback)
    return route


@pytest.fixture
def mock_ollaya_tags(respx_mock):
    """
    Mocks live Ollaya model tags endpoint GET /api/tags with standard model catalog.
    """
    catalog = {
        "models": [
            {
                "name": "decider:latest",
                "model": "decider:latest",
                "size": 3984588800,
                "digest": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
                "modified_at": "2026-09-01T12:00:00.000000Z",
                "details": {
                    "format": "onnx",
                    "parameter_size": "1.9B",
                    "quantization_level": "fp16",
                },
            },
            {
                "name": "laya:latest",
                "model": "laya:latest",
                "size": 891289600,
                "digest": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
                "modified_at": "2026-09-01T12:00:00.000000Z",
                "details": {
                    "format": "onnx",
                    "parameter_size": "421M",
                    "quantization_level": "fp16",
                },
            },
            {
                "name": "kev:4b",
                "model": "kev:4b",
                "size": 9961472000,
                "digest": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
                "modified_at": "2026-09-01T12:00:00.000000Z",
                "details": {
                    "format": "onnx",
                    "parameter_size": "4.0B",
                    "quantization_level": "fp16",
                },
            },
            {
                "name": "qwen3guard:latest",
                "model": "qwen3guard:latest",
                "size": 1572864000,
                "digest": "sha256:4444444444444444444444444444444444444444444444444444444444444444",
                "modified_at": "2026-09-01T12:00:00.000000Z",
                "details": {
                    "format": "onnx",
                    "parameter_size": "1.5B",
                    "quantization_level": "fp16",
                },
            },
        ]
    }
    route = respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(status_code=200, json=catalog)
    return route


@pytest.fixture
def mock_ollaya_tags_corrupt(respx_mock):
    """
    Mocks live Ollaya model tags endpoint containing corrupt models:
    - one with 'details': None (causes crash if unhandled)
    - one with missing 'details'
    - one with valid details
    """
    corrupt_catalog = {
        "models": [
            {
                "name": "corrupt_null_details:latest",
                "size": 1024000,
                "modified_at": "2026-09-01T12:00:00Z",
                "details": None,
            },
            {
                "name": "corrupt_missing_details:latest",
                "size": 2048000,
                "modified_at": "2026-09-01T12:00:00Z",
            },
            {
                "name": "decider:latest",
                "size": 3984588800,
                "modified_at": "2026-09-01T12:00:00Z",
                "details": {"format": "onnx", "parameter_size": "1.9B"},
            },
        ]
    }
    route = respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(status_code=200, json=corrupt_catalog)
    return route


@pytest.fixture
def mock_ollaya_pull(respx_mock):
    """Mocks standard model pull response on POST /api/pull."""
    route = respx_mock.post(f"{OLLAYA_URL}/api/pull").respond(
        status_code=200, json={"status": "success", "message": "Model downloaded successfully"}
    )
    return route


@pytest.fixture
def mock_ollaya_pull_stream(respx_mock):
    """Mocks streaming NDJSON response on POST /api/pull."""
    ndjson_body = (
        b'{"status": "pulling manifest"}\n'
        b'{"status": "downloading", "completed": 250000000, "total": 1000000000}\n'
        b'{"status": "downloading", "completed": 750000000, "total": 1000000000}\n'
        b'{"status": "verifying checksum"}\n'
        b'{"status": "success"}\n'
    )
    route = respx_mock.post(f"{OLLAYA_URL}/api/pull").respond(
        status_code=200, headers={"content-type": "application/x-ndjson"}, content=ndjson_body
    )
    return route


@pytest.fixture
def mock_ollaya_delete(respx_mock):
    """Mocks model deletion response on DELETE /api/delete."""
    route = respx_mock.delete(f"{OLLAYA_URL}/api/delete").respond(
        status_code=200, json={"status": "success", "message": "Model deleted successfully"}
    )
    return route


# ---------------------------------------------------------------------------
# Upstream MCP SSE Mock Fixtures (Ollaya MCP on 11436)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ollaya_mcp_sse(respx_mock):
    """
    Mocks standard Ollaya MCP SSE stream on port 11436 with session ID header
    and standard SSE events including 15s quiet keepalive pings.
    """
    sse_body = (
        b"retry: 3000\n\n"
        b'event: endpoint\ndata: "/mcp/messages?session=test-uuid"\n\n'
        b":\n\n"
        b'event: message\ndata: {"jsonrpc": "2.0", "method": "tools/list", "params": {}}\n\n'
        b":\n\n"
    )

    get_route = respx_mock.get(f"{OLLAYA_MCP_URL}/mcp").respond(
        status_code=200,
        headers={
            "content-type": "text/event-stream",
            "cache-control": "no-cache",
            "mcp-session-id": "test-session-uuid-12345",
        },
        content=sse_body,
    )

    post_route = respx_mock.post(f"{OLLAYA_MCP_URL}/mcp").respond(
        status_code=200,
        headers={"content-type": "application/json", "mcp-session-id": "test-session-uuid-12345"},
        json={
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "ollaya-mcp", "version": "1.0.0"},
            },
        },
    )

    return get_route, post_route


# ---------------------------------------------------------------------------
# Failure Simulation Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simulate_network_disconnect(respx_mock):
    """Helper to simulate upstream connection failure (ConnectError)."""

    def _apply(url: str, method: str = "POST"):
        if method.upper() == "POST":
            return respx_mock.post(url).mock(
                side_effect=httpx.ConnectError("Connection refused: 127.0.0.1")
            )
        elif method.upper() == "GET":
            return respx_mock.get(url).mock(
                side_effect=httpx.ConnectError("Connection refused: 127.0.0.1")
            )
        elif method.upper() == "DELETE":
            return respx_mock.delete(url).mock(
                side_effect=httpx.ConnectError("Connection refused: 127.0.0.1")
            )

    return _apply


@pytest.fixture
def simulate_timeout(respx_mock):
    """Helper to simulate upstream timeout (ReadTimeout or ConnectTimeout)."""

    def _apply(url: str, method: str = "POST", timeout_type: str = "read"):
        exc = (
            httpx.ReadTimeout("Read timed out on upstream request")
            if timeout_type == "read"
            else httpx.ConnectTimeout("Connect timed out on upstream request")
        )
        if method.upper() == "POST":
            return respx_mock.post(url).mock(side_effect=exc)
        elif method.upper() == "GET":
            return respx_mock.get(url).mock(side_effect=exc)
        elif method.upper() == "DELETE":
            return respx_mock.delete(url).mock(side_effect=exc)

    return _apply


@pytest.fixture
def simulate_upstream_crashed(respx_mock):
    """Helper to simulate upstream 500/503 server errors."""

    def _apply(
        url: str,
        method: str = "POST",
        status_code: int = 500,
        error_msg: str = "CUDA out of memory",
    ):
        if method.upper() == "POST":
            return respx_mock.post(url).respond(status_code=status_code, json={"error": error_msg})
        elif method.upper() == "GET":
            return respx_mock.get(url).respond(status_code=status_code, json={"error": error_msg})
        elif method.upper() == "DELETE":
            return respx_mock.delete(url).respond(
                status_code=status_code, json={"error": error_msg}
            )

    return _apply


@pytest.fixture
def simulate_missing_model(respx_mock):
    """Helper to simulate upstream 404 Model Not Found."""

    def _apply(url: str, model_name: str = "unknown_model"):
        return respx_mock.post(url).respond(
            status_code=404, json={"error": f"model '{model_name}' not found, try pulling it first"}
        )

    return _apply


# ---------------------------------------------------------------------------
# Hermetic Autouse Baseline Mock
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def hermetic_upstream_baseline(respx_mock):
    """
    Autouse fixture that sets up default offline mocks for upstream Ollaya endpoints.
    Allows tests to run completely offline without hitting real ports 11435 or 11436.
    Specific tests can override these routes via respx_mock.
    """
    respx_mock.assert_all_mocked = False

    # Default inference endpoint
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").respond(
        status_code=200,
        json={
            "answers": {"default_q": "default_answer"},
            "model": "decider",
            "usage": {"total_tokens": 30},
        },
    )

    # Default tags endpoint
    respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
        status_code=200,
        json={
            "models": [
                {
                    "name": "decider:latest",
                    "size": 3984588800,
                    "modified_at": "2026-09-01T12:00:00Z",
                    "details": {"format": "onnx", "parameter_size": "1.9B"},
                },
                {
                    "name": "laya:latest",
                    "size": 891289600,
                    "modified_at": "2026-09-01T12:00:00Z",
                    "details": {"format": "onnx", "parameter_size": "421M"},
                },
            ]
        },
    )

    # Default pull endpoint
    respx_mock.post(f"{OLLAYA_URL}/api/pull").respond(
        status_code=200, json={"status": "success", "message": "Model downloaded successfully"}
    )

    # Default delete endpoint
    respx_mock.delete(f"{OLLAYA_URL}/api/delete").respond(
        status_code=200, json={"status": "success", "message": "Model deleted"}
    )

    # Default MCP endpoint
    respx_mock.get(f"{OLLAYA_MCP_URL}/mcp").respond(
        status_code=200,
        headers={"content-type": "text/event-stream", "mcp-session-id": "default-session"},
        content=b":\n\n",
    )
    respx_mock.post(f"{OLLAYA_MCP_URL}/mcp").respond(
        status_code=200,
        headers={"content-type": "application/json", "mcp-session-id": "default-session"},
        json={"jsonrpc": "2.0", "result": {"serverInfo": {"name": "ollaya-mcp"}}},
    )
