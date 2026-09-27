"""
E2E Upstream Resilience Test Suite (Feature F2, Tiers 1-2).
Tests fault tolerance when upstream Ollaya fails across all error modes:
- Network disconnects / connection refused -> HTTP 502 Bad Gateway
- Crashed inference / 500 / 503 errors -> HTTP 502 Bad Gateway
- Connect and read timeouts -> HTTP 504 Gateway Timeout
- Missing models / 404 responses -> Clean structured handling without crashes
"""

import sys
import os
import pytest
import httpx
import respx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.config import OLLAYA_URL

# ===========================================================================
# Tier 1: Core Feature Coverage (F2 Upstream Fault Mapping)
# ===========================================================================


@pytest.mark.asyncio
async def test_auto_upstream_connect_error_returns_502(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
):
    """F2: Upstream connection refused on /v1/auto must return 502 Bad Gateway."""
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").mock(
        side_effect=httpx.ConnectError("Connection refused: 127.0.0.1:11435")
    )
    resp = await async_client.post("/v1/auto", json=sample_payloads["choice"])
    assert resp.status_code == 502, f"Expected 502, got {resp.status_code}: {resp.text}"
    assert "detail" in resp.json()


@pytest.mark.asyncio
async def test_systemone_upstream_connect_error_returns_502(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
):
    """F2: Upstream connection refused on /v1/systemone must return 502 Bad Gateway."""
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").mock(
        side_effect=httpx.ConnectError("Connection refused: 127.0.0.1:11435")
    )
    resp = await async_client.post("/v1/systemone", json=sample_payloads["choice"])
    assert resp.status_code == 502, f"Expected 502, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_upstream_read_timeout_returns_504(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
):
    """F2: Upstream read timeout on /v1/auto must return 504 Gateway Timeout."""
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").mock(
        side_effect=httpx.ReadTimeout("Read timed out on 127.0.0.1:11435")
    )
    resp = await async_client.post("/v1/auto", json=sample_payloads["choice"])
    assert resp.status_code == 504, f"Expected 504, got {resp.status_code}: {resp.text}"
    assert "detail" in resp.json()


@pytest.mark.asyncio
async def test_systemone_upstream_read_timeout_returns_504(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
):
    """F2: Upstream read timeout on /v1/systemone must return 504 Gateway Timeout."""
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").mock(
        side_effect=httpx.ReadTimeout("Read timed out on 127.0.0.1:11435")
    )
    resp = await async_client.post("/v1/systemone", json=sample_payloads["choice"])
    assert resp.status_code == 504, f"Expected 504, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_upstream_connect_timeout_returns_504(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
):
    """F2: Upstream connect timeout on /v1/auto must return 504 Gateway Timeout."""
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").mock(
        side_effect=httpx.ConnectTimeout("Connect timed out on 127.0.0.1:11435")
    )
    resp = await async_client.post("/v1/auto", json=sample_payloads["choice"])
    assert resp.status_code == 504, f"Expected 504, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_upstream_500_crashed_returns_502(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
):
    """F2: Upstream 500 (CUDA OOM/crash) must be converted to 502 Bad Gateway."""
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").respond(
        status_code=500, json={"error": "CUDA out of memory in inference worker"}
    )
    resp = await async_client.post("/v1/auto", json=sample_payloads["choice"])
    assert resp.status_code == 502, f"Expected 502, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_upstream_503_unavailable_returns_502(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
):
    """F2: Upstream 503 Service Unavailable must be converted to 502 Bad Gateway."""
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").respond(
        status_code=503, json={"error": "Inference worker busy or unavailable"}
    )
    resp = await async_client.post("/v1/auto", json=sample_payloads["choice"])
    assert resp.status_code == 502, f"Expected 502, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_pull_sync_upstream_connect_error_returns_502(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F2: Upstream ConnectError on /v1/models/pull must return 502 Bad Gateway."""
    respx_mock.post(f"{OLLAYA_URL}/api/pull").mock(
        side_effect=httpx.ConnectError("Connection refused: 127.0.0.1:11435")
    )
    resp = await async_client.post("/v1/models/pull", json={"model": "decider"})
    assert resp.status_code == 502, f"Expected 502, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_pull_sync_upstream_read_timeout_returns_504(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F2: Upstream ReadTimeout on /v1/models/pull must return 504 Gateway Timeout."""
    respx_mock.post(f"{OLLAYA_URL}/api/pull").mock(
        side_effect=httpx.ReadTimeout("Read timed out pulling model")
    )
    resp = await async_client.post("/v1/models/pull", json={"model": "decider"})
    assert resp.status_code == 504, f"Expected 504, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_pull_sync_upstream_connect_timeout_returns_504(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F2: Upstream ConnectTimeout on /v1/models/pull must return 504 Gateway Timeout."""
    respx_mock.post(f"{OLLAYA_URL}/api/pull").mock(
        side_effect=httpx.ConnectTimeout("Connect timed out pulling model")
    )
    resp = await async_client.post("/v1/models/pull", json={"model": "decider"})
    assert resp.status_code == 504, f"Expected 504, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_pull_sync_upstream_500_returns_502(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F2: Upstream 500 error on /v1/models/pull must return 502 Bad Gateway."""
    respx_mock.post(f"{OLLAYA_URL}/api/pull").respond(
        status_code=500, json={"error": "Disk quota exceeded on model registry"}
    )
    resp = await async_client.post("/v1/models/pull", json={"model": "decider"})
    assert resp.status_code == 502, f"Expected 502, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_delete_upstream_connect_error_returns_502(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F2: Upstream ConnectError on DELETE /v1/models/{model} must return 502."""
    respx_mock.delete(f"{OLLAYA_URL}/api/delete").mock(
        side_effect=httpx.ConnectError("Connection refused: 127.0.0.1:11435")
    )
    resp = await async_client.delete("/v1/models/decider")
    assert resp.status_code == 502, f"Expected 502, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_delete_upstream_timeout_returns_504(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F2: Upstream Timeout on DELETE /v1/models/{model} must return 504."""
    respx_mock.delete(f"{OLLAYA_URL}/api/delete").mock(
        side_effect=httpx.ReadTimeout("Timeout deleting model")
    )
    resp = await async_client.delete("/v1/models/decider")
    assert resp.status_code == 504, f"Expected 504, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_delete_upstream_500_returns_502(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F2: Upstream 500 on DELETE /v1/models/{model} must return 502."""
    respx_mock.delete(f"{OLLAYA_URL}/api/delete").respond(
        status_code=500, json={"error": "File lock active on model weights"}
    )
    resp = await async_client.delete("/v1/models/decider")
    assert resp.status_code == 502, f"Expected 502, got {resp.status_code}: {resp.text}"


# ===========================================================================
# Tier 2: Boundary, Adversarial & Failure Boundary Cases (F2)
# ===========================================================================


@pytest.mark.asyncio
async def test_auto_upstream_404_missing_model_handled(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
):
    """F2 Boundary: Missing model 404 from upstream should return 404 or structured error."""
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").respond(
        status_code=404, json={"error": "model 'decider' not found, try pulling it first"}
    )
    resp = await async_client.post("/v1/auto", json=sample_payloads["choice"])
    assert resp.status_code in [404, 502], f"Expected 404 or 502, got {resp.status_code}"


@pytest.mark.asyncio
async def test_pull_streaming_upstream_connect_error_handled(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F2 Boundary: Streaming model pull with upstream ConnectError must not crash gateway."""
    respx_mock.post(f"{OLLAYA_URL}/api/pull").mock(
        side_effect=httpx.ConnectError("Connection refused on pull stream")
    )
    resp = await async_client.post("/v1/models/pull", json={"model": "decider", "stream": True})
    assert resp.status_code in [502, 504], f"Expected 502 or 504, got {resp.status_code}"


@pytest.mark.asyncio
async def test_pull_streaming_upstream_timeout_handled(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F2 Boundary: Streaming model pull with upstream timeout returns 504."""
    respx_mock.post(f"{OLLAYA_URL}/api/pull").mock(
        side_effect=httpx.ReadTimeout("Timeout initiating pull stream")
    )
    resp = await async_client.post("/v1/models/pull", json={"model": "decider", "stream": True})
    assert resp.status_code in [502, 504], f"Expected 504 (or 502), got {resp.status_code}"


@pytest.mark.asyncio
async def test_pull_streaming_upstream_500_not_masked_as_200(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F2 Boundary: Streaming pull where upstream returns 500 immediately must NOT return 200."""
    respx_mock.post(f"{OLLAYA_URL}/api/pull").respond(
        status_code=500, json={"error": "Registry storage corrupt"}
    )
    resp = await async_client.post("/v1/models/pull", json={"model": "decider", "stream": True})
    assert resp.status_code != 200, f"Expected non-200 error code, got {resp.status_code}"


@pytest.mark.asyncio
async def test_auto_upstream_remote_protocol_error_returns_502(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
):
    """F2 Adversarial: Upstream connection reset (RemoteProtocolError) returns 502."""
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").mock(
        side_effect=httpx.RemoteProtocolError("Connection reset by peer")
    )
    resp = await async_client.post("/v1/auto", json=sample_payloads["choice"])
    assert resp.status_code == 502, f"Expected 502, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_upstream_malformed_html_500_handled(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
):
    """F2 Adversarial: Upstream returns non-JSON HTML crash page with status 500."""
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").respond(
        status_code=500,
        headers={"content-type": "text/html"},
        content=b"<html><body><h1>Internal Server Error: CUDA segfault</h1></body></html>",
    )
    resp = await async_client.post("/v1/auto", json=sample_payloads["choice"])
    assert resp.status_code == 502, f"Expected 502, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_error_response_contains_detail_contract(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
):
    """F2: Error response must conform to JSON contract with 'detail' field."""
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    resp = await async_client.post("/v1/auto", json=sample_payloads["choice"])
    assert resp.status_code == 502
    body = resp.json()
    assert "detail" in body
    assert isinstance(body["detail"], str)


@pytest.mark.asyncio
async def test_consecutive_upstream_failures_keep_gateway_alive(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
):
    """F2 Adversarial: Gateway handles multiple consecutive failures without dying."""
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    for _ in range(5):
        resp = await async_client.post("/v1/auto", json=sample_payloads["choice"])
        assert resp.status_code == 502

    # Verify gateway health remains 200
    health_resp = await async_client.get("/healthz")
    assert health_resp.status_code == 200


@pytest.mark.asyncio
async def test_pull_upstream_404_model_not_found(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F2 Boundary: Pulling a non-existent model returning 404 upstream is handled cleanly."""
    respx_mock.post(f"{OLLAYA_URL}/api/pull").respond(
        status_code=404, json={"error": "model 'nonexistent-model' not found in registry"}
    )
    resp = await async_client.post("/v1/models/pull", json={"model": "nonexistent-model"})
    assert resp.status_code in [404, 502], f"Expected 404 or 502, got {resp.status_code}"


@pytest.mark.asyncio
async def test_delete_upstream_404_model_not_found(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F2 Boundary: Deleting a non-existent model returning 404 upstream is handled cleanly."""
    respx_mock.delete(f"{OLLAYA_URL}/api/delete").respond(
        status_code=404, json={"error": "model 'nonexistent-model' not found"}
    )
    resp = await async_client.delete("/v1/models/nonexistent-model")
    assert resp.status_code in [404, 502], f"Expected 404 or 502, got {resp.status_code}"
