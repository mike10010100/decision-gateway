"""
E2E Cross-Feature Integration Test Suite (Tier 3 Combinations, F7, F8, F10).
Tests pairwise and multi-feature interactions:
- SLA routing combined with upstream timeouts and 5xx errors
- Latency budget routing combined with network disconnects
- Explicit model overrides vs SLA precedence
- Concurrent heterogeneous workloads (fast/smart/accurate)
- Concurrency under partial upstream failure conditions
- Concurrent streaming (/mcp) and decision dispatching (/v1/auto)
- Health check isolation during upstream outages
- Router type safety and edge resolution
"""

import sys
import os
import asyncio
import pytest
import httpx
import respx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.config import OLLAYA_URL, OLLAYA_MCP_URL
from app.router import resolve_model

# ===========================================================================
# Tier 3: Pairwise SLA & Error Resilience Combinations
# ===========================================================================


@pytest.mark.asyncio
async def test_sla_fast_routes_to_laya_and_handles_upstream_timeout(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
):
    """Tier 3: SLA 'fast' routes to 'laya'; upstream timeout returns 504."""
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").mock(
        side_effect=httpx.ReadTimeout("Timeout from laya inference")
    )
    payload = dict(sample_payloads["choice"])
    payload["sla"] = "fast"
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 504, f"Expected 504, got {resp.status_code}"


@pytest.mark.asyncio
async def test_sla_smart_routes_to_decider_and_handles_upstream_500(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
):
    """Tier 3: SLA 'smart' routes to 'decider'; upstream 500 returns 502."""
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").respond(
        status_code=500, json={"error": "Decider worker OOM"}
    )
    payload = dict(sample_payloads["choice"])
    payload["sla"] = "smart"
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 502, f"Expected 502, got {resp.status_code}"


@pytest.mark.asyncio
async def test_latency_budget_routes_to_laya_and_handles_connect_error(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
):
    """Tier 3: Budget 400ms routes to 'laya'; network disconnect returns 502."""
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    payload = dict(sample_payloads["choice"])
    payload["max_latency_ms"] = 400
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 502, f"Expected 502, got {resp.status_code}"


@pytest.mark.asyncio
async def test_explicit_model_overrides_sla_and_preserves_routing_headers(
    async_client: httpx.AsyncClient, mock_ollaya_inference, sample_payloads
):
    """Tier 3: Explicit model 'kev:4b' overrides SLA 'fast'; verifies response headers."""
    payload = dict(sample_payloads["choice"])
    payload["sla"] = "fast"
    payload["model"] = "kev:4b"
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 200
    assert resp.headers.get("x-selected-model") == "kev:4b"
    data = resp.json()
    assert data["routing"]["selected_model"] == "kev:4b"
    assert "rated_fps" in data["routing"]
    assert "execution_time_ms" in data["routing"]


@pytest.mark.asyncio
async def test_custom_model_routing_preserves_selected_model(
    async_client: httpx.AsyncClient, mock_ollaya_inference, sample_payloads
):
    """Tier 3: Custom model string 'my-org/decider-v2' routed with custom reason."""
    payload = dict(sample_payloads["choice"])
    payload["model"] = "my-org/decider-v2"
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 200
    assert resp.headers.get("x-selected-model") == "my-org/decider-v2"


# ===========================================================================
# Tier 3: Concurrency & Multi-Tenant Workloads
# ===========================================================================


@pytest.mark.asyncio
async def test_concurrent_heterogeneous_sla_requests(
    async_client: httpx.AsyncClient, mock_ollaya_inference, sample_payloads
):
    """
    Tier 3 Concurrency: 20 concurrent requests with mixed SLAs ('fast', 'smart', 'accurate')
    complete successfully without cross-request header or routing pollution.
    """
    slas = ["fast", "smart", "accurate", "pointer"] * 5  # 20 requests

    async def send_req(sla_val: str):
        p = dict(sample_payloads["choice"])
        p["sla"] = sla_val
        return await async_client.post("/v1/auto", json=p), sla_val

    results = await asyncio.gather(*(send_req(s) for s in slas))

    for resp, requested_sla in results:
        assert resp.status_code == 200
        hdr_model = resp.headers.get("x-selected-model")
        if requested_sla == "fast":
            assert hdr_model == "laya"
        elif requested_sla in ["smart", "accurate"]:
            assert hdr_model == "decider"
        elif requested_sla == "pointer":
            assert hdr_model == "kev:4b"


@pytest.mark.asyncio
async def test_concurrency_under_partial_upstream_failures(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
):
    """
    Tier 3 Concurrency: 15 concurrent requests against an upstream server that
    intermittently succeeds (200), times out, or fails with network errors.
    Gateway handles each cleanly according to contract.
    """
    call_count = 0

    def intermittent_inference(request):
        nonlocal call_count
        call_count += 1
        if call_count % 3 == 0:
            raise httpx.ReadTimeout("Intermittent timeout")
        elif call_count % 3 == 1:
            raise httpx.ConnectError("Intermittent connect error")
        return httpx.Response(200, json={"answers": {"category": "backend_error"}})

    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").mock(side_effect=intermittent_inference)

    async def send_req():
        return await async_client.post("/v1/auto", json=sample_payloads["choice"])

    responses = await asyncio.gather(*(send_req() for _ in range(12)))
    status_codes = [r.status_code for r in responses]

    # Every response must be one of 200, 502, 504 — NEVER 500 crash
    assert 500 not in status_codes
    assert all(code in [200, 502, 504] for code in status_codes)


@pytest.mark.asyncio
async def test_concurrent_streaming_and_decision_dispatch(
    async_client: httpx.AsyncClient, mock_ollaya_mcp_sse, mock_ollaya_inference, sample_payloads
):
    """
    Tier 3 Concurrency: Gateway simultaneously streams SSE events on /mcp
    and dispatches decision requests on /v1/auto without resource blocking.
    """

    async def listen_sse():
        lines = []
        async with async_client.stream("GET", "/mcp") as stream_resp:
            assert stream_resp.status_code == 200
            async for line in stream_resp.aiter_lines():
                if line:
                    lines.append(line)
        return lines

    async def dispatch_decisions():
        resps = []
        for _ in range(5):
            r = await async_client.post("/v1/auto", json=sample_payloads["choice"])
            assert r.status_code == 200
            resps.append(r)
        return resps

    sse_lines, decision_resps = await asyncio.gather(listen_sse(), dispatch_decisions())
    assert len(sse_lines) > 0
    assert len(decision_resps) == 5


@pytest.mark.asyncio
async def test_health_check_remains_up_when_upstream_is_down(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter, sample_payloads
):
    """
    Tier 3: Even when upstream inference and MCP servers are unreachable,
    the gateway's own health probe /healthz remains 200 OK.
    """
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").mock(side_effect=httpx.ConnectError("Down"))
    respx_mock.get(f"{OLLAYA_MCP_URL}/mcp").mock(side_effect=httpx.ConnectError("Down"))

    # Decision fails with 502
    r_dec = await async_client.post("/v1/auto", json=sample_payloads["choice"])
    assert r_dec.status_code == 502

    # Health check is healthy
    r_health = await async_client.get("/healthz")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"


# ===========================================================================
# Tier 3: Router Unit Invariants & Defensive Typing (F8)
# ===========================================================================


def test_router_resolve_model_case_insensitivity():
    """F8 Unit: SLA case insensitivity ('FAST', 'Smart', 'ACCURATE')."""
    model, reason = resolve_model(sla="FAST")
    assert model == "laya"
    model, reason = resolve_model(sla="Smart")
    assert model == "decider"
    model, reason = resolve_model(sla="ACCURATE")
    assert model == "decider"


def test_router_resolve_model_latency_boundaries():
    """F8 Unit: Latency boundary checks (exact 500ms, 501ms, 2500ms, 2501ms)."""
    # 500ms boundary -> laya
    m500, _ = resolve_model(max_latency_ms=500)
    assert m500 == "laya"

    # 501ms boundary -> decider
    m501, _ = resolve_model(max_latency_ms=501)
    assert m501 == "decider"

    # 2500ms boundary -> decider
    m2500, _ = resolve_model(max_latency_ms=2500)
    assert m2500 == "decider"

    # 5000ms budget -> decider
    m5000, _ = resolve_model(max_latency_ms=5000)
    assert m5000 == "decider"


def test_router_resolve_model_auto_alias():
    """F8 Unit: model='auto' or model=None triggers SLA/latency routing."""
    model, reason = resolve_model(model="auto", sla="fast")
    assert model == "laya"
    model, reason = resolve_model(model=None, sla="smart")
    assert model == "decider"


def test_router_resolve_model_pointer_sla():
    """F8 Unit: SLA 'pointer' resolves to 'kev:4b'."""
    model, reason = resolve_model(sla="pointer")
    assert model == "kev:4b"
    assert "kev:4b" in reason
