"""
E2E Capabilities Resilience Test Suite (Feature F3, Tiers 1-2).
Tests dynamic capabilities discovery, model catalog listing, tool manifests,
and resilience against corrupt upstream metadata (e.g. 'details': None, null sizes,
empty catalogs, and upstream communication failures).
"""

import sys
import os
import pytest
import httpx
import respx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.config import OLLAYA_URL

# ===========================================================================
# Tier 1: Core Feature Capabilities Coverage (F3)
# ===========================================================================


@pytest.mark.asyncio
async def test_get_capabilities_manifest_normal(async_client: httpx.AsyncClient, mock_ollaya_tags):
    """F3: GET /v1/capabilities returns complete manifest with models, SLA profiles,
    and device info."""
    resp = await async_client.get("/v1/capabilities")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "service" in data
    assert "device" in data
    assert "models" in data
    assert "sla_profiles" in data
    assert "endpoints" in data
    assert "decider" in data["models"]
    assert "laya" in data["models"]


@pytest.mark.asyncio
async def test_root_endpoint_aliases_capabilities(
    async_client: httpx.AsyncClient, mock_ollaya_tags
):
    """F3: GET / is an alias for GET /v1/capabilities."""
    resp_root = await async_client.get("/")
    resp_cap = await async_client.get("/v1/capabilities")
    assert resp_root.status_code == 200
    assert resp_cap.status_code == 200
    assert resp_root.json() == resp_cap.json()


@pytest.mark.asyncio
async def test_list_models_normal(async_client: httpx.AsyncClient, mock_ollaya_tags):
    """F3: GET /v1/models returns OpenAI-compatible list of models."""
    resp = await async_client.get("/v1/models")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["object"] == "list"
    assert "data" in data
    assert len(data["data"]) > 0
    model_ids = [m["id"] for m in data["data"]]
    assert "decider" in model_ids
    assert "laya" in model_ids


@pytest.mark.asyncio
async def test_list_models_fields_contract(async_client: httpx.AsyncClient, mock_ollaya_tags):
    """F3: Each model in GET /v1/models contains required performance & tier fields."""
    resp = await async_client.get("/v1/models")
    assert resp.status_code == 200
    for model_item in resp.json()["data"]:
        assert "id" in model_item
        assert "name" in model_item
        assert "latency_ms" in model_item
        assert "throughput_fps" in model_item
        assert "reasoning_tier" in model_item
        assert "installed" in model_item
        assert isinstance(model_item["latency_ms"], (int, float))
        assert isinstance(model_item["throughput_fps"], (int, float))


@pytest.mark.asyncio
async def test_agent_tools_schema_contract(async_client: httpx.AsyncClient):
    """F3: GET /v1/tools returns OpenAI/Anthropic function calling tool definitions."""
    resp = await async_client.get("/v1/tools")
    assert resp.status_code == 200
    data = resp.json()
    assert "tools" in data
    assert len(data["tools"]) >= 1
    tool = data["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "system_one_decision"
    props = tool["function"]["parameters"]["properties"]
    assert "state" in props
    assert "questions" in props
    assert "sla" in props
    assert "model" in props
    assert "state" in tool["function"]["parameters"]["required"]
    assert "questions" in tool["function"]["parameters"]["required"]


@pytest.mark.asyncio
async def test_sla_profiles_manifest_contract(async_client: httpx.AsyncClient, mock_ollaya_tags):
    """F3: /v1/capabilities manifest contains SLA routing profiles with target models."""
    resp = await async_client.get("/v1/capabilities")
    assert resp.status_code == 200
    profiles = resp.json()["sla_profiles"]
    assert "fast" in profiles
    assert "smart" in profiles
    assert "accurate" in profiles
    assert "pointer" in profiles
    assert profiles["fast"]["target_model"] == "laya"
    assert profiles["smart"]["target_model"] == "decider"
    assert profiles["accurate"]["target_model"] == "decider"
    assert profiles["pointer"]["target_model"] == "kev:4b"


@pytest.mark.asyncio
async def test_device_info_contract(async_client: httpx.AsyncClient, mock_ollaya_tags):
    """F3: Capabilities manifest advertises hardware device info."""
    resp = await async_client.get("/v1/capabilities")
    assert resp.status_code == 200
    device = resp.json()["device"]
    assert "name" in device and len(device["name"]) > 0
    assert device["memory_total_gb"] == 64
    assert "Ampere" in device["gpu"]
    assert device["cuda_version"] == "12.6"


@pytest.mark.asyncio
async def test_capabilities_fallback_on_upstream_connect_error(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F3: When upstream /api/tags has ConnectError, fallback to cached capabilities and
    return 200."""
    respx_mock.get(f"{OLLAYA_URL}/api/tags").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    resp = await async_client.get("/v1/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "decider" in data["models"]
    assert "laya" in data["models"]


@pytest.mark.asyncio
async def test_capabilities_fallback_on_upstream_timeout(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F3: When upstream /api/tags times out, fallback to cached capabilities and return 200."""
    respx_mock.get(f"{OLLAYA_URL}/api/tags").mock(
        side_effect=httpx.ReadTimeout("Timeout reading tags")
    )
    resp = await async_client.get("/v1/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) > 0


@pytest.mark.asyncio
async def test_capabilities_fallback_on_upstream_500(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F3: When upstream /api/tags returns 500, fallback to cached capabilities and return 200."""
    respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
        status_code=500, json={"error": "Database error in tag catalog"}
    )
    resp = await async_client.get("/v1/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "decider" in data["models"]


# ===========================================================================
# Tier 2: Boundary, Adversarial & Partial Failure Cases (F3)
# ===========================================================================


@pytest.mark.asyncio
async def test_capabilities_isolated_on_details_null(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """
    F3 Adversarial: Upstream model with 'details': None (null in json) must NOT crash
    the gateway with AttributeError. Corrupt model is isolated and valid models returned.
    """
    corrupt_tags = {
        "models": [
            {
                "name": "broken_model:latest",
                "size": 1024000,
                "modified_at": "2026-09-01T12:00:00Z",
                "details": None,
            },
            {
                "name": "decider:latest",
                "size": 3984588800,
                "modified_at": "2026-09-01T12:00:00Z",
                "details": {"format": "onnx", "parameter_size": "1.9B"},
            },
        ]
    }
    respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(status_code=200, json=corrupt_tags)
    resp = await async_client.get("/v1/capabilities")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    models = resp.json()["models"]
    assert "decider" in models


@pytest.mark.asyncio
async def test_list_models_isolated_on_details_null(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F3 Adversarial: GET /v1/models does not crash when upstream model has 'details': None."""
    corrupt_tags = {
        "models": [
            {"name": "corrupt:latest", "size": 512, "details": None},
            {
                "name": "laya:latest",
                "size": 891289600,
                "details": {"format": "onnx", "parameter_size": "421M"},
            },
        ]
    }
    respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(status_code=200, json=corrupt_tags)
    resp = await async_client.get("/v1/models")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    model_ids = [m["id"] for m in resp.json()["data"]]
    assert "laya" in model_ids


@pytest.mark.asyncio
async def test_capabilities_isolated_on_missing_details_key(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F3 Boundary: Upstream model completely lacking 'details' key is safely handled."""
    tags = {
        "models": [
            {
                "name": "simple_model:latest",
                "size": 2048000,
                # 'details' key omitted entirely
            }
        ]
    }
    respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(status_code=200, json=tags)
    resp = await async_client.get("/v1/capabilities")
    assert resp.status_code == 200
    assert "models" in resp.json()


@pytest.mark.asyncio
async def test_capabilities_isolated_on_null_size(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F3 Boundary: Upstream model with 'size': None is safely handled without TypeError."""
    tags = {
        "models": [{"name": "null_size_model:latest", "size": None, "details": {"format": "onnx"}}]
    }
    respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(status_code=200, json=tags)
    resp = await async_client.get("/v1/capabilities")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_capabilities_custom_model_dynamic_discovery(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F3: A custom model installed on upstream is dynamically discovered and advertised."""
    custom_tags = {
        "models": [
            {
                "name": "custom_enterprise_model:latest",
                "size": 5242880000,
                "modified_at": "2026-09-15T08:00:00Z",
                "details": {"format": "onnx", "parameter_size": "7B"},
            }
        ]
    }
    respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(status_code=200, json=custom_tags)
    resp = await async_client.get("/v1/capabilities")
    assert resp.status_code == 200
    models = resp.json()["models"]
    assert "custom_enterprise_model" in models
    assert models["custom_enterprise_model"]["installed"] is True


@pytest.mark.asyncio
async def test_capabilities_fallback_on_empty_models_list(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F3 Boundary: When upstream returns empty list {'models': []}, fallback to static catalog."""
    respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(status_code=200, json={"models": []})
    resp = await async_client.get("/v1/capabilities")
    assert resp.status_code == 200
    models = resp.json()["models"]
    assert "decider" in models
    assert "laya" in models


@pytest.mark.asyncio
async def test_capabilities_malformed_json_from_upstream(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """F3 Adversarial: Upstream returns non-JSON garbage, gateway falls back to static catalog."""
    respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
        status_code=200,
        headers={"content-type": "application/json"},
        content=b"{corrupt json syntax: 123",
    )
    resp = await async_client.get("/v1/capabilities")
    assert resp.status_code == 200
    models = resp.json()["models"]
    assert "decider" in models
