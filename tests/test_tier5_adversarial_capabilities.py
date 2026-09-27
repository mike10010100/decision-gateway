"""
Tier 5 White-Box Adversarial Hardening Test Suite: Capabilities Catalog.
Integrated from challenger_m5_3 for Milestone M5 Iteration 2.

Focus areas:
1. Extreme whitespace permutations: {"name": "   "}, {"name": "\t\t\n"}, {"name": "\u200b"},
   {"name": "  :tag  "}, {"name": "decider:   "}, {"name": ":decider"}.
2. 100% corrupt models catalog -> verify default static fallback models ('decider', 'laya')
   are returned and /v1/models does not return empty or corrupt IDs.
3. Mixed upstream catalog: 5 valid models + 5 corrupt models -> verify only valid models
   are returned and corrupt items are strictly isolated.
4. Endpoint contracts: GET /v1/models and GET /v1/capabilities return 200 with valid,
   routable model IDs.
"""

import sys
import os
import pytest
import httpx
import respx

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app  # noqa: E402
from app.config import OLLAYA_URL  # noqa: E402
from app.capabilities import get_live_models_capabilities, MODEL_CAPABILITIES  # noqa: E402

# ---------------------------------------------------------------------------
# Test Fixtures & Utilities
# ---------------------------------------------------------------------------

EXTREME_WHITESPACE_CORRUPT_MODELS = [
    {"name": "   "},
    {"name": "\t\t\n"},
    {"name": "\u200b"},  # Unicode zero-width space U+200B
    {"name": "  :tag  "},  # Whitespace padded leading colon -> clean_id is ""
    {"name": ":decider"},  # Leading colon -> clean_id is ""
]

ALL_WHITESPACE_PERMUTATIONS = [
    {"name": "   "},
    {"name": "\t\t\n"},
    {"name": "\u200b"},  # Unicode zero-width space U+200B
    {"name": "  :tag  "},  # clean_id is ""
    {"name": "decider:   "},  # trailing colon in clean_name
    {"name": ":decider"},  # clean_id is ""
]

FIVE_VALID_MODELS = [
    {
        "name": "alpha:latest",
        "size": 1000000,
        "details": {"format": "onnx", "parameter_size": "1B"},
    },
    {
        "name": "beta:latest",
        "size": 2000000,
        "details": {"format": "onnx", "parameter_size": "2B"},
    },
    {
        "name": "gamma:latest",
        "size": 3000000,
        "details": {"format": "onnx", "parameter_size": "3B"},
    },
    {
        "name": "delta:latest",
        "size": 4000000,
        "details": {"format": "onnx", "parameter_size": "4B"},
    },
    {
        "name": "epsilon:latest",
        "size": 5000000,
        "details": {"format": "onnx", "parameter_size": "5B"},
    },
]


@pytest.fixture
def test_client():
    """Asynchronous client fixture with ASGITransport."""
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ===========================================================================
# 1. Extreme Whitespace Permutations & Empty ID Stress Tests
# ===========================================================================


class TestWhitespacePermutationsAdversarial:
    """Stress tests extreme whitespace permutations and invalid tag splits."""

    @pytest.mark.asyncio
    async def test_unit_whitespace_permutations_not_registered_in_catalog(
        self, respx_mock: respx.MockRouter
    ):
        """
        Adversarial: Model entries with whitespace, zero-width space, or empty clean_id
        ('   ', '\t\t\n', '\u200b', '  :tag  ', ':decider', 'decider:   ')
        must NOT produce empty string IDs or corrupt keys in catalog.
        """
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": ALL_WHITESPACE_PERMUTATIONS}
        )
        result = await get_live_models_capabilities(OLLAYA_URL)

        # 1. Empty string must NEVER be a catalog key
        assert "" not in result, "Corrupt model resulted in empty string '' key in catalog!"

        # 2. Leading-colon identifiers must NEVER be catalog keys
        assert ":tag" not in result, "Corrupt model ':tag' with missing name was registered!"
        assert (
            ":decider" not in result
        ), "Corrupt model ':decider' with missing name was registered!"

        # 3. Zero-width space U+200B must NEVER be a catalog key
        assert "\u200b" not in result, "Zero-width space model ID '\u200b' was not filtered out!"

        # 4. Trailing colon must not be in model keys
        assert "decider:" not in result, "Model key 'decider:' with trailing colon was registered!"

    @pytest.mark.asyncio
    async def test_api_models_endpoint_does_not_advertise_empty_or_corrupt_ids(
        self, test_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """
        Adversarial: GET /v1/models must never advertise empty string IDs or corrupt IDs
        when upstream returns whitespace permutations.
        """
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": ALL_WHITESPACE_PERMUTATIONS}
        )
        async with test_client as client:
            resp = await client.get("/v1/models")
            assert resp.status_code == 200
            data = resp.json().get("data", [])
            model_ids = [m["id"] for m in data]

            assert "" not in model_ids, "GET /v1/models advertised an empty string '' model ID!"
            assert (
                "\u200b" not in model_ids
            ), "GET /v1/models advertised zero-width space '\u200b' model ID!"
            assert ":tag" not in model_ids, "GET /v1/models advertised invalid ':tag' model ID!"
            assert (
                ":decider" not in model_ids
            ), "GET /v1/models advertised invalid ':decider' model ID!"
            assert (
                "decider:" not in model_ids
            ), "GET /v1/models advertised invalid 'decider:' model ID!"


# ===========================================================================
# 2. 100% Corrupt Models Fallback Stress Tests
# ===========================================================================


class Test100PercentCorruptFallback:
    """Stress tests fallback when upstream returns 100% corrupt models."""

    @pytest.mark.asyncio
    async def test_unit_100_percent_corrupt_models_triggers_static_fallback(
        self, respx_mock: respx.MockRouter
    ):
        """
        Adversarial: When upstream /api/tags returns 100% corrupt models,
        the catalog must reject all corrupt items, detect that installed_models is empty,
        and fall back to default static models (decider, laya, etc.).
        """
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": EXTREME_WHITESPACE_CORRUPT_MODELS}
        )
        result = await get_live_models_capabilities(OLLAYA_URL)

        # Fallback to static models must occur
        assert (
            "decider" in result
        ), "Fallback to default 'decider' model failed on 100% corrupt catalog!"
        assert "laya" in result, "Fallback to default 'laya' model failed on 100% corrupt catalog!"
        assert len(result) >= len(
            MODEL_CAPABILITIES
        ), "Fallback did not populate all static models!"

        # No corrupt keys must be present
        assert "" not in result, "Corrupt empty string ID present in fallback result!"
        assert "\u200b" not in result, "Zero-width space ID present in fallback result!"
        assert ":tag" not in result, "Corrupt ':tag' present in fallback result!"
        assert ":decider" not in result, "Corrupt ':decider' present in fallback result!"

    @pytest.mark.asyncio
    async def test_api_100_percent_corrupt_models_models_endpoint(
        self, test_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """
        Adversarial: GET /v1/models with 100% corrupt upstream models must return 200
        with default fallback models and NO empty or corrupt IDs.
        """
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": EXTREME_WHITESPACE_CORRUPT_MODELS}
        )
        async with test_client as client:
            resp = await client.get("/v1/models")
            assert resp.status_code == 200
            data = resp.json().get("data", [])
            model_ids = [m["id"] for m in data]

            # Assert static fallbacks are returned
            assert "decider" in model_ids, "Static fallback 'decider' missing from /v1/models!"
            assert "laya" in model_ids, "Static fallback 'laya' missing from /v1/models!"

            # Assert no empty or corrupt IDs
            assert (
                "" not in model_ids
            ), "Empty string ID advertised in /v1/models on 100% corrupt catalog!"
            assert "\u200b" not in model_ids, "Zero-width space ID advertised in /v1/models!"
            assert ":tag" not in model_ids, "Corrupt ID ':tag' advertised in /v1/models!"
            assert ":decider" not in model_ids, "Corrupt ID ':decider' advertised in /v1/models!"

    @pytest.mark.asyncio
    async def test_api_100_percent_corrupt_models_capabilities_endpoint(
        self, test_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """
        Adversarial: GET /v1/capabilities with 100% corrupt upstream models must return 200
        with static fallback models and NO empty or corrupt model keys.
        """
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": EXTREME_WHITESPACE_CORRUPT_MODELS}
        )
        async with test_client as client:
            resp = await client.get("/v1/capabilities")
            assert resp.status_code == 200
            models_dict = resp.json().get("models", {})

            assert "decider" in models_dict
            assert "laya" in models_dict
            assert "" not in models_dict
            assert "\u200b" not in models_dict
            assert ":tag" not in models_dict
            assert ":decider" not in models_dict


# ===========================================================================
# 3. Mixed Upstream Catalog (5 Valid + 5 Corrupt Models)
# ===========================================================================


class TestMixedUpstreamCatalog:
    """Stress tests mixed upstream catalog: 5 valid + 5 corrupt models."""

    @pytest.mark.asyncio
    async def test_unit_mixed_catalog_returns_only_valid_models(self, respx_mock: respx.MockRouter):
        """
        Adversarial: Given 5 valid models + 5 corrupt models, assert only the 5 valid
        models (and their clean full-name aliases) are returned, and corrupt models
        are strictly isolated and excluded.
        """
        mixed_payload = FIVE_VALID_MODELS + EXTREME_WHITESPACE_CORRUPT_MODELS
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": mixed_payload}
        )
        result = await get_live_models_capabilities(OLLAYA_URL)

        # Valid models must be present
        for valid_item in FIVE_VALID_MODELS:
            clean_base = valid_item["name"].split(":")[0]
            assert clean_base in result, f"Valid model '{clean_base}' missing from catalog!"

        # Corrupt models must NOT be present
        assert "" not in result, "Empty string '' present in mixed catalog!"
        assert "\u200b" not in result, "Zero-width space '\u200b' present in mixed catalog!"
        assert ":tag" not in result, "Corrupt ':tag' present in mixed catalog!"
        assert ":decider" not in result, "Corrupt ':decider' present in mixed catalog!"

        # Total unique installed base models must equal exactly 5
        base_ids = set()
        for k, v in result.items():
            base_ids.add(v.get("alias", k.split(":")[0]))
        assert (
            len(base_ids) == 5
        ), f"Expected exactly 5 valid base models, got {len(base_ids)}: {base_ids}"

    @pytest.mark.asyncio
    async def test_api_mixed_catalog_models_endpoint(
        self, test_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """
        Adversarial: GET /v1/models with 5 valid + 5 corrupt models returns HTTP 200
        containing only valid model entries.
        """
        mixed_payload = FIVE_VALID_MODELS + EXTREME_WHITESPACE_CORRUPT_MODELS
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": mixed_payload}
        )
        async with test_client as client:
            resp = await client.get("/v1/models")
            assert resp.status_code == 200
            data = resp.json().get("data", [])
            model_ids = [m["id"] for m in data]

            # All valid models must be present
            for valid_item in FIVE_VALID_MODELS:
                clean_base = valid_item["name"].split(":")[0]
                assert clean_base in model_ids

            # Corrupt models must not be present
            assert "" not in model_ids
            assert "\u200b" not in model_ids
            assert ":tag" not in model_ids
            assert ":decider" not in model_ids


# ===========================================================================
# 4. Model Routability & Endpoint Response Robustness
# ===========================================================================


class TestModelRoutabilityAndEndpointStatus:
    """Verifies GET /v1/models and GET /v1/capabilities return 200 with valid, routable IDs."""

    @pytest.mark.asyncio
    async def test_endpoints_return_200_and_routable_ids(
        self, test_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """
        Verify GET /v1/models and GET /v1/capabilities return 200 with valid, routable model IDs.
        Every advertised model ID must be a non-empty string that passes DecisionRequest validation.
        """
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": ALL_WHITESPACE_PERMUTATIONS}
        )
        async with test_client as client:
            # 1. Verify GET /v1/models returns 200
            resp_models = await client.get("/v1/models")
            assert resp_models.status_code == 200
            models_data = resp_models.json().get("data", [])

            # 2. Verify GET /v1/capabilities returns 200
            resp_cap = await client.get("/v1/capabilities")
            assert resp_cap.status_code == 200
            cap_data = resp_cap.json().get("models", {})

            # 3. Check all model IDs from /v1/models
            for item in models_data:
                mid = item.get("id")
                assert isinstance(mid, str), f"Model id is not a string: {mid}"
                assert len(mid.strip()) > 0, f"Model id is empty or whitespace: {mid!r}"
                assert not mid.startswith(":"), f"Model id starts with colon: {mid!r}"
                assert not mid.endswith(":"), f"Model id ends with colon: {mid!r}"
                assert "\u200b" not in mid, f"Model id contains zero-width space: {mid!r}"

            # 4. Check all keys from /v1/capabilities
            for mid in cap_data.keys():
                assert isinstance(mid, str), f"Capability model key is not a string: {mid}"
                assert len(mid.strip()) > 0, f"Capability model key is empty or whitespace: {mid!r}"
                assert not mid.startswith(":"), f"Capability model key starts with colon: {mid!r}"
                assert not mid.endswith(":"), f"Capability model key ends with colon: {mid!r}"
                assert (
                    "\u200b" not in mid
                ), f"Capability model key contains zero-width space: {mid!r}"
