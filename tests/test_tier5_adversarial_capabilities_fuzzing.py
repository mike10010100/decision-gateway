"""
Tier 5 White-Box Adversarial Hardening Test Suite: Capabilities Catalog & Fuzzing.
Integrated from challenger_m5_iter2_1 for Milestone M5 Iteration 3.
Target: tests/test_tier5_adversarial_capabilities_fuzzing.py

Requirements challenged:
1. Fuzz model names:
   - Non-printable characters (\\x00, \\x01, \\x02, \\x1b, \\x7f, etc.)
   - RTL overrides (\\u202e)
   - Zero-width joiners (\\u200d)
   - Multiple consecutive colons (::::)
   - Colons at start / middle / end (:start, end:, :start:end:)
   - Empty strings ("", "   ", "\\t\\n\\r")
   - Null values (None)
   - Huge strings (10,000 chars)
2. Upstream returns 100% corrupt models:
   - Confirm static benchmark models ('decider', 'laya') are available and returned.
3. Upstream returns mixed models (10 corrupt + 10 valid):
   - Confirm exactly 10 valid models are returned and 0 corrupt models leak into /v1/models.
4. Upstream returns 500, network error, or invalid JSON:
   - Confirm fallback catalog is returned.
5. Downstream routing:
   - Verify DecisionRequest model validation accepts valid cleaned IDs
   - Verify DecisionRequest cleanly rejects empty/corrupt model names with 400 Bad Request.
"""

import sys
import os
import pytest
import httpx
import respx

# Ensure project root is in python path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if not os.path.exists(os.path.join(_project_root, "app")):
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _project_root)

from app.main import app  # noqa: E402
from app.config import OLLAYA_URL  # noqa: E402
from app.capabilities import get_live_models_capabilities, MODEL_CAPABILITIES  # noqa: E402
from app.models import DecisionRequest  # noqa: E402

# ---------------------------------------------------------------------------
# Test Fixtures & Data Definitions
# ---------------------------------------------------------------------------

TEN_VALID_MODELS = [
    {
        "name": f"valid_model_{i}:latest",
        "size": (i + 1) * 1024 * 1024,
        "details": {"format": "onnx", "parameter_size": f"{i+1}B"},
    }
    for i in range(10)
]

TEN_CORRUPT_MODELS = [
    {"name": "\x00bad\x01"},  # non-printable embedded
    {"name": "\u202ebadmodel"},  # RTL override embedded
    {"name": "bad\u200dmodel"},  # ZWJ embedded
    {"name": "model::::tag"},  # multiple consecutive colons
    {"name": ":colonstart"},  # colon at start
    {"name": "colonend:"},  # colon at end
    {"name": "::::"},  # only colons
    {"name": "   \t\n   "},  # whitespace only
    {"name": None},  # null value
    {"name": "a" * 10000},  # huge string (10,000 chars)
]

TEN_PURE_CORRUPT_MODELS = [
    {"name": "\x00\x01\x02"},  # non-printable only
    {"name": "\u202e"},  # RTL override only
    {"name": "\u200d"},  # ZWJ only
    {"name": "::::"},  # multiple consecutive colons only
    {"name": ":leading"},  # colon at start
    {"name": "trailing:"},  # colon at end
    {"name": ""},  # empty string
    {"name": "   "},  # whitespace
    {"name": None},  # null value
    {"name": ":"},  # single colon
]


@pytest.fixture
def test_client():
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ===========================================================================
# 1. Pure Corrupt Model Names Fuzzing & Isolation (Baseline verification)
# ===========================================================================


class TestPureCorruptModelFuzzing:
    """Tests that purely corrupt model strings without alphanumeric base IDs are rejected."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "corrupt_item",
        [
            {"name": "\x00\x01\x02"},
            {"name": "\u202e"},
            {"name": "\u200d"},
            {"name": "::::"},
            {"name": ":colonstart"},
            {"name": "colonend:"},
            {"name": ""},
            {"name": "   "},
            {"name": "\t\t\n\r"},
            {"name": None},
            {"name": ":"},
            {"name": "::"},
        ],
    )
    async def test_unit_pure_corrupt_item_triggers_static_fallback(
        self, corrupt_item, respx_mock: respx.MockRouter
    ):
        """When upstream returns only a single pure corrupt model item,
        static fallback is triggered."""
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": [corrupt_item]}
        )
        catalog = await get_live_models_capabilities(OLLAYA_URL)

        # Must fall back to static benchmark models
        assert "decider" in catalog
        assert "laya" in catalog

        # Corrupt keys must NOT be in catalog
        raw = corrupt_item.get("name")
        if raw is not None:
            assert raw not in catalog
            assert raw.strip() not in catalog


# ===========================================================================
# 2. 100% Corrupt Models Upstream Fallback
# ===========================================================================


class Test100PercentCorruptUpstream:
    """Tests behavior when upstream returns 100% corrupt models."""

    @pytest.mark.asyncio
    async def test_unit_100_percent_pure_corrupt_catalog_falls_back(
        self, respx_mock: respx.MockRouter
    ):
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": TEN_PURE_CORRUPT_MODELS}
        )
        catalog = await get_live_models_capabilities(OLLAYA_URL)
        assert "decider" in catalog
        assert "laya" in catalog
        assert set(catalog.keys()) == set(MODEL_CAPABILITIES.keys())

    @pytest.mark.asyncio
    async def test_api_100_percent_pure_corrupt_models_endpoint(
        self, test_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": TEN_PURE_CORRUPT_MODELS}
        )
        async with test_client as client:
            resp = await client.get("/v1/models")
            assert resp.status_code == 200
            data = resp.json().get("data", [])
            ids = [m["id"] for m in data]
            assert "decider" in ids
            assert "laya" in ids
            for item in TEN_PURE_CORRUPT_MODELS:
                name = item.get("name")
                if name:
                    assert name not in ids
                    assert name.strip() not in ids

    @pytest.mark.asyncio
    async def test_unit_100_percent_fuzz_corrupt_catalog_fallback_adversarial_gap(
        self, respx_mock: respx.MockRouter
    ):
        """
        Adversarial test: Upstream returns 100% corrupt models, but containing embedded
        non-printable chars, RTL overrides, ZWJ, multiple colons, or huge 10,000 char strings.
        Requirement: Static benchmark models ('decider', 'laya') must be available and returned.
        """
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": TEN_CORRUPT_MODELS}
        )
        catalog = await get_live_models_capabilities(OLLAYA_URL)
        assert (
            "decider" in catalog
        ), "100% corrupt catalog with fuzzed models failed to fall back to 'decider'!"
        assert (
            "laya" in catalog
        ), "100% corrupt catalog with fuzzed models failed to fall back to 'laya'!"


# ===========================================================================
# 3. Upstream Failure / Fallback (500, Network Error, Invalid JSON)
# ===========================================================================


class TestUpstreamFailuresFallback:
    """Tests catalog fallback when upstream fails (500, network error, invalid JSON)."""

    @pytest.mark.asyncio
    async def test_upstream_500_internal_server_error(
        self, test_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=500, text="Internal Server Error"
        )
        async with test_client as client:
            resp = await client.get("/v1/models")
            assert resp.status_code == 200
            ids = [m["id"] for m in resp.json().get("data", [])]
            assert "decider" in ids
            assert "laya" in ids

    @pytest.mark.asyncio
    async def test_upstream_network_connect_error(
        self, test_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        respx_mock.get(f"{OLLAYA_URL}/api/tags").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        async with test_client as client:
            resp = await client.get("/v1/models")
            assert resp.status_code == 200
            ids = [m["id"] for m in resp.json().get("data", [])]
            assert "decider" in ids
            assert "laya" in ids

    @pytest.mark.asyncio
    async def test_upstream_network_read_timeout(
        self, test_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        respx_mock.get(f"{OLLAYA_URL}/api/tags").mock(
            side_effect=httpx.ReadTimeout("Read timed out")
        )
        async with test_client as client:
            resp = await client.get("/v1/models")
            assert resp.status_code == 200
            ids = [m["id"] for m in resp.json().get("data", [])]
            assert "decider" in ids
            assert "laya" in ids

    @pytest.mark.asyncio
    async def test_upstream_invalid_json_html(
        self, test_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200,
            text="<html><body>Bad Gateway</body></html>",
            headers={"content-type": "text/html"},
        )
        async with test_client as client:
            resp = await client.get("/v1/models")
            assert resp.status_code == 200
            ids = [m["id"] for m in resp.json().get("data", [])]
            assert "decider" in ids
            assert "laya" in ids

    @pytest.mark.asyncio
    async def test_upstream_invalid_json_not_a_dict(
        self, test_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(status_code=200, json=["not", "a", "dict"])
        async with test_client as client:
            resp = await client.get("/v1/models")
            assert resp.status_code == 200
            ids = [m["id"] for m in resp.json().get("data", [])]
            assert "decider" in ids
            assert "laya" in ids


# ===========================================================================
# 4. Mixed Catalog Stress: 10 Corrupt + 10 Valid Models
# ===========================================================================


class TestMixedCatalogAdversarial:
    """
    Adversarial test: Upstream returns 10 corrupt models + 10 valid models.
    Requirement:
      Confirm exactly 10 valid models are returned and 0 corrupt models leak into /v1/models.
    """

    @pytest.mark.asyncio
    async def test_mixed_catalog_isolation_unit(self, respx_mock: respx.MockRouter):
        """Asserts that in get_live_models_capabilities, 0 corrupt models leak."""
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": TEN_CORRUPT_MODELS + TEN_VALID_MODELS}
        )
        catalog = await get_live_models_capabilities(OLLAYA_URL)

        # 1. All valid models must be present
        for v in TEN_VALID_MODELS:
            clean_base = v["name"].split(":")[0]
            assert clean_base in catalog, f"Valid model {clean_base} missing from catalog!"

        # 2. Corrupt models must NOT leak into catalog
        leaked = []
        for c in TEN_CORRUPT_MODELS:
            cname = c.get("name")
            if cname:
                clean_base = cname.split(":")[0].strip()
                if clean_base in catalog or cname in catalog:
                    leaked.append(cname[:30])

        assert not leaked, f"Corrupt models leaked into capabilities catalog: {leaked}"

        # 3. Exactly 10 valid base models must be present
        base_ids = {v.get("alias", k.split(":")[0]) for k, v in catalog.items()}
        assert (
            len(base_ids) == 10
        ), f"Expected exactly 10 base models, got {len(base_ids)}: {base_ids}"

    @pytest.mark.asyncio
    async def test_mixed_catalog_models_endpoint_leakage(
        self, test_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """GET /v1/models must return exactly the 10 valid models and 0 corrupt models."""
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": TEN_CORRUPT_MODELS + TEN_VALID_MODELS}
        )
        async with test_client as client:
            resp = await client.get("/v1/models")
            assert resp.status_code == 200
            data = resp.json().get("data", [])
            model_ids = [m["id"] for m in data]

            # All 10 valid models must be present
            for v in TEN_VALID_MODELS:
                clean_base = v["name"].split(":")[0]
                assert clean_base in model_ids, f"Valid model {clean_base} missing from /v1/models!"

            # 0 corrupt models must leak into /v1/models
            leaked = []
            for c in TEN_CORRUPT_MODELS:
                cname = c.get("name")
                if cname:
                    clean_base = cname.split(":")[0].strip()
                    if clean_base in model_ids or cname in model_ids:
                        leaked.append(cname[:30])

            assert not leaked, f"Corrupt models leaked into /v1/models: {leaked}"


# ===========================================================================
# 5. Downstream Routing & DecisionRequest Model Validation
# ===========================================================================


class TestDownstreamDecisionRequestValidation:
    """
    Tests downstream routing:
    - Verify DecisionRequest model validation accepts valid cleaned IDs
    - Cleanly rejects empty/corrupt model names with 400 Bad Request.
    """

    def test_decision_request_accepts_valid_cleaned_ids(self):
        """DecisionRequest model validation must accept valid cleaned IDs."""
        valid_ids = [
            "decider",
            "laya",
            "valid_model_1",
            "alpha:latest",
            "auto",
            "qwen-2.5-7b",
            "model-name.v1",
            None,
        ]
        for mid in valid_ids:
            req = DecisionRequest(
                state="Valid evaluation state document.",
                questions={"q1": {"type": "noul"}},
                model=mid,
            )
            if mid is None:
                assert req.model is None
            else:
                assert req.model == mid.strip()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "empty_or_whitespace_model",
        [
            "",
            "   ",
            "\t\t\n",
            " \r\n ",
        ],
    )
    async def test_decision_request_rejects_empty_whitespace_model_with_400(
        self, empty_or_whitespace_model, test_client: httpx.AsyncClient
    ):
        """DecisionRequest cleanly rejects empty/whitespace model names with 400 Bad Request."""
        async with test_client as client:
            resp = await client.post(
                "/v1/auto",
                json={
                    "state": "Sample document",
                    "questions": {"q1": {"type": "noul"}},
                    "model": empty_or_whitespace_model,
                },
            )
            assert resp.status_code == 400
            err_data = resp.json()
            assert "detail" in err_data
            errors = err_data.get("errors", [])
            assert any("model" in str(e.get("loc", [])) for e in errors)

    @pytest.mark.asyncio
    async def test_decision_request_rejects_huge_model_string_with_400(
        self, test_client: httpx.AsyncClient
    ):
        """DecisionRequest cleanly rejects 10,000 character model name with 400 Bad Request."""
        async with test_client as client:
            resp = await client.post(
                "/v1/auto",
                json={
                    "state": "Sample document",
                    "questions": {"q1": {"type": "noul"}},
                    "model": "a" * 10000,
                },
            )
            assert resp.status_code == 400
            err_data = resp.json()
            assert "detail" in err_data
            errors = err_data.get("errors", [])
            assert any("model" in str(e.get("loc", [])) for e in errors)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "invalid_type_model", [12345, True, False, ["model"], {"name": "model"}]
    )
    async def test_decision_request_rejects_non_string_model_with_400(
        self, invalid_type_model, test_client: httpx.AsyncClient
    ):
        """DecisionRequest cleanly rejects non-string model types with 400 Bad Request."""
        async with test_client as client:
            resp = await client.post(
                "/v1/auto",
                json={
                    "state": "Sample document",
                    "questions": {"q1": {"type": "noul"}},
                    "model": invalid_type_model,
                },
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "corrupt_model",
        [
            "::::",
            ":colonstart",
            "colonend:",
            "\u202e",
            "\u200d",
            "\x00\x01\x02",
            "\u202ebadmodel",
            "bad\u200dmodel",
            "model::::tag",
        ],
    )
    async def test_decision_request_corrupt_model_validation(
        self, corrupt_model, test_client: httpx.AsyncClient
    ):
        """
        Adversarial test: Tests whether DecisionRequest model validation cleanly rejects
        corrupt model names (colons, RTL overrides, ZWJ, non-printable characters)
        with 400 Bad Request.
        """
        async with test_client as client:
            resp = await client.post(
                "/v1/auto",
                json={
                    "state": "Sample document",
                    "questions": {"q1": {"type": "noul"}},
                    "model": corrupt_model,
                },
            )
            # Must reject with HTTP 400 Bad Request before proxying to upstream
            assert resp.status_code == 400, (
                f"DecisionRequest accepted corrupt model name {repr(corrupt_model)} "
                f"with status {resp.status_code}! Body: {resp.text}"
            )


# ===========================================================================
# 6. Advertisement vs Ingestion Contract Discrepancy
# ===========================================================================


class TestAdvertisementVsIngestionDiscrepancy:
    """Tests contract consistency between /v1/models advertisement and /v1/auto ingestion."""

    @pytest.mark.asyncio
    async def test_advertised_huge_model_id_rejected_by_decision_request(
        self, test_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """
        Adversarial test: Upstream advertises a 10,000 char model name.
        /v1/models accepts and advertises it, but /v1/auto rejects its own advertised ID!
        """
        huge_name = "huge_model_" + ("x" * 9980)
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": [{"name": huge_name, "size": 1000000}]}
        )
        async with test_client as client:
            resp_models = await client.get("/v1/models")
            assert resp_models.status_code == 200
            data = resp_models.json().get("data", [])
            advertised_ids = [m["id"] for m in data]

            # Hardened contract: The huge corrupt ID must NOT be advertised by /v1/models
            assert (
                huge_name not in advertised_ids
            ), "Huge model ID leaked into /v1/models advertisement"

            # When client consumes the advertised ID in /v1/auto:
            resp_auto = await client.post(
                "/v1/auto",
                json={
                    "state": "Sample document",
                    "questions": {"q1": {"type": "noul"}},
                    "model": huge_name,
                },
            )
            # Fails with 400 Bad Request because DecisionRequest limits model length to 128
            assert resp_auto.status_code == 400, "Expected /v1/auto to reject huge model name"
