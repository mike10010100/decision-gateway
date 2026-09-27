"""
Tier 5 White-Box Adversarial Coverage Hardening Test Suite:
Router, Capabilities Manifest, and Question Models.

Validates:
1. Exact latency boundary conditions (1ms, 500ms, 501ms, 2500ms, 2501ms, 600000ms, 999999ms, <=0)
2. Routing under conflicting constraints (model vs sla vs max_latency_ms vs auto fallback)
3. Capabilities catalog resilience (0 models, all corrupt models, mixed formats, network errors)
4. Complex question schemas (min/max choices 2-255, min/max anchors 2-10, unicode, empty labels)
5. Malformed decision requests and edge-case question keys (bounds, unicode IDs, 1MB state)
"""

import sys
import os
import pytest
import pydantic
import httpx
import respx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.router import resolve_model
from app.capabilities import (
    MODEL_CAPABILITIES,
    SLA_ROUTING_PROFILES,
    get_live_models_capabilities,
)
from app.models import (
    ChoiceQuestion,
    ScoreQuestion,
    NoulQuestion,
    DecisionRequest,
)
from app.config import OLLAYA_URL

# ===========================================================================
# 1. Exact Latency Boundary Conditions
# ===========================================================================


class TestLatencyBoundaryConditions:
    """Verifies router resolution and endpoint validation along exact latency boundaries."""

    @pytest.mark.parametrize(
        "latency_val,expected_model,expected_substr",
        [
            (1, "laya", "1ms <= 500ms"),
            (499, "laya", "499ms <= 500ms"),
            (500, "laya", "500ms <= 500ms"),
            (501, "decider", "501ms, ~1.6s latency"),
            (2499, "decider", "2499ms, ~1.6s latency"),
            (2500, "decider", "2500ms, ~1.6s latency"),
            (2501, "decider", "within 2501ms budget"),
            (600000, "decider", "within 600000ms budget"),
            (999999, "decider", "within 999999ms budget"),
        ],
    )
    def test_unit_router_latency_exact_thresholds(
        self, latency_val, expected_model, expected_substr
    ):
        """Unit: resolve_model accurately maps boundary latency budgets to target models."""
        model, reason = resolve_model(max_latency_ms=latency_val)
        assert model == expected_model
        assert expected_substr in reason

    @pytest.mark.parametrize("invalid_val", [0, -1, -500])
    def test_unit_router_latency_non_positive_rejected(self, invalid_val):
        """Unit: resolve_model rejects non-positive latency values with ValueError."""
        with pytest.raises(ValueError, match="max_latency_ms must be greater than 0"):
            resolve_model(max_latency_ms=invalid_val)

    @pytest.mark.parametrize("invalid_type", [True, False, "100", 3.14, [500], {"ms": 500}])
    def test_unit_router_latency_invalid_type_rejected(self, invalid_type):
        """Unit: resolve_model rejects non-integer latency types with TypeError."""
        with pytest.raises(TypeError, match="max_latency_ms must be an integer"):
            resolve_model(max_latency_ms=invalid_type)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("endpoint", ["/v1/auto", "/v1/systemone"])
    async def test_api_latency_minimum_bound_1ms_routes_to_laya(
        self, async_client: httpx.AsyncClient, mock_ollaya_inference, endpoint: str
    ):
        """API: Minimum valid latency budget (1ms) routes to 'laya' with 200 OK."""
        payload = {
            "state": "High-frequency edge telemetry packet",
            "questions": {"triage": {"type": "noul"}},
            "max_latency_ms": 1,
        }
        resp = await async_client.post(endpoint, json=payload)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        assert resp.headers.get("X-Selected-Model") == "laya"
        data = resp.json()
        assert data["routing"]["selected_model"] == "laya"
        assert "1ms <= 500ms" in data["routing"]["reason"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("endpoint", ["/v1/auto", "/v1/systemone"])
    async def test_api_latency_exact_threshold_500ms_routes_to_laya(
        self, async_client: httpx.AsyncClient, mock_ollaya_inference, endpoint: str
    ):
        """API: Exact upper bound of laya tier (500ms) routes to 'laya'."""
        payload = {
            "state": "High-frequency edge telemetry packet",
            "questions": {"triage": {"type": "noul"}},
            "max_latency_ms": 500,
        }
        resp = await async_client.post(endpoint, json=payload)
        assert resp.status_code == 200
        assert resp.headers.get("X-Selected-Model") == "laya"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("endpoint", ["/v1/auto", "/v1/systemone"])
    async def test_api_latency_exact_threshold_501ms_routes_to_decider(
        self, async_client: httpx.AsyncClient, mock_ollaya_inference, endpoint: str
    ):
        """API: Exact lower bound of decider tier (501ms) transitions routing to 'decider'."""
        payload = {
            "state": "Standard decision state evaluation",
            "questions": {"triage": {"type": "noul"}},
            "max_latency_ms": 501,
        }
        resp = await async_client.post(endpoint, json=payload)
        assert resp.status_code == 200
        assert resp.headers.get("X-Selected-Model") == "decider"
        assert "501ms, ~1.6s latency" in resp.json()["routing"]["reason"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("endpoint", ["/v1/auto", "/v1/systemone"])
    async def test_api_latency_maximum_schema_bound_600000ms_routes_to_decider(
        self, async_client: httpx.AsyncClient, mock_ollaya_inference, endpoint: str
    ):
        """API: Max schema latency budget (600,000ms = 10 min) passes and routes to decider."""
        payload = {
            "state": "Long batch document analysis",
            "questions": {"triage": {"type": "noul"}},
            "max_latency_ms": 600000,
        }
        resp = await async_client.post(endpoint, json=payload)
        assert resp.status_code == 200
        assert resp.headers.get("X-Selected-Model") == "decider"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("endpoint", ["/v1/auto", "/v1/systemone"])
    async def test_api_latency_exceeds_schema_bound_999999ms_returns_400(
        self, async_client: httpx.AsyncClient, endpoint: str
    ):
        """API Adversarial: Latency budget of 999,999ms exceeds le=600000 schema bound."""
        payload = {
            "state": "State with excessively high latency budget",
            "questions": {"triage": {"type": "noul"}},
            "max_latency_ms": 999999,
        }
        resp = await async_client.post(endpoint, json=payload)
        assert resp.status_code == 400
        data = resp.json()
        assert data["detail"] == "Validation error"
        assert any("max_latency_ms" in str(err["loc"]) for err in data["errors"])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_val", [0, -1, -500, True, "500ms"])
    async def test_api_latency_invalid_values_return_400(
        self, async_client: httpx.AsyncClient, bad_val
    ):
        """API Adversarial: Non-positive or non-integer max_latency_ms values return HTTP 400."""
        payload = {
            "state": "Valid state string",
            "questions": {"triage": {"type": "noul"}},
            "max_latency_ms": bad_val,
        }
        resp = await async_client.post("/v1/auto", json=payload)
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Validation error"


# ===========================================================================
# 2. Routing with Conflicting Constraints
# ===========================================================================


class TestConflictingConstraintsRouting:
    """Verifies priority resolution when multiple conflicting routing constraints are supplied."""

    def test_unit_explicit_model_preempts_all_other_constraints(self):
        """Unit: Explicit model override takes absolute precedence over SLA and latency."""
        # Explicit heavy model overrides fast SLA and tiny latency budget
        m1, r1 = resolve_model(model="decider", sla="fast", max_latency_ms=100)
        assert m1 == "decider"
        assert "Explicit model requested: 'decider'" in r1

        # Explicit light model overrides smart SLA and huge latency budget
        m2, r2 = resolve_model(model="laya", sla="smart", max_latency_ms=10000)
        assert m2 == "laya"
        assert "Explicit model requested: 'laya'" in r2

        # Custom external model override takes precedence
        m3, r3 = resolve_model(model="custom-enterprise:70b", sla="fast", max_latency_ms=50)
        assert m3 == "custom-enterprise:70b"
        assert "Custom model requested: 'custom-enterprise:70b'" in r3

    def test_unit_auto_model_falls_through_to_latency_then_sla(self):
        """Unit: When model is 'auto' (or whitespace padded), it falls through cleanly."""
        # 'auto' with latency and sla -> latency wins
        m1, r1 = resolve_model(model="auto", sla="smart", max_latency_ms=300)
        assert m1 == "laya"
        assert "300ms <= 500ms" in r1

        # '  AUTO  ' with only sla -> sla wins
        m2, r2 = resolve_model(model="  AUTO  ", sla="pointer")
        assert m2 == "kev:4b"
        assert "Routing to 'kev:4b' via SLA profile 'pointer'" in r2

        # 'auto' with nothing -> defaults to decider
        m3, r3 = resolve_model(model="auto")
        assert m3 == "decider"
        assert "Defaulting to 'decider'" in r3

    def test_unit_latency_preempts_sla_profile(self):
        """Unit: Latency budget evaluates before SLA profile when both are present."""
        # Low latency budget overrides slow SLA
        m1, _ = resolve_model(sla="pointer", max_latency_ms=250)
        assert m1 == "laya"

        # High latency budget overrides fast SLA
        m2, _ = resolve_model(sla="fast", max_latency_ms=3000)
        assert m2 == "decider"

    def test_unit_case_insensitive_and_whitespace_handling(self):
        """Unit: Router strips whitespace and case-folds both model and SLA arguments."""
        m1, _ = resolve_model(model="  LAYA  ")
        assert m1 == "laya"

        m2, _ = resolve_model(sla="  FAST  ")
        assert m2 == "laya"

        m3, _ = resolve_model(sla="PoInTeR")
        assert m3 == "kev:4b"

    @pytest.mark.asyncio
    async def test_api_conflicting_explicit_model_beats_sla_and_latency(
        self, async_client: httpx.AsyncClient, mock_ollaya_inference
    ):
        """API: When client passes model='laya' with sla='smart' and max_latency_ms=5000,
        laya is selected."""
        payload = {
            "state": "Incident report",
            "questions": {"cat": {"type": "noul"}},
            "model": "laya",
            "sla": "smart",
            "max_latency_ms": 5000,
        }
        resp = await async_client.post("/v1/auto", json=payload)
        assert resp.status_code == 200
        assert resp.headers.get("X-Selected-Model") == "laya"
        assert "Explicit model requested: 'laya'" in resp.json()["routing"]["reason"]

    @pytest.mark.asyncio
    async def test_api_conflicting_auto_model_yields_to_latency_budget(
        self, async_client: httpx.AsyncClient, mock_ollaya_inference
    ):
        """API: When model='auto' is passed alongside max_latency_ms=200, router selects 'laya'."""
        payload = {
            "state": "Incident report",
            "questions": {"cat": {"type": "noul"}},
            "model": "auto",
            "sla": "pointer",
            "max_latency_ms": 200,
        }
        resp = await async_client.post("/v1/auto", json=payload)
        assert resp.status_code == 200
        assert resp.headers.get("X-Selected-Model") == "laya"

    @pytest.mark.asyncio
    async def test_api_conflicting_latency_overrides_sla_without_model(
        self, async_client: httpx.AsyncClient, mock_ollaya_inference
    ):
        """API: With no model override, max_latency_ms=400 overrides sla='pointer'."""
        payload = {
            "state": "Incident report",
            "questions": {"cat": {"type": "noul"}},
            "sla": "pointer",
            "max_latency_ms": 400,
        }
        resp = await async_client.post("/v1/auto", json=payload)
        assert resp.status_code == 200
        assert resp.headers.get("X-Selected-Model") == "laya"

    @pytest.mark.asyncio
    async def test_api_sla_case_normalization(
        self, async_client: httpx.AsyncClient, mock_ollaya_inference
    ):
        """API: Upper-cased SLA profiles (e.g. 'FAST', 'POINTER') are normalized and accepted."""
        for sla_val, expected_model in [
            ("FAST", "laya"),
            ("POINTER", "kev:4b"),
            ("SMART", "decider"),
        ]:
            payload = {
                "state": "Case normalization test",
                "questions": {"cat": {"type": "noul"}},
                "sla": sla_val,
            }
            resp = await async_client.post("/v1/auto", json=payload)
            assert resp.status_code == 200, f"Failed for SLA {sla_val}: {resp.text}"
            assert resp.headers.get("X-Selected-Model") == expected_model

    @pytest.mark.asyncio
    async def test_api_unrecognized_sla_returns_400(self, async_client: httpx.AsyncClient):
        """API Adversarial: Invalid SLA string returns HTTP 400 Bad Request."""
        payload = {
            "state": "Test state",
            "questions": {"cat": {"type": "noul"}},
            "sla": "turbo_quantum_speed",
        }
        resp = await async_client.post("/v1/auto", json=payload)
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Validation error"

    @pytest.mark.asyncio
    async def test_api_blank_model_override_returns_400(self, async_client: httpx.AsyncClient):
        """API Adversarial: Blank or whitespace-only model override returns HTTP 400."""
        payload = {
            "state": "Test state",
            "questions": {"cat": {"type": "noul"}},
            "model": "    ",
        }
        resp = await async_client.post("/v1/auto", json=payload)
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Validation error"


# ===========================================================================
# 3. Capabilities Catalog: 0 Models, All Corrupted Models, Mixed Formats
# ===========================================================================


class TestCapabilitiesCatalogAdversarial:
    """Verifies capabilities extraction resilience against empty, corrupt, or mixed catalogs."""

    @pytest.mark.asyncio
    async def test_unit_capabilities_with_zero_models_falls_back_to_static(
        self, respx_mock: respx.MockRouter
    ):
        """Unit: When upstream /api/tags returns 0 models {'models': []}, static catalog is used."""
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(status_code=200, json={"models": []})
        result = await get_live_models_capabilities(OLLAYA_URL)
        assert len(result) >= len(MODEL_CAPABILITIES)
        assert "decider" in result
        assert "laya" in result
        assert result["decider"]["installed"] is True

    @pytest.mark.asyncio
    async def test_unit_capabilities_with_all_corrupted_models_falls_back_to_static(
        self, respx_mock: respx.MockRouter
    ):
        """Unit Adversarial: When all models in /api/tags are malformed, returns static fallback."""
        corrupted_models = [
            None,
            12345,
            "not_a_dictionary",
            True,
            [],
            {},
            {"name": ""},
            {"name": None},
            {"name": 999},
            {"size": 1024000},  # missing name
            {"name": "bad_details_str", "details": "not_a_dict"},
            {"name": "bad_details_int", "details": 42},
        ]
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": corrupted_models}
        )
        result = await get_live_models_capabilities(OLLAYA_URL)
        assert len(result) >= len(MODEL_CAPABILITIES)
        assert "decider" in result
        assert "laya" in result

    @pytest.mark.asyncio
    async def test_unit_capabilities_whitespace_only_name_adversarial_gap(
        self, respx_mock: respx.MockRouter
    ):
        """Unit Adversarial: Upstream model with whitespace-only name ('   ') must be skipped
        as corrupt instead of polluting catalog with an un-routable model ID."""
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": [{"name": "   ", "size": 1024000}]}
        )
        result = await get_live_models_capabilities(OLLAYA_URL)
        assert "   " not in result, "Whitespace model name '   ' was not filtered out as corrupt!"
        assert len(result) >= len(MODEL_CAPABILITIES)

    @pytest.mark.asyncio
    async def test_unit_capabilities_mixed_formats_and_partial_corruptions(
        self, respx_mock: respx.MockRouter
    ):
        """Unit: Mixed models with valid, corrupt, and non-standard formats are parsed cleanly."""
        catalog = {
            "models": [
                # Corrupt item
                {"name": "corrupt_item", "details": "bad"},
                # Valid custom gguf model
                {
                    "name": "custom_gguf:v1",
                    "size": 2147483648,
                    "modified_at": "2026-09-20T10:00:00Z",
                    "details": {"format": "gguf", "parameter_size": "7B"},
                },
                # Valid item without details
                {
                    "name": "bare_model:latest",
                    "size": 1048576,
                    "modified_at": "2026-09-20T10:00:00Z",
                },
                # Valid item with null size
                {
                    "name": "null_size_model:latest",
                    "size": None,
                    "details": {"format": "onnx", "parameter_size": "1B"},
                },
                # Valid item with negative size
                {
                    "name": "negative_size:latest",
                    "size": -1024,
                    "details": {"format": "safetensors", "parameter_size": "500M"},
                },
                # Multiple colons in name
                {
                    "name": "registry.corp.io:8080/ns/model:v2.0",
                    "size": 524288000,
                    "details": {"format": "onnx", "parameter_size": "3B"},
                },
            ]
        }
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(status_code=200, json=catalog)
        result = await get_live_models_capabilities(OLLAYA_URL)

        # custom_gguf parsed correctly
        assert "custom_gguf" in result
        assert result["custom_gguf"]["format"] == "gguf"
        assert result["custom_gguf"]["parameter_size"] == "7B"

        # bare_model defaulted safely
        assert "bare_model" in result
        assert result["bare_model"]["format"] is None
        assert result["bare_model"]["parameter_size"] == "unknown"

        # null size handled without TypeError
        assert "null_size_model" in result
        assert result["null_size_model"]["vram_footprint_mb"] == 0

        # multiple colons handled cleanly
        assert "registry.corp.io" in result
        assert "registry.corp.io:8080/ns/model:v2.0" in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_payload",
        [
            ["item1", "item2"],  # list root
            "raw text instead of json",  # string root
            12345,  # int root
            None,  # null root
        ],
    )
    async def test_unit_capabilities_non_dict_response_root_falls_back(
        self, respx_mock: respx.MockRouter, bad_payload
    ):
        """Unit Adversarial: Non-dict upstream response root falls back safely to static catalog."""
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(status_code=200, json=bad_payload)
        result = await get_live_models_capabilities(OLLAYA_URL)
        assert "decider" in result
        assert "laya" in result

    @pytest.mark.asyncio
    async def test_api_capabilities_endpoint_zero_models_resilience(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """API: GET /v1/capabilities returns 200 with default manifest
        when upstream has 0 models."""
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(status_code=200, json={"models": []})
        resp = await async_client.get("/v1/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert "service" in data
        assert "decider" in data["models"]
        assert "laya" in data["models"]

    @pytest.mark.asyncio
    async def test_api_models_endpoint_all_corrupt_models_resilience(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """API: GET /v1/models returns 200 with fallback list when models are corrupt."""
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": [None, {}, {"name": ""}]}
        )
        resp = await async_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "decider" in model_ids
        assert "laya" in model_ids

    @pytest.mark.asyncio
    async def test_api_models_endpoint_whitespace_only_name_adversarial_gap(
        self, async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
    ):
        """API Adversarial: GET /v1/models should not advertise whitespace model IDs
        when upstream returns a model item with name='   '."""
        respx_mock.get(f"{OLLAYA_URL}/api/tags").respond(
            status_code=200, json={"models": [{"name": "   ", "size": 1024000}]}
        )
        resp = await async_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        model_ids = [m["id"] for m in data["data"]]
        assert "   " not in model_ids, "GET /v1/models advertised invalid whitespace model ID!"
        assert "decider" in model_ids, "Fallback to default models failed on whitespace model!"


# ===========================================================================
# 4. Complex Question Schemas: Boundaries, Max Choices, Unicode, Special Chars
# ===========================================================================


class TestComplexQuestionSchemas:
    """Verifies Pydantic bounds and character robustness across Choice, Score, and Noul schemas."""

    def test_unit_choice_minimum_2_choices_boundary(self):
        """Unit: ChoiceQuestion with exactly 2 criteria choices is valid."""
        q = ChoiceQuestion(
            type="choice", criteria={"opt_a": "First option", "opt_b": "Second option"}
        )
        assert len(q.criteria) == 2

    def test_unit_choice_maximum_255_choices_boundary(self):
        """Unit: ChoiceQuestion with exactly 255 criteria choices is valid."""
        large_criteria = {f"choice_{i}": f"Description for option {i}" for i in range(255)}
        q = ChoiceQuestion(type="choice", criteria=large_criteria)
        assert len(q.criteria) == 255

    def test_unit_choice_1_choice_rejected(self):
        """Unit: ChoiceQuestion with fewer than 2 choices (1 choice) raises ValidationError."""
        with pytest.raises(pydantic.ValidationError):
            ChoiceQuestion(type="choice", criteria={"only_one": "Single option"})

    def test_unit_choice_256_choices_rejected(self):
        """Unit: ChoiceQuestion with > 255 choices (256 choices) raises ValidationError."""
        too_many = {f"choice_{i}": f"Description {i}" for i in range(256)}
        with pytest.raises(pydantic.ValidationError):
            ChoiceQuestion(type="choice", criteria=too_many)

    @pytest.mark.parametrize("empty_key", ["", "   ", "\t", "\n"])
    def test_unit_choice_empty_or_whitespace_label_rejected(self, empty_key):
        """Unit: ChoiceQuestion with empty or whitespace-only label key raises ValueError."""
        with pytest.raises(pydantic.ValidationError, match="Criteria label keys cannot be empty"):
            ChoiceQuestion(
                type="choice", criteria={empty_key: "Description", "valid": "Valid desc"}
            )

    @pytest.mark.parametrize("empty_desc", ["", "   ", "\t", "\n"])
    def test_unit_choice_empty_or_whitespace_description_rejected(self, empty_desc):
        """Unit: ChoiceQuestion with empty or whitespace-only description raises ValueError."""
        with pytest.raises(pydantic.ValidationError, match="cannot be empty or whitespace"):
            ChoiceQuestion(type="choice", criteria={"key_a": empty_desc, "key_b": "Valid desc"})

    def test_unit_choice_special_characters_and_emojis(self):
        """Unit: ChoiceQuestion accepts special symbols, code syntax, SQL strings, and emojis."""
        special_criteria = {
            "🚨 emergency_outage": "Critical service interruption",
            "✓ normal_operation": "Healthy operational metrics",
            "<script>tag</script>": "HTML tags in label",
            "'; DROP TABLE users;--": "SQL injection characters in label",
            "key/with:colons.and[brackets]": "Structured identifier key",
        }
        q = ChoiceQuestion(type="choice", criteria=special_criteria)
        assert len(q.criteria) == 5

    def test_unit_choice_unicode_normalization_and_alphabets(self):
        """Unit: ChoiceQuestion supports Arabic, CJK, Cyrillic, and accented Latin characters."""
        multilingual_criteria = {
            "日本語_緊急": "Japanese critical issue",
            "العربية_طوارئ": "Arabic emergency issue",
            "Русский_сбой": "Russian outage description",
            "café_au_lait": "Accented Latin criteria",
        }
        q = ChoiceQuestion(type="choice", criteria=multilingual_criteria)
        assert len(q.criteria) == 4

    def test_unit_score_minimum_2_anchors_boundary(self):
        """Unit: ScoreQuestion with exactly 2 rating anchors is valid."""
        q = ScoreQuestion(type="score", criteria=["1 - low", "5 - high"])
        assert len(q.criteria) == 2

    def test_unit_score_maximum_10_anchors_boundary(self):
        """Unit: ScoreQuestion with exactly 10 rating anchors is valid."""
        anchors = [f"Tier {i}" for i in range(1, 11)]
        q = ScoreQuestion(type="score", criteria=anchors)
        assert len(q.criteria) == 10

    def test_unit_score_1_anchor_rejected(self):
        """Unit: ScoreQuestion with fewer than 2 anchors raises ValidationError."""
        with pytest.raises(pydantic.ValidationError):
            ScoreQuestion(type="score", criteria=["Single anchor"])

    def test_unit_score_11_anchors_rejected(self):
        """Unit: ScoreQuestion with > 10 anchors (11 anchors) raises ValidationError."""
        anchors = [f"Tier {i}" for i in range(1, 12)]
        with pytest.raises(pydantic.ValidationError):
            ScoreQuestion(type="score", criteria=anchors)

    @pytest.mark.parametrize("empty_anchor", ["", "   ", "\t", "\n"])
    def test_unit_score_empty_or_whitespace_anchor_rejected(self, empty_anchor):
        """Unit: ScoreQuestion with empty or whitespace-only anchor raises ValueError."""
        with pytest.raises(pydantic.ValidationError, match="cannot be empty or whitespace"):
            ScoreQuestion(type="score", criteria=["1 - valid", empty_anchor])

    def test_unit_score_special_characters_and_emojis(self):
        """Unit: ScoreQuestion accepts emoji ratings and unicode symbols."""
        anchors = ["⭐ Poor quality", "⭐⭐⭐⭐⭐ Outstanding quality"]
        q = ScoreQuestion(type="score", criteria=anchors)
        assert len(q.criteria) == 2

    def test_unit_noul_criteria_variants(self):
        """Unit: NoulQuestion supports None, empty dict, and populated criteria."""
        q1 = NoulQuestion(type="noul")
        assert q1.criteria is None

        q2 = NoulQuestion(type="noul", criteria={})
        assert q2.criteria == {}

        q3 = NoulQuestion(type="noul", criteria={"yes": "Approve action", "no": "Reject action"})
        assert q3.criteria is not None
        assert len(q3.criteria) == 2

    @pytest.mark.parametrize("bad_criteria", [{"": "blank key"}, {"yes": "   "}])
    def test_unit_noul_invalid_criteria_rejected(self, bad_criteria):
        """Unit: NoulQuestion rejects blank keys or blank descriptions in criteria."""
        with pytest.raises(pydantic.ValidationError):
            NoulQuestion(type="noul", criteria=bad_criteria)

    @pytest.mark.asyncio
    async def test_api_choice_max_choices_255_evaluated(
        self, async_client: httpx.AsyncClient, mock_ollaya_inference
    ):
        """API: Choice question with exactly 255 choices evaluates with HTTP 200."""
        payload = {
            "state": "Taxonomy classification across 255 legal categories",
            "questions": {
                "category": {
                    "type": "choice",
                    "criteria": {f"cat_{i}": f"Legal category description {i}" for i in range(255)},
                }
            },
        }
        resp = await async_client.post("/v1/auto", json=payload)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        assert "category" in resp.json()["answers"]

    @pytest.mark.asyncio
    async def test_api_choice_exceeds_max_256_returns_400(self, async_client: httpx.AsyncClient):
        """API Adversarial: Choice question with 256 choices returns HTTP 400 Bad Request."""
        payload = {
            "state": "Taxonomy classification exceeding 255 categories",
            "questions": {
                "category": {
                    "type": "choice",
                    "criteria": {f"cat_{i}": f"Legal category description {i}" for i in range(256)},
                }
            },
        }
        resp = await async_client.post("/v1/auto", json=payload)
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Validation error"

    @pytest.mark.asyncio
    async def test_api_score_max_anchors_10_evaluated(
        self, async_client: httpx.AsyncClient, mock_ollaya_inference
    ):
        """API: Score question with exactly 10 rating anchors evaluates with HTTP 200."""
        payload = {
            "state": "Incident severity assessment along 10 tiers",
            "questions": {
                "severity": {
                    "type": "score",
                    "criteria": [f"Level {i} definition" for i in range(1, 11)],
                }
            },
        }
        resp = await async_client.post("/v1/auto", json=payload)
        assert resp.status_code == 200
        assert "severity" in resp.json()["answers"]

    @pytest.mark.asyncio
    async def test_api_score_exceeds_max_11_returns_400(self, async_client: httpx.AsyncClient):
        """API Adversarial: Score question with 11 rating anchors returns HTTP 400."""
        payload = {
            "state": "Incident severity assessment with 11 tiers",
            "questions": {
                "severity": {
                    "type": "score",
                    "criteria": [f"Level {i} definition" for i in range(1, 12)],
                }
            },
        }
        resp = await async_client.post("/v1/auto", json=payload)
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Validation error"


# ===========================================================================
# 5. Malformed Decision Requests with Edge-Case Question Keys & Bounds
# ===========================================================================


class TestMalformedDecisionRequestsAndEdgeCaseKeys:
    """Verifies robustness against boundary counts, special question IDs, and malformed keys."""

    def test_unit_decision_request_question_count_boundary_256(self):
        """Unit: DecisionRequest accepts up to 256 distinct questions."""
        questions_dict = {f"q_{i}": {"type": "noul"} for i in range(256)}
        req = DecisionRequest(state="Evaluation state", questions=questions_dict)
        assert len(req.questions) == 256

    def test_unit_decision_request_question_count_exceeds_257_rejected(self):
        """Unit: DecisionRequest with 257 questions raises ValidationError (max_length=256)."""
        questions_dict = {f"q_{i}": {"type": "noul"} for i in range(257)}
        with pytest.raises(pydantic.ValidationError):
            DecisionRequest(state="Evaluation state", questions=questions_dict)

    @pytest.mark.parametrize("empty_qid", ["", "   ", "\t", "\n"])
    def test_unit_decision_request_empty_or_whitespace_question_id_rejected(self, empty_qid):
        """Unit: Empty or whitespace-only question ID raises ValidationError."""
        with pytest.raises(
            pydantic.ValidationError, match="cannot be empty or contain only whitespace"
        ):
            DecisionRequest(state="Evaluation state", questions={empty_qid: {"type": "noul"}})

    def test_unit_decision_request_special_and_unicode_question_ids(self):
        """Unit: DecisionRequest allows structured identifiers, dots, hyphens, and unicode IDs."""
        questions_dict = {
            "namespace:metric.status[0]": {"type": "noul"},
            "질문_한국어_식별자": {"type": "noul"},
            "question_🔥_severity": {"type": "noul"},
            "q-with_hyphen.and_underscore": {"type": "noul"},
        }
        req = DecisionRequest(state="Evaluation state", questions=questions_dict)
        assert len(req.questions) == 4

    def test_unit_decision_request_state_boundary_1mb(self):
        """Unit: DecisionRequest state exactly 1MB (1,048,576 chars) is accepted."""
        state_1mb = "a" * 1048576
        req = DecisionRequest(state=state_1mb, questions={"q": {"type": "noul"}})
        assert len(req.state) == 1048576

    def test_unit_decision_request_state_exceeds_1mb_rejected(self):
        """Unit: DecisionRequest state > 1MB (1,048,577 chars) raises ValidationError."""
        state_oversize = "a" * 1048577
        with pytest.raises(pydantic.ValidationError):
            DecisionRequest(state=state_oversize, questions={"q": {"type": "noul"}})

    def test_unit_decision_request_model_override_boundary_128(self):
        """Unit: DecisionRequest model override up to 128 characters is accepted."""
        req = DecisionRequest(state="state", questions={"q": {"type": "noul"}}, model="m" * 128)
        assert req.model == "m" * 128

    def test_unit_decision_request_model_override_exceeds_128_rejected(self):
        """Unit: DecisionRequest model override with 129 characters raises ValidationError."""
        with pytest.raises(pydantic.ValidationError):
            DecisionRequest(state="state", questions={"q": {"type": "noul"}}, model="m" * 129)

    def test_unit_decision_request_extra_fields_ignored(self):
        """Unit: Unknown extra fields on DecisionRequest payload are safely ignored."""
        req = DecisionRequest(
            state="state",
            questions={"q": {"type": "noul"}},
            arbitrary_extra_field="unexpected_value",
            meta_timestamp=1234567890,
        )
        assert not hasattr(req, "arbitrary_extra_field")

    @pytest.mark.asyncio
    async def test_api_256_questions_evaluated(
        self, async_client: httpx.AsyncClient, mock_ollaya_inference
    ):
        """API: Batch decision request with exactly 256 questions evaluates with HTTP 200."""
        payload = {
            "state": "Multi-question enterprise compliance verification",
            "questions": {f"compliance_rule_{i}": {"type": "noul"} for i in range(256)},
        }
        resp = await async_client.post("/v1/auto", json=payload)
        assert resp.status_code == 200
        assert len(resp.json()["answers"]) == 256

    @pytest.mark.asyncio
    async def test_api_257_questions_returns_400(self, async_client: httpx.AsyncClient):
        """API Adversarial: Batch decision request with 257 questions returns HTTP 400."""
        payload = {
            "state": "Multi-question enterprise compliance verification",
            "questions": {f"compliance_rule_{i}": {"type": "noul"} for i in range(257)},
        }
        resp = await async_client.post("/v1/auto", json=payload)
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Validation error"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("empty_key", ["", "   ", "\t"])
    async def test_api_empty_or_whitespace_question_id_returns_400(
        self, async_client: httpx.AsyncClient, empty_key
    ):
        """API Adversarial: Empty or whitespace question IDs return HTTP 400."""
        payload = {
            "state": "Valid evaluation state",
            "questions": {empty_key: {"type": "noul"}},
        }
        resp = await async_client.post("/v1/auto", json=payload)
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Validation error"

    @pytest.mark.asyncio
    async def test_api_special_and_unicode_question_ids_evaluated(
        self, async_client: httpx.AsyncClient, mock_ollaya_inference
    ):
        """API: Complex question IDs with unicode, colons, brackets, and emojis evaluate cleanly."""
        payload = {
            "state": "Valid evaluation state",
            "questions": {
                "ns:auth.session[0]": {"type": "noul"},
                "질문_한국어": {"type": "noul"},
                "q_🔥_priority": {"type": "noul"},
            },
        }
        resp = await async_client.post("/v1/auto", json=payload)
        assert resp.status_code == 200
        answers = resp.json()["answers"]
        assert "ns:auth.session[0]" in answers
        assert "질문_한국어" in answers
        assert "q_🔥_priority" in answers

    @pytest.mark.asyncio
    async def test_api_unknown_question_type_returns_400(self, async_client: httpx.AsyncClient):
        """API Adversarial: Unrecognized question type returns HTTP 400 Bad Request."""
        payload = {
            "state": "Valid evaluation state",
            "questions": {"q1": {"type": "unsupported_slider_type"}},
        }
        resp = await async_client.post("/v1/auto", json=payload)
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Validation error"

    @pytest.mark.asyncio
    async def test_api_non_dict_question_spec_returns_400(self, async_client: httpx.AsyncClient):
        """API Adversarial: Primitive values as question specification return HTTP 400."""
        payload = {
            "state": "Valid evaluation state",
            "questions": {"q1": "just a string instead of dict"},
        }
        resp = await async_client.post("/v1/auto", json=payload)
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Validation error"

    @pytest.mark.asyncio
    async def test_api_extra_payload_fields_silently_ignored(
        self, async_client: httpx.AsyncClient, mock_ollaya_inference
    ):
        """API: Extraneous top-level fields are safely ignored without raising validation error."""
        payload = {
            "state": "Valid evaluation state",
            "questions": {"q1": {"type": "noul"}},
            "client_timestamp": 1727389200,
            "session_token": "random_token_12345",
            "debug_trace": True,
        }
        resp = await async_client.post("/v1/auto", json=payload)
        assert resp.status_code == 200
        assert "q1" in resp.json()["answers"]
