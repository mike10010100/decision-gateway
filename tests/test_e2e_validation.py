"""
E2E Validation Test Suite (Feature F1, Tiers 1-2).
Tests strict schema validation on /v1/auto, /v1/systemone, and /v1/models/pull.
Verifies that all malformed, partial, empty, or type-mismatched payloads
consistently return HTTP 400 Bad Request with descriptive error details.
"""

import sys
import os
import pytest
import httpx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ===========================================================================
# Tier 1: Core Feature Validation Coverage (F1)
# ===========================================================================


@pytest.mark.asyncio
async def test_auto_missing_state_returns_400(async_client: httpx.AsyncClient):
    """F1: Missing 'state' on /v1/auto must return 400 Bad Request."""
    payload = {"questions": {"intent": {"type": "choice", "criteria": {"a": "Option A"}}}}
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "detail" in data or "errors" in data


@pytest.mark.asyncio
async def test_systemone_missing_state_returns_400(async_client: httpx.AsyncClient):
    """F1: Missing 'state' on /v1/systemone must return 400 Bad Request."""
    payload = {"questions": {"intent": {"type": "choice", "criteria": {"a": "Option A"}}}}
    resp = await async_client.post("/v1/systemone", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_missing_questions_returns_400(async_client: httpx.AsyncClient):
    """F1: Missing 'questions' on /v1/auto must return 400 Bad Request."""
    payload = {"state": "The user reported an error on checkout."}
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_systemone_missing_questions_returns_400(async_client: httpx.AsyncClient):
    """F1: Missing 'questions' on /v1/systemone must return 400 Bad Request."""
    payload = {"state": "The user reported an error on checkout."}
    resp = await async_client.post("/v1/systemone", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_empty_body_returns_400(async_client: httpx.AsyncClient):
    """F1: Completely empty JSON body on /v1/auto must return 400."""
    resp = await async_client.post("/v1/auto", json={})
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_systemone_empty_body_returns_400(async_client: httpx.AsyncClient):
    """F1: Completely empty JSON body on /v1/systemone must return 400."""
    resp = await async_client.post("/v1/systemone", json={})
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_pull_model_missing_model_returns_400(async_client: httpx.AsyncClient):
    """F1: /v1/models/pull with missing 'model' field must return 400 (not 422 or 500)."""
    resp = await async_client.post("/v1/models/pull", json={})
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_pull_model_empty_model_name_returns_400(async_client: httpx.AsyncClient):
    """F1: /v1/models/pull with empty model name string must return 400."""
    resp = await async_client.post("/v1/models/pull", json={"model": ""})
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_decision_happy_path_choice_accepted(
    async_client: httpx.AsyncClient, sample_payloads
):
    """F1: Valid choice decision payload returns 200 OK."""
    resp = await async_client.post("/v1/auto", json=sample_payloads["choice"])
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "routing" in data
    assert "answers" in data


@pytest.mark.asyncio
async def test_pull_model_happy_path_accepted(async_client: httpx.AsyncClient):
    """F1: Valid pull model payload returns 200 OK."""
    resp = await async_client.post("/v1/models/pull", json={"model": "decider"})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    assert resp.json()["status"] == "success"


# ===========================================================================
# Tier 2: Boundary, Adversarial & Type Error Validation Cases (F1)
# ===========================================================================


@pytest.mark.asyncio
async def test_auto_state_empty_string_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: 'state' as empty string '' must return 400."""
    payload = {"state": "", "questions": {"q1": {"type": "noul"}}}
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_state_whitespace_only_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: 'state' with whitespace-only '   ' must be rejected with 400."""
    payload = {"state": "    \t\n   ", "questions": {"q1": {"type": "noul"}}}
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_state_non_string_int_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: 'state' as an integer 12345 must return 400."""
    payload = {"state": 12345, "questions": {"q1": {"type": "noul"}}}
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_state_non_string_list_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: 'state' as a list must return 400."""
    payload = {"state": ["state line 1", "state line 2"], "questions": {"q1": {"type": "noul"}}}
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_questions_empty_dict_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: 'questions' as empty dict {} must return 400."""
    payload = {"state": "The user reported an incident.", "questions": {}}
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_questions_non_dict_list_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: 'questions' as a list instead of dict must return 400."""
    payload = {
        "state": "The user reported an incident.",
        "questions": [{"id": "q1", "type": "noul"}],
    }
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_questions_non_dict_string_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: 'questions' as a string instead of dict must return 400."""
    payload = {"state": "The user reported an incident.", "questions": "Is this urgent?"}
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_question_item_missing_type_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: Question item missing 'type' field must return 400."""
    payload = {
        "state": "The user reported an incident.",
        "questions": {"q1": {"description": "Missing type key"}},
    }
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_question_item_invalid_type_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: Question item with invalid type 'unsupported' must return 400."""
    payload = {
        "state": "The user reported an incident.",
        "questions": {"q1": {"type": "unsupported_type_name"}},
    }
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_choice_question_missing_criteria_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: 'choice' question missing required 'criteria' field must return 400."""
    payload = {"state": "The user reported an incident.", "questions": {"q1": {"type": "choice"}}}
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_choice_question_empty_criteria_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: 'choice' question with empty criteria dict {} must return 400."""
    payload = {
        "state": "The user reported an incident.",
        "questions": {"q1": {"type": "choice", "criteria": {}}},
    }
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_choice_question_criteria_not_dict_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: 'choice' question with list criteria instead of dict must return 400."""
    payload = {
        "state": "The user reported an incident.",
        "questions": {"q1": {"type": "choice", "criteria": ["option1", "option2"]}},
    }
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_sla_invalid_type_int_returns_400(async_client: httpx.AsyncClient):
    """F1 Adversarial: Non-string SLA integer 123 must return 400 without crashing."""
    payload = {
        "state": "The user reported an incident.",
        "questions": {"q1": {"type": "noul"}},
        "sla": 123,
    }
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_sla_invalid_type_bool_returns_400(async_client: httpx.AsyncClient):
    """F1 Adversarial: Non-string SLA boolean True must return 400 without crashing."""
    payload = {
        "state": "The user reported an incident.",
        "questions": {"q1": {"type": "noul"}},
        "sla": True,
    }
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_sla_invalid_value_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: Unrecognized SLA value 'ultra_mega_fast' must return 400."""
    payload = {
        "state": "The user reported an incident.",
        "questions": {"q1": {"type": "noul"}},
        "sla": "ultra_mega_fast",
    }
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_max_latency_non_int_string_returns_400(async_client: httpx.AsyncClient):
    """F1 Adversarial: 'max_latency_ms' as string '500' or 'fast' must return 400 without crash."""
    payload = {
        "state": "The user reported an incident.",
        "questions": {"q1": {"type": "noul"}},
        "max_latency_ms": "fast",
    }
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_max_latency_negative_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: Negative 'max_latency_ms' (-50) must return 400."""
    payload = {
        "state": "The user reported an incident.",
        "questions": {"q1": {"type": "noul"}},
        "max_latency_ms": -50,
    }
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_max_latency_zero_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: Zero 'max_latency_ms' (0) must return 400 (budget must be > 0)."""
    payload = {
        "state": "The user reported an incident.",
        "questions": {"q1": {"type": "noul"}},
        "max_latency_ms": 0,
    }
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_model_non_string_int_returns_400(async_client: httpx.AsyncClient):
    """F1 Adversarial: 'model' as integer 999 must return 400."""
    payload = {
        "state": "The user reported an incident.",
        "questions": {"q1": {"type": "noul"}},
        "model": 999,
    }
    resp = await async_client.post("/v1/auto", json=payload)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_json_array_root_returns_400(async_client: httpx.AsyncClient):
    """F1 Adversarial: JSON array root [1, 2, 3] must return 400 without crashing."""
    resp = await async_client.post(
        "/v1/auto", content="[1, 2, 3]", headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_json_primitive_string_returns_400(async_client: httpx.AsyncClient):
    """F1 Adversarial: JSON primitive string root must return 400 without crashing."""
    resp = await async_client.post(
        "/v1/auto", content='"just a string"', headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_auto_invalid_json_syntax_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: Corrupt JSON syntax '{state: bad}' must return 400."""
    resp = await async_client.post(
        "/v1/auto", content="{state: bad syntax,}", headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_pull_model_whitespace_only_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: 'model' with whitespace only '   ' must return 400."""
    resp = await async_client.post("/v1/models/pull", json={"model": "   "})
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_pull_model_non_string_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: 'model' as non-string (int 123) must return 400."""
    resp = await async_client.post("/v1/models/pull", json={"model": 123})
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


@pytest.mark.asyncio
async def test_pull_stream_non_boolean_returns_400(async_client: httpx.AsyncClient):
    """F1 Boundary: 'stream' field with non-boolean string 'not_a_bool' must return 400."""
    resp = await async_client.post(
        "/v1/models/pull", json={"model": "decider", "stream": "not_a_bool"}
    )
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"
