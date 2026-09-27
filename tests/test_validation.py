"""
Unit Tests for M1 Input Validation & 400 Bad Request Transformation
Covers /v1/auto, /v1/systemone, and /v1/models/pull across 8 validation failure modes.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

DECISION_ENDPOINTS = ["/v1/auto", "/v1/systemone"]

VALID_QUESTION = {
    "intent": {
        "type": "choice",
        "criteria": {
            "auth": "Authentication or login issue",
            "billing": "Invoice or payment issue",
        },
    }
}


def assert_validation_error(resp, expected_loc_contains=None, expected_type=None):
    assert resp.status_code == 400, f"Expected 400 Bad Request, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["detail"] == "Validation error"
    assert isinstance(data["errors"], list)
    assert len(data["errors"]) > 0
    for err in data["errors"]:
        assert "loc" in err
        assert "msg" in err
        assert "type" in err
    if expected_loc_contains:
        all_locs = [str(item) for err in data["errors"] for item in err["loc"]]
        assert any(
            expected_loc_contains in loc for loc in all_locs
        ), f"Expected loc containing '{expected_loc_contains}', got locs: {all_locs}"
    if expected_type:
        all_types = [err["type"] for err in data["errors"]]
        assert any(
            expected_type == t for t in all_types
        ), f"Expected error type '{expected_type}', got types: {all_types}"


class TestMalformedJsonSyntax:
    """Category 1: Malformed JSON syntax must return HTTP 400 Bad Request."""

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS + ["/v1/models/pull"])
    def test_unclosed_json(self, endpoint):
        resp = client.post(
            endpoint,
            content=b'{"state": "broken unclosed json',
            headers={"Content-Type": "application/json"},
        )
        assert_validation_error(resp, "body")

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS + ["/v1/models/pull"])
    def test_garbage_bytes(self, endpoint):
        resp = client.post(
            endpoint, content=b"NOT_JSON_AT_ALL", headers={"Content-Type": "application/json"}
        )
        assert_validation_error(resp)


class TestTopLevelArray:
    """Category 2: Top-level JSON array must return HTTP 400 Bad Request."""

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_array_body_decision_endpoints(self, endpoint):
        resp = client.post(endpoint, json=[{"state": "test", "questions": VALID_QUESTION}])
        assert_validation_error(resp, "body")

    def test_array_body_models_pull(self):
        resp = client.post("/v1/models/pull", json=["decider", "laya"])
        assert_validation_error(resp, "body")


class TestMissingRequiredFields:
    """Category 3: Missing required fields must return HTTP 400 Bad Request."""

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_missing_state(self, endpoint):
        resp = client.post(endpoint, json={"questions": VALID_QUESTION})
        assert_validation_error(resp, "state")

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_missing_questions(self, endpoint):
        resp = client.post(endpoint, json={"state": "Valid text state"})
        assert_validation_error(resp, "questions")

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_empty_object_decision(self, endpoint):
        resp = client.post(endpoint, json={})
        assert_validation_error(resp, "state")
        assert_validation_error(resp, "questions")

    def test_missing_model_pull(self):
        resp = client.post("/v1/models/pull", json={})
        assert_validation_error(resp, "model")

    def test_empty_body_pull(self):
        resp = client.post("/v1/models/pull")
        assert_validation_error(resp, "body")


class TestEmptyOrWhitespaceStrings:
    """Category 4: Empty or whitespace-only state/model must return HTTP 400 Bad Request."""

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    @pytest.mark.parametrize("blank_state", ["", "   ", "\t\n  \r\n"])
    def test_blank_state_rejected(self, endpoint, blank_state):
        resp = client.post(endpoint, json={"state": blank_state, "questions": VALID_QUESTION})
        assert_validation_error(resp, "state")

    @pytest.mark.parametrize("blank_model", ["", "   ", "\t\n "])
    def test_blank_model_pull_rejected(self, blank_model):
        resp = client.post("/v1/models/pull", json={"model": blank_model})
        assert_validation_error(resp, "model")

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    @pytest.mark.parametrize("blank_model", ["", "   "])
    def test_blank_model_override_rejected(self, endpoint, blank_model):
        resp = client.post(
            endpoint,
            json={"state": "Valid state", "questions": VALID_QUESTION, "model": blank_model},
        )
        assert_validation_error(resp, "model")


class TestInvalidSlaField:
    """Category 5: Invalid SLA types or enum values must return HTTP 400 Bad Request."""

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    @pytest.mark.parametrize("invalid_sla", [123, ["fast"], True, {"sla": "fast"}])
    def test_invalid_sla_types(self, endpoint, invalid_sla):
        payload = {"state": "Valid state", "questions": VALID_QUESTION, "sla": invalid_sla}
        resp = client.post(endpoint, json=payload)
        assert_validation_error(resp, "sla")

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    @pytest.mark.parametrize("invalid_enum", ["ultra_fast", "turbo", "invalid_profile", ""])
    def test_invalid_sla_enum_values(self, endpoint, invalid_enum):
        payload = {"state": "Valid state", "questions": VALID_QUESTION, "sla": invalid_enum}
        resp = client.post(endpoint, json=payload)
        assert_validation_error(resp, "sla")


class TestInvalidMaxLatencyField:
    """Category 6: Invalid max_latency_ms types or non-positive values must return
    HTTP 400 Bad Request."""

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    @pytest.mark.parametrize("invalid_type", ["fast", "500ms", "500", True, [500]])
    def test_invalid_max_latency_types(self, endpoint, invalid_type):
        payload = {
            "state": "Valid state",
            "questions": VALID_QUESTION,
            "max_latency_ms": invalid_type,
        }
        resp = client.post(endpoint, json=payload)
        assert_validation_error(resp, "max_latency_ms")

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    @pytest.mark.parametrize("string_latency", ["500", "1", "600000"])
    def test_string_max_latency_rejected(self, endpoint, string_latency):
        """String representation of integers must be rejected instead of silently coerced."""
        payload = {
            "state": "Valid state",
            "questions": VALID_QUESTION,
            "max_latency_ms": string_latency,
        }
        resp = client.post(endpoint, json=payload)
        assert_validation_error(
            resp, expected_loc_contains="max_latency_ms", expected_type="value_error"
        )

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    @pytest.mark.parametrize("non_positive", [0, -1, -500])
    def test_non_positive_max_latency(self, endpoint, non_positive):
        payload = {
            "state": "Valid state",
            "questions": VALID_QUESTION,
            "max_latency_ms": non_positive,
        }
        resp = client.post(endpoint, json=payload)
        assert_validation_error(resp, "max_latency_ms")


class TestEmptyQuestionsDict:
    """Category 7: Empty questions dictionary must return HTTP 400 Bad Request."""

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_empty_questions_dict(self, endpoint):
        payload = {"state": "Valid state", "questions": {}}
        resp = client.post(endpoint, json=payload)
        assert_validation_error(resp, "questions")


class TestInvalidQuestionFormat:
    """Category 8: Questions with invalid structure or criteria must return HTTP 400 Bad Request."""

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_questions_not_a_dict(self, endpoint):
        payload = {"state": "Valid state", "questions": ["not", "a", "dict"]}
        resp = client.post(endpoint, json=payload)
        assert_validation_error(resp, "questions")

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_question_missing_type(self, endpoint):
        payload = {"state": "Valid state", "questions": {"q1": {"criteria": {"a": "A", "b": "B"}}}}
        resp = client.post(endpoint, json=payload)
        assert_validation_error(resp, "questions")

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_question_unknown_type(self, endpoint):
        payload = {
            "state": "Valid state",
            "questions": {"q1": {"type": "unsupported_question_type"}},
        }
        resp = client.post(endpoint, json=payload)
        assert_validation_error(resp, "questions")

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_choice_insufficient_criteria(self, endpoint):
        payload = {
            "state": "Valid state",
            "questions": {"q1": {"type": "choice", "criteria": {"only_one_option": "Description"}}},
        }
        resp = client.post(endpoint, json=payload)
        assert_validation_error(resp, "criteria")

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_choice_criteria_wrong_type(self, endpoint):
        payload = {
            "state": "Valid state",
            "questions": {
                "q1": {"type": "choice", "criteria": ["option1", "option2"]}  # Should be dict
            },
        }
        resp = client.post(endpoint, json=payload)
        assert_validation_error(resp, "criteria")

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_choice_criteria_blank_label_or_desc(self, endpoint):
        payload = {
            "state": "Valid state",
            "questions": {
                "q1": {
                    "type": "choice",
                    "criteria": {"label1": "   ", "label2": "Valid description"},
                }
            },
        }
        resp = client.post(endpoint, json=payload)
        assert_validation_error(resp, "criteria")

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_score_insufficient_criteria(self, endpoint):
        payload = {
            "state": "Valid state",
            "questions": {
                "q1": {
                    "type": "score",
                    "criteria": ["single_anchor"],  # At least 2 anchors required
                }
            },
        }
        resp = client.post(endpoint, json=payload)
        assert_validation_error(resp, "criteria")

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_score_criteria_wrong_type(self, endpoint):
        payload = {
            "state": "Valid state",
            "questions": {
                "q1": {"type": "score", "criteria": {"a": "1", "b": "2"}}  # Should be list
            },
        }
        resp = client.post(endpoint, json=payload)
        assert_validation_error(resp, "criteria")

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_score_criteria_blank_anchor(self, endpoint):
        payload = {
            "state": "Valid state",
            "questions": {"q1": {"type": "score", "criteria": ["anchor1", "   "]}},
        }
        resp = client.post(endpoint, json=payload)
        assert_validation_error(resp, "criteria")

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_noul_criteria_wrong_type(self, endpoint):
        payload = {
            "state": "Valid state",
            "questions": {
                "q1": {"type": "noul", "criteria": "invalid_string"}  # Should be optional dict
            },
        }
        resp = client.post(endpoint, json=payload)
        assert_validation_error(resp, "criteria")


class TestCollectionBoundsAndQuestionKeys:
    """Category 9: Upper bounds on collections and question key validation."""

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_questions_exceeds_max_length_rejected(self, endpoint):
        """Test 257 questions payload returns HTTP 400 Bad Request with 'too_long' error."""
        payload = {
            "state": "Valid evaluation state text",
            "questions": {f"q_{i}": {"type": "noul"} for i in range(257)},
        }
        resp = client.post(endpoint, json=payload)
        assert_validation_error(resp, expected_loc_contains="questions", expected_type="too_long")

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_questions_exact_max_length_accepted(self, endpoint):
        """Test 256 questions payload passes validation without 'too_long' error."""
        payload = {
            "state": "Valid evaluation state text",
            "questions": {f"q_{i}": {"type": "noul"} for i in range(256)},
        }
        resp = client.post(endpoint, json=payload)
        assert resp.status_code != 400, f"256 questions should not fail validation: {resp.text}"

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_empty_string_question_id_rejected(self, endpoint):
        """Test empty string question ID '' is rejected with HTTP 400."""
        payload = {"state": "Valid evaluation state text", "questions": {"": {"type": "noul"}}}
        resp = client.post(endpoint, json=payload)
        assert_validation_error(
            resp, expected_loc_contains="questions", expected_type="value_error"
        )

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_whitespace_only_question_id_rejected(self, endpoint):
        """Test whitespace-only question ID '   ' is rejected with HTTP 400."""
        payload = {"state": "Valid evaluation state text", "questions": {"   ": {"type": "noul"}}}
        resp = client.post(endpoint, json=payload)
        assert_validation_error(
            resp, expected_loc_contains="questions", expected_type="value_error"
        )

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_choice_criteria_exceeds_max_length_rejected(self, endpoint):
        """Test 256 choice criteria payload returns HTTP 400 Bad Request with 'too_long' error."""
        crit = {f"opt_{i}": f"Candidate option description {i}" for i in range(256)}
        payload = {
            "state": "Valid evaluation state text",
            "questions": {"q1": {"type": "choice", "criteria": crit}},
        }
        resp = client.post(endpoint, json=payload)
        assert_validation_error(resp, expected_loc_contains="criteria", expected_type="too_long")

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_choice_criteria_exact_max_length_accepted(self, endpoint):
        """Test 255 choice criteria payload passes validation."""
        crit = {f"opt_{i}": f"Candidate option description {i}" for i in range(255)}
        payload = {
            "state": "Valid evaluation state text",
            "questions": {"q1": {"type": "choice", "criteria": crit}},
        }
        resp = client.post(endpoint, json=payload)
        assert (
            resp.status_code != 400
        ), f"255 choice criteria should not fail validation: {resp.text}"

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_score_criteria_exceeds_max_length_rejected(self, endpoint):
        """Test 11 score anchors payload returns HTTP 400 Bad Request with 'too_long' error."""
        anchors = [f"score_anchor_{i}" for i in range(11)]
        payload = {
            "state": "Valid evaluation state text",
            "questions": {"q1": {"type": "score", "criteria": anchors}},
        }
        resp = client.post(endpoint, json=payload)
        assert_validation_error(resp, expected_loc_contains="criteria", expected_type="too_long")

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_score_criteria_exact_max_length_accepted(self, endpoint):
        """Test 10 score anchors payload passes validation."""
        anchors = [f"score_anchor_{i}" for i in range(10)]
        payload = {
            "state": "Valid evaluation state text",
            "questions": {"q1": {"type": "score", "criteria": anchors}},
        }
        resp = client.post(endpoint, json=payload)
        assert resp.status_code != 400, f"10 score anchors should not fail validation: {resp.text}"

    @pytest.mark.parametrize("endpoint", DECISION_ENDPOINTS)
    def test_string_max_latency_rejected(self, endpoint):
        """Test string max_latency_ms '500' returns HTTP 400 Bad Request."""
        payload = {
            "state": "Valid state text",
            "questions": {"q1": {"type": "noul"}},
            "max_latency_ms": "500",
        }
        resp = client.post(endpoint, json=payload)
        assert_validation_error(
            resp, expected_loc_contains="max_latency_ms", expected_type="value_error"
        )
