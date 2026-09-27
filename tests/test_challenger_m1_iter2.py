"""
Milestone M1 Iteration 2 Challenger Verification & Adversarial Fuzzing Suite.
Verifies the remediation of previous HTTP 422 leakage vulnerabilities and tests
deep fuzzing vectors against /v1/auto, /v1/systemone, and /v1/models/pull.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app, raise_server_exceptions=False)
ENDPOINTS = ["/v1/auto", "/v1/systemone"]


def assert_400_validation_error(resp, expected_error_type=None, expected_loc_field=None):
    """Assert response is HTTP 400 Bad Request with structured validation error body."""
    assert resp.status_code == 400, f"Expected HTTP 400, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert (
        body.get("detail") == "Validation error"
    ), f"Expected detail 'Validation error', got: {body}"
    assert "errors" in body, f"Missing 'errors' in body: {body}"
    assert isinstance(body["errors"], list)
    assert len(body["errors"]) > 0

    if expected_error_type:
        types = [err.get("type", "") for err in body["errors"]]
        assert any(
            expected_error_type in t for t in types
        ), f"Expected error type containing '{expected_error_type}', got {types}"

    if expected_loc_field:
        all_locs = [str(part) for err in body["errors"] for part in err.get("loc", [])]
        assert any(
            expected_loc_field in loc for loc in all_locs
        ), f"Expected loc containing '{expected_loc_field}', got {all_locs}"


class TestRechallengeIteration1Bugs:
    """
    Re-challenges the 4 previously failing 422 vulnerabilities from Iteration 1.
    All must now return HTTP 400 Bad Request with detail 'Validation error' (zero 422s).
    """

    @pytest.mark.parametrize("ep", ENDPOINTS)
    def test_bug1_257_questions_rejected(self, ep):
        payload = {
            "state": "Normal valid state text",
            "questions": {f"q_{i}": {"type": "noul"} for i in range(257)},
        }
        resp = client.post(ep, json=payload)
        assert resp.status_code != 422, "Bug 1 regression: returned 422!"
        assert_400_validation_error(
            resp, expected_error_type="too_long", expected_loc_field="questions"
        )

    @pytest.mark.parametrize("ep", ENDPOINTS)
    def test_bug2_empty_string_question_id_rejected(self, ep):
        payload = {"state": "Normal valid state text", "questions": {"": {"type": "noul"}}}
        resp = client.post(ep, json=payload)
        assert resp.status_code != 422, "Bug 2 regression: returned 422!"
        assert_400_validation_error(
            resp, expected_error_type="value_error", expected_loc_field="questions"
        )

    @pytest.mark.parametrize("ep", ENDPOINTS)
    def test_bug3_256_choice_criteria_rejected(self, ep):
        crit = {f"opt_{i}": f"Desc {i}" for i in range(256)}
        payload = {
            "state": "Normal valid state text",
            "questions": {"q1": {"type": "choice", "criteria": crit}},
        }
        resp = client.post(ep, json=payload)
        assert resp.status_code != 422, "Bug 3 regression: returned 422!"
        assert_400_validation_error(
            resp, expected_error_type="too_long", expected_loc_field="criteria"
        )

    @pytest.mark.parametrize("ep", ENDPOINTS)
    def test_bug4_11_score_anchors_rejected(self, ep):
        anchors = [f"anchor_{i}" for i in range(11)]
        payload = {
            "state": "Normal valid state text",
            "questions": {"q1": {"type": "score", "criteria": anchors}},
        }
        resp = client.post(ep, json=payload)
        assert resp.status_code != 422, "Bug 4 regression: returned 422!"
        assert_400_validation_error(
            resp, expected_error_type="too_long", expected_loc_field="criteria"
        )


class TestAdversarialCollectionBoundsAndOffByOne:
    """Stress tests exact boundary limits and off-by-one/two violations."""

    @pytest.mark.parametrize("ep", ENDPOINTS)
    def test_questions_256_exact_bound_accepted(self, ep):
        payload = {
            "state": "Normal valid state text",
            "questions": {f"q_{i}": {"type": "noul"} for i in range(256)},
        }
        resp = client.post(ep, json=payload)
        assert resp.status_code in [200, 502], f"Expected valid acceptance, got {resp.status_code}"
        assert resp.status_code != 422

    @pytest.mark.parametrize("ep", ENDPOINTS)
    def test_questions_258_rejected(self, ep):
        payload = {
            "state": "Normal valid state text",
            "questions": {f"q_{i}": {"type": "noul"} for i in range(258)},
        }
        resp = client.post(ep, json=payload)
        assert_400_validation_error(resp, expected_error_type="too_long")

    @pytest.mark.parametrize("ep", ENDPOINTS)
    def test_choice_criteria_255_exact_bound_accepted(self, ep):
        crit = {f"opt_{i}": f"Desc {i}" for i in range(255)}
        payload = {
            "state": "Normal valid state text",
            "questions": {"q1": {"type": "choice", "criteria": crit}},
        }
        resp = client.post(ep, json=payload)
        assert resp.status_code in [200, 502], f"Expected valid acceptance, got {resp.status_code}"
        assert resp.status_code != 422

    @pytest.mark.parametrize("ep", ENDPOINTS)
    def test_choice_criteria_257_rejected(self, ep):
        crit = {f"opt_{i}": f"Desc {i}" for i in range(257)}
        payload = {
            "state": "Normal valid state text",
            "questions": {"q1": {"type": "choice", "criteria": crit}},
        }
        resp = client.post(ep, json=payload)
        assert_400_validation_error(resp, expected_error_type="too_long")

    @pytest.mark.parametrize("ep", ENDPOINTS)
    def test_score_criteria_10_exact_bound_accepted(self, ep):
        anchors = [f"anchor_{i}" for i in range(10)]
        payload = {
            "state": "Normal valid state text",
            "questions": {"q1": {"type": "score", "criteria": anchors}},
        }
        resp = client.post(ep, json=payload)
        assert resp.status_code in [200, 502], f"Expected valid acceptance, got {resp.status_code}"
        assert resp.status_code != 422

    @pytest.mark.parametrize("ep", ENDPOINTS)
    def test_score_criteria_12_rejected(self, ep):
        anchors = [f"anchor_{i}" for i in range(12)]
        payload = {
            "state": "Normal valid state text",
            "questions": {"q1": {"type": "score", "criteria": anchors}},
        }
        resp = client.post(ep, json=payload)
        assert_400_validation_error(resp, expected_error_type="too_long")


class TestAdversarialQuestionKeyFuzzing:
    """Stress tests on question dictionary keys: whitespace variants and unicode."""

    @pytest.mark.parametrize("ep", ENDPOINTS)
    @pytest.mark.parametrize(
        "ws_key",
        [
            " ",
            "   ",
            "\t",
            "\n",
            "\r\n",
            "  \t\n  ",
            "\x0b",
            "\x0c",
        ],
    )
    def test_whitespace_question_ids_rejected(self, ep, ws_key):
        payload = {"state": "Normal valid state text", "questions": {ws_key: {"type": "noul"}}}
        resp = client.post(ep, json=payload)
        assert resp.status_code != 422
        assert_400_validation_error(
            resp, expected_error_type="value_error", expected_loc_field="questions"
        )

    @pytest.mark.parametrize("ep", ENDPOINTS)
    @pytest.mark.parametrize(
        "u_key",
        [
            "问题_分类",
            "سؤال_توجيه",
            "question_❓_alert",
            "question_déjà_vu",
            "🚀_launcher",
            "q-1.2.3_alpha",
        ],
    )
    def test_unicode_question_ids_accepted(self, ep, u_key):
        payload = {"state": "Normal valid state text", "questions": {u_key: {"type": "noul"}}}
        resp = client.post(ep, json=payload)
        assert resp.status_code in [200, 502], f"Failed with {resp.status_code}: {resp.text}"
        assert resp.status_code != 422


class TestAdversarialLatencyFuzzing:
    """Stress tests on max_latency_ms: float, string, boolean, out-of-range."""

    @pytest.mark.parametrize("ep", ENDPOINTS)
    @pytest.mark.parametrize(
        "str_latency",
        [
            "500",
            "0",
            "-10",
            "abc",
            "500.5",
            "",
            "   ",
            "None",
        ],
    )
    def test_string_latencies_rejected(self, ep, str_latency):
        payload = {
            "state": "Normal valid state text",
            "questions": {"q1": {"type": "noul"}},
            "max_latency_ms": str_latency,
        }
        resp = client.post(ep, json=payload)
        assert resp.status_code != 422
        assert_400_validation_error(resp, expected_loc_field="max_latency_ms")

    @pytest.mark.parametrize("ep", ENDPOINTS)
    @pytest.mark.parametrize(
        "float_latency",
        [
            500.5,
            0.1,
            -0.5,
            123.456,
        ],
    )
    def test_fractional_float_latencies_rejected(self, ep, float_latency):
        payload = {
            "state": "Normal valid state text",
            "questions": {"q1": {"type": "noul"}},
            "max_latency_ms": float_latency,
        }
        resp = client.post(ep, json=payload)
        assert resp.status_code != 422
        assert_400_validation_error(resp, expected_loc_field="max_latency_ms")

    @pytest.mark.parametrize("ep", ENDPOINTS)
    @pytest.mark.parametrize("bool_val", [True, False])
    def test_boolean_latencies_rejected(self, ep, bool_val):
        payload = {
            "state": "Normal valid state text",
            "questions": {"q1": {"type": "noul"}},
            "max_latency_ms": bool_val,
        }
        resp = client.post(ep, json=payload)
        assert resp.status_code != 422
        assert_400_validation_error(resp, expected_loc_field="max_latency_ms")

    @pytest.mark.parametrize("ep", ENDPOINTS)
    @pytest.mark.parametrize("bad_int", [0, -1, -500, 600001, 1000000])
    def test_out_of_bounds_integer_latencies_rejected(self, ep, bad_int):
        payload = {
            "state": "Normal valid state text",
            "questions": {"q1": {"type": "noul"}},
            "max_latency_ms": bad_int,
        }
        resp = client.post(ep, json=payload)
        assert resp.status_code != 422
        assert_400_validation_error(resp, expected_loc_field="max_latency_ms")


class TestAdversarialCriteriaKeysAndValues:
    """Stress tests on criteria dictionary keys and descriptions."""

    @pytest.mark.parametrize("ep", ENDPOINTS)
    @pytest.mark.parametrize(
        "bad_criteria",
        [
            {"": "desc", "b": "desc"},
            {"   ": "desc", "b": "desc"},
            {"a": "", "b": "desc"},
            {"a": "   ", "b": "desc"},
            {"a": "desc"},  # less than 2
        ],
    )
    def test_choice_malformed_criteria_rejected(self, ep, bad_criteria):
        payload = {
            "state": "Normal valid state text",
            "questions": {"q1": {"type": "choice", "criteria": bad_criteria}},
        }
        resp = client.post(ep, json=payload)
        assert resp.status_code != 422
        assert_400_validation_error(resp, expected_loc_field="criteria")

    @pytest.mark.parametrize("ep", ENDPOINTS)
    @pytest.mark.parametrize(
        "bad_score",
        [
            ["a"],  # less than 2
            ["a", ""],
            ["a", "   "],
            ["a", 123],
        ],
    )
    def test_score_malformed_criteria_rejected(self, ep, bad_score):
        payload = {
            "state": "Normal valid state text",
            "questions": {"q1": {"type": "score", "criteria": bad_score}},
        }
        resp = client.post(ep, json=payload)
        assert resp.status_code != 422
        assert_400_validation_error(resp, expected_loc_field="criteria")

    @pytest.mark.parametrize("ep", ENDPOINTS)
    @pytest.mark.parametrize(
        "bad_noul",
        [
            {"": "yes"},
            {"   ": "yes"},
            {"yes": ""},
            {"yes": "   "},
            "not_a_dict",
        ],
    )
    def test_noul_malformed_criteria_rejected(self, ep, bad_noul):
        payload = {
            "state": "Normal valid state text",
            "questions": {"q1": {"type": "noul", "criteria": bad_noul}},
        }
        resp = client.post(ep, json=payload)
        assert resp.status_code != 422
        assert_400_validation_error(resp, expected_loc_field="criteria")
