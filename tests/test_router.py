"""
Unit Tests for app.router (SLA & Intent-Based Router)
Tests type safety, defensive validation, and SLA resolution.
"""

import pytest
from app.router import resolve_model
from app.capabilities import SLA_ROUTING_PROFILES


class TestRouterDefensiveChecks:
    """Verifies that resolve_model rejects invalid argument types and values with
    proper exceptions."""

    def test_model_non_string_type_raises_type_error(self):
        with pytest.raises(TypeError, match="model must be a string"):
            resolve_model(model=123)  # type: ignore

        with pytest.raises(TypeError, match="model must be a string"):
            resolve_model(model=["decider"])  # type: ignore

        with pytest.raises(TypeError, match="model must be a string"):
            resolve_model(model=True)  # type: ignore

    def test_max_latency_ms_non_integer_type_raises_type_error(self):
        with pytest.raises(TypeError, match="max_latency_ms must be an integer"):
            resolve_model(max_latency_ms="500")  # type: ignore

        with pytest.raises(TypeError, match="max_latency_ms must be an integer"):
            resolve_model(max_latency_ms=True)  # type: ignore[arg-type]  # (bool check)

        with pytest.raises(TypeError, match="max_latency_ms must be an integer"):
            resolve_model(max_latency_ms=[500])  # type: ignore

        with pytest.raises(TypeError, match="max_latency_ms must be an integer"):
            resolve_model(max_latency_ms=12.5)  # type: ignore

    def test_max_latency_ms_non_positive_value_raises_value_error(self):
        with pytest.raises(ValueError, match="max_latency_ms must be greater than 0"):
            resolve_model(max_latency_ms=0)

        with pytest.raises(ValueError, match="max_latency_ms must be greater than 0"):
            resolve_model(max_latency_ms=-100)

    def test_sla_non_string_type_raises_type_error(self):
        with pytest.raises(TypeError, match="sla must be a string"):
            resolve_model(sla=123)  # type: ignore

        with pytest.raises(TypeError, match="sla must be a string"):
            resolve_model(sla=["fast"])  # type: ignore

        with pytest.raises(TypeError, match="sla must be a string"):
            resolve_model(sla=True)  # type: ignore

    def test_sla_unknown_profile_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown SLA profile"):
            resolve_model(sla="ultra_super_fast")


class TestRouterResolutionPaths:
    """Verifies standard routing logic across explicit models, latency, SLA, and default."""

    def test_default_resolution(self):
        model, reason = resolve_model()
        assert model == "decider"
        assert "Defaulting to 'decider'" in reason
        assert isinstance(model, str)
        assert isinstance(reason, str)

    def test_explicit_known_model(self):
        model, reason = resolve_model(model="laya")
        assert model == "laya"
        assert "Explicit model requested: 'laya'" in reason

    def test_explicit_custom_model(self):
        model, reason = resolve_model(model="my-custom-model:latest")
        assert model == "my-custom-model:latest"
        assert "Custom model requested" in reason

    def test_explicit_auto_falls_through(self):
        model, reason = resolve_model(model="auto", sla="fast")
        assert model == "laya"

    def test_latency_budget_500ms(self):
        model, reason = resolve_model(max_latency_ms=450)
        assert model == "laya"
        assert "laya" in reason

    def test_latency_budget_2500ms(self):
        model, reason = resolve_model(max_latency_ms=2000)
        assert model == "decider"

    def test_latency_budget_large(self):
        model, reason = resolve_model(max_latency_ms=10000)
        assert model == "decider"

    @pytest.mark.parametrize(
        "sla_name,expected_model",
        [
            ("fast", "laya"),
            ("cost", "laya"),
            ("smart", "decider"),
            ("accurate", "decider"),
            ("balanced", "decider"),
            ("pointer", "kev:4b"),
        ],
    )
    def test_sla_profiles(self, sla_name, expected_model):
        assert sla_name in SLA_ROUTING_PROFILES
        model, reason = resolve_model(sla=sla_name)
        assert model == expected_model
        assert isinstance(model, str)
        assert isinstance(reason, str)
