"""
SLA & Intent-Based Router
Resolves the best model based on caller constraints (SLA, latency budget, or explicit model).
"""

from typing import Tuple, Optional
from app.capabilities import MODEL_CAPABILITIES, SLA_ROUTING_PROFILES

def resolve_model(
    model: Optional[str] = None,
    sla: Optional[str] = None,
    max_latency_ms: Optional[int] = None
) -> Tuple[str, str]:
    """
    Returns (selected_model_name, routing_reason)
    """
    # 1. Explicit model request (non-auto)
    if model and model != "auto":
        clean_model = model.strip().lower()
        if clean_model in MODEL_CAPABILITIES:
            return clean_model, f"Explicit model requested: '{clean_model}'"
        # If it's a known Ollaya model format
        return clean_model, f"Custom model requested: '{clean_model}'"

    # 2. Latency budget constraint
    if max_latency_ms is not None:
        if max_latency_ms <= 500:
            return "laya", f"Routing to 'laya' to meet max_latency_ms budget ({max_latency_ms}ms <= 500ms, ~3.3 FPS)"
        elif max_latency_ms <= 2500:
            return "decider", f"Routing to 'decider' within max_latency_ms budget ({max_latency_ms}ms, ~1.6s latency)"
        else:
            return "decider", f"Routing to default high-accuracy model 'decider' (within {max_latency_ms}ms budget)"

    # 3. SLA profile request
    if sla:
        clean_sla = sla.strip().lower()
        if clean_sla in SLA_ROUTING_PROFILES:
            profile = SLA_ROUTING_PROFILES[clean_sla]
            target = profile["target_model"]
            return target, f"Routing to '{target}' via SLA profile '{clean_sla}': {profile['description']}"

    # 4. Default: Decider (highest intelligence and reliability)
    return "decider", "Defaulting to 'decider' for high accuracy and causal context reasoning"
