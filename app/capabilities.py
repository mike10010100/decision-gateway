"""
Capabilities Manifest & Model Profiles
Advertises performance, memory, FPS, and reasoning tiers to remote entities.
"""

from typing import Dict, Any

MODEL_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    "decider": {
        "name": "decider:latest",
        "alias": "decider",
        "description": "Decoder decision model built on Qwen 3.5 2B with direct logit readout.",
        "parameter_size": "1.9B",
        "vram_footprint_mb": 3800,
        "avg_latency_ms": 1672,
        "throughput_fps": 0.60,
        "reasoning_tier": "deep_causal",
        "decision_quality": "High. Accurately deduces implied urgency, intent, and subtle emotions.",
        "best_for": [
            "Customer support triage & incident routing",
            "Complex semantic classification",
            "High-reliability guardrails",
            "Urgency & churn detection"
        ],
        "is_default": True
    },
    "laya": {
        "name": "laya:latest",
        "alias": "laya",
        "description": "Encoder decision model based on ModernBERT-large (421M params).",
        "parameter_size": "421M",
        "vram_footprint_mb": 850,
        "avg_latency_ms": 304,
        "throughput_fps": 3.29,
        "reasoning_tier": "surface_keyword",
        "decision_quality": "Fast, but misses subtle context (requires explicit keywords for severity).",
        "best_for": [
            "High-throughput edge loops (>3 FPS)",
            "Spam and profanity filtering",
            "Dead-simple keyword routing",
            "Low-power or memory-constrained scenarios"
        ],
        "is_default": False
    },
    "kev:4b": {
        "name": "kev:4b",
        "alias": "kev:4b",
        "description": "Pointer-head decision model on Qwen 4B base with block-causal attention spans.",
        "parameter_size": "4.0B",
        "vram_footprint_mb": 9500,
        "avg_latency_ms": 4333,
        "throughput_fps": 0.23,
        "reasoning_tier": "deep_pointer_head",
        "decision_quality": "High accuracy, but 2.6x slower than Decider on local hardware.",
        "best_for": [
            "Workflows requiring Jared Palmer's Kev pointer-head span extraction",
            "Dynamic candidate span ranking"
        ],
        "is_default": False
    },
    "kev": {
        "name": "kev:latest",
        "alias": "kev",
        "description": "Pointer-head decision model on Qwen 3.5 0.8B base.",
        "parameter_size": "0.76B",
        "vram_footprint_mb": 1800,
        "avg_latency_ms": 667,
        "throughput_fps": 1.50,
        "reasoning_tier": "lightweight_pointer_head",
        "decision_quality": "Medium. Struggles with multi-hop context reasoning.",
        "best_for": [
            "Medium-speed pointer-head experiments"
        ],
        "is_default": False
    }
}

SLA_ROUTING_PROFILES = {
    "fast": {
        "target_model": "laya",
        "description": "Ultra-low latency (<500ms, ~3.3 FPS) for simple keyword filtering and high-frequency loops.",
        "max_latency_budget_ms": 500
    },
    "smart": {
        "target_model": "decider",
        "description": "High accuracy (~1.6s, ~0.6 FPS) with deep context and implied urgency comprehension. (Recommended default)",
        "max_latency_budget_ms": 2500
    },
    "accurate": {
        "target_model": "decider",
        "description": "Same as 'smart' - resolves to Decider for maximum decision fidelity.",
        "max_latency_budget_ms": 2500
    },
    "pointer": {
        "target_model": "kev:4b",
        "description": "Full 4B pointer-head evaluation for span-based decisions.",
        "max_latency_budget_ms": 6000
    }
}
