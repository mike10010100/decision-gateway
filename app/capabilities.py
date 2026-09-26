"""
Capabilities Manifest & Model Profiles
Advertises performance, memory, FPS, and reasoning tiers to remote entities.
Supports live dynamic merging with Ollaya's local installed models.
"""

from typing import Dict, Any, List
import httpx

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
    },
    "qwen3guard": {
        "name": "qwen3guard:latest",
        "alias": "qwen3guard",
        "description": "Safety & guardrail classification model.",
        "parameter_size": "1.5B",
        "vram_footprint_mb": 1500,
        "avg_latency_ms": 950,
        "throughput_fps": 1.05,
        "reasoning_tier": "safety_guardrail",
        "decision_quality": "High precision for safety and policy violation detection.",
        "best_for": ["Safety moderation", "Content policy guardrails"],
        "is_default": False
    },
    "nli": {
        "name": "nli:latest",
        "alias": "nli",
        "description": "Natural Language Inference zero-shot classification model.",
        "parameter_size": "400M",
        "vram_footprint_mb": 880,
        "avg_latency_ms": 380,
        "throughput_fps": 2.63,
        "reasoning_tier": "entailment_classification",
        "decision_quality": "Good for zero-shot hypothesis entailment.",
        "best_for": ["Zero-shot premise/hypothesis testing"],
        "is_default": False
    },
    "gliclass": {
        "name": "gliclass:latest",
        "alias": "gliclass",
        "description": "Zero-shot text classification model with arbitrary taxonomy support.",
        "parameter_size": "300M",
        "vram_footprint_mb": 650,
        "avg_latency_ms": 280,
        "throughput_fps": 3.57,
        "reasoning_tier": "zero_shot_classification",
        "decision_quality": "High throughput for arbitrary taxonomies.",
        "best_for": ["Zero-shot topic classification"],
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

async def get_live_models_capabilities(ollaya_url: str) -> Dict[str, Dict[str, Any]]:
    """
    Fetches installed models dynamically from Ollaya and merges with benchmark profiles.
    """
    installed_models: Dict[str, Dict[str, Any]] = {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ollaya_url}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("models", []):
                    raw_name = item.get("name", "")
                    clean_id = raw_name.split(":")[0]
                    # Check exact name or base id
                    meta = MODEL_CAPABILITIES.get(raw_name) or MODEL_CAPABILITIES.get(clean_id) or {
                        "name": raw_name,
                        "alias": clean_id,
                        "description": f"Installed model ({item.get('details', {}).get('format', 'onnx')})",
                        "parameter_size": item.get("details", {}).get("parameter_size", "unknown"),
                        "vram_footprint_mb": round(item.get("size", 0) / (1024 * 1024)),
                        "avg_latency_ms": 1000,
                        "throughput_fps": 1.0,
                        "reasoning_tier": "standard",
                        "best_for": ["General decision making"],
                        "is_default": False
                    }
                    meta_copy = dict(meta)
                    meta_copy["installed"] = True
                    meta_copy["size_bytes"] = item.get("size")
                    meta_copy["modified_at"] = item.get("modified_at")
                    meta_copy["format"] = item.get("details", {}).get("format")
                    installed_models[clean_id] = meta_copy
                    if raw_name != clean_id:
                        installed_models[raw_name] = meta_copy
    except Exception:
        pass

    # If upstream query failed or returned empty, fallback to cached capabilities
    if not installed_models:
        for k, v in MODEL_CAPABILITIES.items():
            meta_copy = dict(v)
            meta_copy["installed"] = True
            installed_models[k] = meta_copy

    return installed_models
