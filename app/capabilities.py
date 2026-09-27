"""
Capabilities Manifest & Model Profiles
Advertises performance, memory, FPS, and reasoning tiers to remote entities.
Supports live dynamic merging with Ollaya's local installed models.
"""

import logging
import re
from typing import Dict, Any
import httpx

logger = logging.getLogger(__name__)


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
        "decision_quality": (
            "High. Accurately deduces implied urgency, intent, and subtle emotions."
        ),
        "best_for": [
            "Customer support triage & incident routing",
            "Complex semantic classification",
            "High-reliability guardrails",
            "Urgency & churn detection",
        ],
        "is_default": True,
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
        "decision_quality": (
            "Fast, but misses subtle context (requires explicit keywords for severity)."
        ),
        "best_for": [
            "High-throughput edge loops (>3 FPS)",
            "Spam and profanity filtering",
            "Dead-simple keyword routing",
            "Low-power or memory-constrained scenarios",
        ],
        "is_default": False,
    },
    "kev:4b": {
        "name": "kev:4b",
        "alias": "kev:4b",
        "description": (
            "Pointer-head decision model on Qwen 4B base with block-causal attention spans."
        ),
        "parameter_size": "4.0B",
        "vram_footprint_mb": 9500,
        "avg_latency_ms": 4333,
        "throughput_fps": 0.23,
        "reasoning_tier": "deep_pointer_head",
        "decision_quality": "High accuracy, but 2.6x slower than Decider on local hardware.",
        "best_for": [
            "Workflows requiring Jared Palmer's Kev pointer-head span extraction",
            "Dynamic candidate span ranking",
        ],
        "is_default": False,
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
        "best_for": ["Medium-speed pointer-head experiments"],
        "is_default": False,
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
        "is_default": False,
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
        "is_default": False,
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
        "is_default": False,
    },
}

SLA_ROUTING_PROFILES: Dict[str, Dict[str, Any]] = {
    "fast": {
        "target_model": "laya",
        "description": (
            "Ultra-low latency (<500ms, ~3.3 FPS) for simple keyword filtering and"
            " high-frequency loops."
        ),
        "max_latency_budget_ms": 500,
    },
    "cost": {
        "target_model": "laya",
        "description": (
            "Resource-efficient profile (~850MB VRAM footprint) optimized for low compute overhead."
        ),
        "max_latency_budget_ms": 500,
    },
    "smart": {
        "target_model": "decider",
        "description": (
            "High accuracy (~1.6s, ~0.6 FPS) with deep context and implied urgency"
            " comprehension. (Recommended default)"
        ),
        "max_latency_budget_ms": 2500,
    },
    "accurate": {
        "target_model": "decider",
        "description": "Same as 'smart' - resolves to Decider for maximum decision fidelity.",
        "max_latency_budget_ms": 2500,
    },
    "balanced": {
        "target_model": "decider",
        "description": "Optimal balance between latency (~1.6s) and deep reasoning quality.",
        "max_latency_budget_ms": 2500,
    },
    "pointer": {
        "target_model": "kev:4b",
        "description": "Full 4B pointer-head evaluation for span-based decisions.",
        "max_latency_budget_ms": 6000,
    },
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
                    try:
                        if not isinstance(item, dict):
                            logger.warning("Skipping corrupt model item (not a dict): %s", item)
                            continue

                        raw_name = item.get("name")
                        if not raw_name or not isinstance(raw_name, str):
                            logger.warning(
                                "Skipping corrupt model item with missing or invalid name: %s",
                                item,
                            )
                            continue

                        # Strip leading/trailing ASCII whitespace and zero-width characters
                        clean_name = raw_name.strip(" \t\n\r\v\f\u200b\ufeff").strip()
                        if not clean_name:
                            logger.warning(
                                "Skipping corrupt model item with empty name: %s",
                                item,
                            )
                            continue

                        # Reject model names exceeding 128 characters
                        # (matches DecisionRequest max_length)
                        if len(clean_name) > 128:
                            logger.warning(
                                "Skipping corrupt model item exceeding max length (128): %s",
                                item,
                            )
                            continue

                        # Reject ASCII control characters (0-31 and 127)
                        if any(ord(c) < 32 or ord(c) == 127 for c in clean_name):
                            logger.warning(
                                "Skipping corrupt model item with control characters: %s",
                                item,
                            )
                            continue

                        # Reject bidirectional overrides and zero-width characters
                        if any(
                            c in clean_name
                            for c in [
                                "\u200b",
                                "\u200c",
                                "\u200d",
                                "\ufeff",
                                "\u202a",
                                "\u202b",
                                "\u202c",
                                "\u202d",
                                "\u202e",
                            ]
                        ):
                            logger.warning(
                                "Skipping corrupt model item with Unicode control/zero-width"
                                " characters: %s",
                                item,
                            )
                            continue

                        # Reject model names with consecutive colons or invalid colon placement
                        if (
                            "::" in clean_name
                            or clean_name.startswith(":")
                            or clean_name.endswith(":")
                        ):
                            logger.warning(
                                "Skipping corrupt model item with invalid colon syntax: %s",
                                item,
                            )
                            continue

                        # Validate structure: base_id[:tag] or host:port/path:tag
                        parts = clean_name.split(":")
                        if len(parts) > 3:
                            logger.warning(
                                "Skipping corrupt model item with multiple colons: %s",
                                item,
                            )
                            continue

                        if len(parts) == 3:
                            # Format: host:port/path:tag
                            clean_id = parts[0].strip()
                            if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$", clean_id):
                                logger.warning(
                                    "Skipping corrupt model item with invalid host: %s",
                                    item,
                                )
                                continue
                            if "/" not in parts[1]:
                                logger.warning(
                                    "Skipping corrupt model item with invalid port/path: %s",
                                    item,
                                )
                                continue
                            port, path = parts[1].split("/", 1)
                            if not port.isdigit() or not re.match(
                                r"^[a-zA-Z0-9][a-zA-Z0-9._-]*(/[a-zA-Z0-9._-]+)*$", path
                            ):
                                logger.warning(
                                    "Skipping corrupt model item with invalid port/path: %s",
                                    item,
                                )
                                continue
                            tag = parts[2].strip()
                            if not tag or not re.match(r"^[a-zA-Z0-9._-]+$", tag):
                                logger.warning(
                                    "Skipping corrupt model item with invalid tag: %s",
                                    item,
                                )
                                continue
                        else:
                            clean_id = parts[0].strip()
                            if not re.match(
                                r"^[a-zA-Z0-9][a-zA-Z0-9._-]*(/[a-zA-Z0-9._-]+)*$", clean_id
                            ):
                                logger.warning(
                                    "Skipping corrupt model item with invalid base id: %s",
                                    item,
                                )
                                continue

                            if len(parts) == 2:
                                tag = parts[1].strip()
                                if not tag or not re.match(r"^[a-zA-Z0-9._-]+$", tag):
                                    logger.warning(
                                        "Skipping corrupt model item with invalid tag: %s",
                                        item,
                                    )
                                    continue

                        details = item.get("details")
                        if details is not None and not isinstance(details, dict):
                            logger.warning(
                                "Skipping corrupt model item with invalid details: %s",
                                item,
                            )
                            continue
                        if details is None:
                            details = {}

                        model_format = details.get("format") or "onnx"
                        param_size = details.get("parameter_size") or "unknown"
                        raw_size = item.get("size")
                        if raw_size is None or not isinstance(raw_size, (int, float)):
                            raw_size = 0
                        vram_mb = round(raw_size / (1024 * 1024))

                        # Check exact name or base id
                        meta = (
                            MODEL_CAPABILITIES.get(clean_name)
                            or MODEL_CAPABILITIES.get(clean_id)
                            or {
                                "name": clean_name,
                                "alias": clean_id,
                                "description": f"Installed model ({model_format})",
                                "parameter_size": param_size,
                                "vram_footprint_mb": vram_mb,
                                "avg_latency_ms": 1000,
                                "throughput_fps": 1.0,
                                "reasoning_tier": "standard",
                                "best_for": ["General decision making"],
                                "is_default": False,
                            }
                        )
                        meta_copy = dict(meta)
                        meta_copy["installed"] = True
                        meta_copy["size_bytes"] = item.get("size")
                        meta_copy["modified_at"] = item.get("modified_at")
                        meta_copy["format"] = details.get("format")
                        installed_models[clean_id] = meta_copy
                        if clean_name != clean_id and not clean_name.endswith(":"):
                            installed_models[clean_name] = meta_copy
                    except Exception as item_exc:
                        logger.warning("Error processing model item %s: %s", item, item_exc)
                        continue
    except Exception as exc:
        logger.warning("Error fetching live model capabilities: %s", exc)

    # If upstream query failed or returned empty, fallback to cached capabilities
    if not installed_models:
        for k, v in MODEL_CAPABILITIES.items():
            meta_copy = dict(v)
            meta_copy["installed"] = True
            installed_models[k] = meta_copy

    return installed_models
