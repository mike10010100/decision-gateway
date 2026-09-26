import os

# Server Configuration
GATEWAY_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8000"))

# Upstream Ollaya Service Configuration
OLLAYA_URL = os.getenv("OLLAYA_URL", "http://127.0.0.1:11435")
OLLAYA_MCP_URL = os.getenv("OLLAYA_MCP_URL", "http://127.0.0.1:11436")

# Automated Startup Preload Models (comma-separated)
PRELOAD_MODELS = [
    m.strip() for m in os.getenv("PRELOAD_MODELS", "decider,laya,kev:4b").split(",") if m.strip()
]

# Hardware Description
DEVICE_INFO = {
    "name": os.getenv("DEVICE_NAME", "Accelerated Compute Host 64GB"),
    "architecture": os.getenv("DEVICE_ARCH", "aarch64 (ARM Cortex 12-core)"),
    "gpu": os.getenv("DEVICE_GPU", "NVIDIA GPU (Ampere architecture, Compute Capability 8.7)"),
    "memory_total_gb": int(os.getenv("DEVICE_MEMORY_GB", "64")),
    "memory_type": os.getenv("DEVICE_MEMORY_TYPE", "Unified LPDDR5"),
    "cuda_version": os.getenv("DEVICE_CUDA_VERSION", "12.6"),
    "network_hostname": os.getenv("DEVICE_HOSTNAME", "localhost")
}
