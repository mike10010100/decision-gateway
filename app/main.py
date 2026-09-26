"""
Decision Gateway Main FastAPI Application
Provides:
  - Pattern 1: MCP Server reverse-proxy & Tool definitions (/mcp, /v1/tools)
  - Pattern 2: Capabilities & SLA Manifest API (/, /v1/capabilities, /v1/models)
  - Pattern 3: SLA-Driven Smart Gateway & Auto-Router (/v1/systemone, /v1/auto)
"""

import time
import subprocess
import httpx
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    GATEWAY_HOST,
    GATEWAY_PORT,
    OLLAYA_URL,
    OLLAYA_MCP_URL,
    DEVICE_INFO
)
from app.capabilities import MODEL_CAPABILITIES, SLA_ROUTING_PROFILES
from app.router import resolve_model

app = FastAPI(
    title="Decision Gateway",
    description="SLA-driven Gateway and Capabilities Manifest for System-One Decision Models",
    version="1.0.0"
)

# Enable CORS so any browser app or remote web client can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def ensure_upstream_mcp_running():
    """Checks if Ollaya's MCP server is running on port 11436; if not, triggers it in docker."""
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(f"{OLLAYA_MCP_URL}/mcp")
            if resp.status_code in [405, 406, 400]:
                return
    except Exception:
        pass

    try:
        subprocess.Popen(
            ["docker", "exec", "-d", "ollaya", "ollaya", "mcp", "--http", "0.0.0.0:11436"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"Warning: Could not start upstream MCP server: {e}")

@app.on_event("startup")
async def startup_event():
    ensure_upstream_mcp_running()

@app.get("/healthz")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}

# ---------------------------------------------------------------------------
# Pattern 2: Capabilities & SLA Manifest API
# ---------------------------------------------------------------------------

@app.get("/")
@app.get("/v1/capabilities")
async def get_capabilities():
    """
    Returns the complete capability manifest advertising available models,
    benchmark performance (FPS, latency), reasoning tiers, and SLA routing rules.
    """
    return {
        "service": "Decision Gateway",
        "device": DEVICE_INFO,
        "models": MODEL_CAPABILITIES,
        "sla_profiles": SLA_ROUTING_PROFILES,
        "recommendations": {
            "default_model": "decider",
            "fastest_model": "laya",
            "smartest_model": "decider"
        },
        "endpoints": {
            "decide": "/v1/systemone (or /v1/auto)",
            "capabilities": "/v1/capabilities",
            "models": "/v1/models",
            "mcp": "/mcp",
            "tools": "/v1/tools"
        }
    }

@app.get("/v1/models")
async def list_models():
    """TypeSafe / OpenAI compatible model list enriched with throughput & latency."""
    models_list = []
    for model_id, meta in MODEL_CAPABILITIES.items():
        models_list.append({
            "id": model_id,
            "name": meta["name"],
            "description": meta["description"],
            "parameter_size": meta["parameter_size"],
            "latency_ms": meta["avg_latency_ms"],
            "throughput_fps": meta["throughput_fps"],
            "reasoning_tier": meta["reasoning_tier"],
            "is_default": meta.get("is_default", False)
        })
    return {"object": "list", "data": models_list}

# ---------------------------------------------------------------------------
# Pattern 3: SLA-Driven Smart Gateway & Auto-Router
# ---------------------------------------------------------------------------

@app.post("/v1/systemone")
@app.post("/v1/auto")
async def make_decision(request: Request):
    """
    Evaluates a state with typed questions. Supports:
      - Automatic SLA-based routing ('sla': 'fast' | 'smart' | 'accurate')
      - Latency budget routing ('max_latency_ms': 500)
      - Explicit model selection ('model': 'decider' | 'laya' | 'kev:4b')
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON body")

    state = body.get("state")
    questions = body.get("questions")
    if not state or not questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'state' and 'questions' are required fields"
        )

    # Extract routing parameters
    requested_model = body.get("model")
    requested_sla = body.get("sla")
    requested_max_latency = body.get("max_latency_ms")

    # Resolve target model
    selected_model, routing_reason = resolve_model(
        model=requested_model,
        sla=requested_sla,
        max_latency_ms=requested_max_latency
    )

    # Build upstream payload for Ollaya
    upstream_payload = {
        "model": selected_model,
        "state": state,
        "questions": questions
    }

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{OLLAYA_URL}/v1/systemone",
                json=upstream_payload,
                headers={"Content-Type": "application/json"}
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error connecting to upstream Ollaya inference server: {exc}"
        )

    elapsed_ms = (time.time() - t0) * 1000.0

    if resp.status_code != 200:
        return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")

    result_json = resp.json()

    # Get model info for headers
    model_meta = MODEL_CAPABILITIES.get(selected_model, {})
    model_fps = model_meta.get("throughput_fps", 0)

    # Enrich response with routing telemetry
    result_json["routing"] = {
        "selected_model": selected_model,
        "reason": routing_reason,
        "execution_time_ms": round(elapsed_ms, 1),
        "rated_fps": model_fps
    }

    response_headers = {
        "X-Selected-Model": selected_model,
        "X-Execution-Time-Ms": f"{elapsed_ms:.1f}",
        "X-Rated-FPS": f"{model_fps:.2f}"
    }

    return JSONResponse(content=result_json, headers=response_headers)

# ---------------------------------------------------------------------------
# Pattern 1: Model Context Protocol (MCP) & Agent Tools
# ---------------------------------------------------------------------------

@app.api_route("/mcp", methods=["GET", "POST"])
async def mcp_proxy(request: Request):
    """
    Proxies streaming SSE / JSON-RPC requests directly to the Ollaya MCP server.
    Allows remote AI agents to discover and invoke decision tools.
    """
    url = f"{OLLAYA_MCP_URL}/mcp"
    headers = {key: value for key, value in request.headers.items() if key.lower() != "host"}
    params = dict(request.query_params)
    body = await request.body()

    try:
        client = httpx.AsyncClient(timeout=120.0)
        client_req = client.build_request(
            method=request.method,
            url=url,
            headers=headers,
            params=params,
            content=body
        )
        upstream_resp = await client.send(client_req, stream=True)

        async def stream_generator():
            try:
                async for chunk in upstream_resp.aiter_raw():
                    yield chunk
            finally:
                await upstream_resp.aclose()
                await client.aclose()

        return StreamingResponse(
            stream_generator(),
            status_code=upstream_resp.status_code,
            headers=dict(upstream_resp.headers)
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unable to connect to upstream MCP server: {exc}"
        )

@app.get("/v1/tools")
async def get_agent_tools():
    """
    OpenAI / Anthropic compatible function-calling tool schema.
    Remote LLM agents can plug this directly into their tools parameter.
    """
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "system_one_decision",
                    "description": (
                        "Fast structured decision model running on local hardware. "
                        "Evaluates state against typed questions (choice, noul/boolean, score) without generating text. "
                        "Choose model based on difficulty: 'decider' (default, high accuracy, ~1.6s) for complex logic/triage; "
                        "'laya' (sub-350ms, ~3.3 FPS) for simple keyword classification or edge routing."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "state": {
                                "type": "string",
                                "description": "The text or document state to evaluate."
                            },
                            "sla": {
                                "type": "string",
                                "enum": ["smart", "fast", "accurate"],
                                "description": "Performance SLA: 'smart' (decider, 1.6s), 'fast' (laya, 300ms)."
                            },
                            "model": {
                                "type": "string",
                                "enum": ["auto", "decider", "laya", "kev:4b"],
                                "description": "Specific model override (default: auto)."
                            },
                            "questions": {
                                "type": "object",
                                "description": "Dictionary of typed questions (e.g. choice, noul).",
                                "additionalProperties": True
                            }
                        },
                        "required": ["state", "questions"]
                    }
                }
            }
        ]
    }
