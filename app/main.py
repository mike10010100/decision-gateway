"""
Decision Gateway Main FastAPI Application
Provides:
  - Pattern 1: MCP Server reverse-proxy & Tool definitions (/mcp, /v1/tools)
  - Pattern 2: Dynamic Capabilities & SLA Manifest API (/, /v1/capabilities, /v1/models)
  - Pattern 3: SLA-Driven Smart Gateway & Auto-Router (/v1/systemone, /v1/auto)
  - Model Management & Automation:
    - Auto-preloading of models on startup (via PRELOAD_MODELS)
    - POST /v1/models/pull (Trigger model download)
    - DELETE /v1/models/{model_name} (Remove model)
"""

from contextlib import asynccontextmanager
import time
import asyncio
import subprocess
import httpx
from typing import Any, Set
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.models import DecisionRequest, PullModelRequest
from app.config import OLLAYA_URL, OLLAYA_MCP_URL, PRELOAD_MODELS, DEVICE_INFO
from app.capabilities import MODEL_CAPABILITIES, SLA_ROUTING_PROFILES, get_live_models_capabilities
from app.router import resolve_model
from app.streaming import (
    HOP_BY_HOP_HEADERS,
    SafeStreamingResponse,
    ndjson_stream_bridge,
    sanitize_upstream_headers,
    sse_stream_bridge,
)

# ---------------------------------------------------------------------------
# Background Task Tracking & Lifecycle Management
# ---------------------------------------------------------------------------

background_tasks: Set[asyncio.Task[Any]] = set()


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
        subprocess.run(
            ["docker", "exec", "-d", "ollaya", "ollaya", "mcp", "--http", "0.0.0.0:11436"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            check=False,
        )
    except Exception as e:
        print(f"Warning: Could not start upstream MCP server: {e}")


async def preload_models_background():
    """Checks installed models against PRELOAD_MODELS and pulls any missing ones."""
    try:
        await asyncio.sleep(2.0)  # Wait for upstream Ollaya to initialize
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{OLLAYA_URL}/api/tags")
            installed_names = []
            if resp.status_code == 200:
                installed_names = [m.get("name", "") for m in resp.json().get("models", [])]

        for model_to_preload in PRELOAD_MODELS:
            # Check if model or model:latest is installed
            is_present = any(
                m == model_to_preload or m.startswith(f"{model_to_preload}:")
                for m in installed_names
            )
            if not is_present:
                print(f"Preloading model: '{model_to_preload}'...")
                async with httpx.AsyncClient(timeout=600.0) as client:
                    await client.post(f"{OLLAYA_URL}/api/pull", json={"model": model_to_preload})
                print(f"Preload completed for '{model_to_preload}'")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"Notice: Preload check completed with note: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager:
    - Startup: Checks upstream MCP server, registers tracked background preloading task
      with strong reference retention, and sets up shared client pool in app.state.client.
    - Shutdown: Cancels all tracked background tasks, gathers them cleanly,
      and closes the shared HTTP client pool.
    """
    ensure_upstream_mcp_running()
    task = asyncio.create_task(preload_models_background())
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=5.0),
        limits=httpx.Limits(
            max_connections=100, max_keepalive_connections=20, keepalive_expiry=30.0
        ),
    )

    yield

    for bg_task in list(background_tasks):
        bg_task.cancel()
    if background_tasks:
        await asyncio.gather(*list(background_tasks), return_exceptions=True)
    background_tasks.clear()

    if hasattr(app.state, "client") and app.state.client is not None:
        await app.state.client.aclose()
        app.state.client = None


app = FastAPI(
    title="Decision Gateway",
    description="SLA-driven Gateway and Capabilities Manifest for System-One Decision Models",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global Exception Handlers
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Transforms FastAPI/Pydantic validation errors (which default to HTTP 422)
    into structured HTTP 400 Bad Request responses detailing invalid fields.
    """
    errors = [
        {
            "loc": list(err["loc"]),
            "msg": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": "Validation error",
            "errors": errors,
        },
    )


@app.get("/health")
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
    Returns the complete capability manifest advertising dynamically discovered models,
    benchmark performance (FPS, latency), reasoning tiers, and SLA routing rules.
    """
    live_models = await get_live_models_capabilities(OLLAYA_URL)
    return {
        "service": "Decision Gateway",
        "device": DEVICE_INFO,
        "models": live_models,
        "sla_profiles": SLA_ROUTING_PROFILES,
        "preload_configuration": PRELOAD_MODELS,
        "recommendations": {
            "default_model": "decider",
            "fastest_model": "laya",
            "smartest_model": "decider",
        },
        "endpoints": {
            "decide": "/v1/systemone (or /v1/auto)",
            "capabilities": "/v1/capabilities",
            "models": "/v1/models",
            "pull_model": "/v1/models/pull",
            "delete_model": "/v1/models/{model_name}",
            "mcp": "/mcp",
            "tools": "/v1/tools",
            "docs": "/docs",
        },
    }


@app.get("/v1/models")
async def list_models():
    """TypeSafe / OpenAI compatible model list enriched with throughput & latency."""
    live_models = await get_live_models_capabilities(OLLAYA_URL)
    models_list = []
    for model_id, meta in live_models.items():
        models_list.append(
            {
                "id": model_id,
                "name": meta.get("name", model_id),
                "description": meta.get("description", ""),
                "parameter_size": meta.get("parameter_size", "unknown"),
                "latency_ms": meta.get("avg_latency_ms", 1000),
                "throughput_fps": meta.get("throughput_fps", 1.0),
                "reasoning_tier": meta.get("reasoning_tier", "standard"),
                "installed": meta.get("installed", True),
                "is_default": meta.get("is_default", False),
            }
        )
    return {"object": "list", "data": models_list}


# ---------------------------------------------------------------------------
# Model Management & Automated Download Endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/models/pull")
async def pull_model(req: PullModelRequest, request: Request):
    """
    Triggers automated download/pull of a decision model from the registry.
    """
    model_name = req.model.strip()
    url = f"{OLLAYA_URL}/api/pull"

    if req.stream:
        # Stream the download progress back to caller
        client = httpx.AsyncClient(timeout=900.0)
        try:
            client_req = client.build_request("POST", url, json={"model": model_name})
            upstream_resp = await client.send(client_req, stream=True)
            if upstream_resp.status_code >= 500:
                body = await upstream_resp.aread()
                await upstream_resp.aclose()
                await client.aclose()
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=(
                        f"Upstream model pull server error ({upstream_resp.status_code}): "
                        f"{body.decode('utf-8', errors='replace')}"
                    ),
                )
            if upstream_resp.status_code != 200:
                body = await upstream_resp.aread()
                await upstream_resp.aclose()
                await client.aclose()
                return Response(
                    content=body,
                    status_code=upstream_resp.status_code,
                    media_type="application/json",
                )
        except httpx.TimeoutException as exc:
            await client.aclose()
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Upstream model pull timed out: {exc}",
            )
        except httpx.RequestError as exc:
            await client.aclose()
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Upstream model pull connection failed: {exc}",
            )

        return SafeStreamingResponse(
            ndjson_stream_bridge(request, upstream_resp, client=client),
            status_code=upstream_resp.status_code,
            headers=upstream_resp.headers,
            media_type="application/x-ndjson",
        )
    else:
        # Synchronous/buffered wait
        try:
            async with httpx.AsyncClient(timeout=900.0) as client:
                resp = await client.post(url, json={"model": model_name})
                if resp.status_code >= 500:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=(
                            f"Upstream model pull server error ({resp.status_code}): {resp.text}"
                        ),
                    )
                if resp.status_code == 200:
                    return {
                        "status": "success",
                        "message": (
                            f"Model '{model_name}' downloaded successfully and ready for use."
                        ),
                        "model": model_name,
                    }
                else:
                    return Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        media_type="application/json",
                    )
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Upstream model pull timed out: {exc}",
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Upstream model pull unavailable: {exc}",
            )


@app.delete("/v1/models/{model_name}")
async def delete_model(model_name: str):
    """
    Removes a downloaded model to free disk space on NVMe storage.
    """
    url = f"{OLLAYA_URL}/api/delete"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request("DELETE", url, json={"model": model_name})
            if resp.status_code >= 500:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=(
                        f"Upstream model delete server error ({resp.status_code}): {resp.text}"
                    ),
                )
            if resp.status_code == 200:
                return {"status": "success", "message": f"Model '{model_name}' removed."}
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="application/json",
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Upstream model delete timed out: {exc}",
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream model delete unavailable: {exc}",
        )


# ---------------------------------------------------------------------------
# Pattern 3: SLA-Driven Smart Gateway & Auto-Router
# ---------------------------------------------------------------------------


@app.post("/v1/systemone")
@app.post("/v1/auto")
async def make_decision(req: DecisionRequest):
    """
    Evaluates a state with typed questions. Supports:
      - Automatic SLA-based routing ('sla': 'fast' | 'smart' | 'accurate' | 'balanced' | 'cost')
      - Latency budget routing ('max_latency_ms': 500)
      - Explicit model selection ('model': 'decider' | 'laya' | 'kev:4b')
    """
    state = req.state
    questions_payload = {
        qid: (q.model_dump(exclude_none=True) if hasattr(q, "model_dump") else q)
        for qid, q in req.questions.items()
    }

    selected_model, routing_reason = resolve_model(
        model=req.model, sla=req.sla, max_latency_ms=req.max_latency_ms
    )

    upstream_payload = {"model": selected_model, "state": state, "questions": questions_payload}

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{OLLAYA_URL}/v1/systemone",
                json=upstream_payload,
                headers={"Content-Type": "application/json"},
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Upstream inference service timed out: {exc}",
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream inference service unavailable: {exc}",
        )

    elapsed_ms = (time.time() - t0) * 1000.0

    if resp.status_code >= 500:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream inference server error ({resp.status_code}): {resp.text}",
        )

    if resp.status_code != 200:
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type="application/json",
        )

    result_json = resp.json()

    model_meta = MODEL_CAPABILITIES.get(selected_model, {})
    model_fps = model_meta.get("throughput_fps", 0)

    result_json["routing"] = {
        "selected_model": selected_model,
        "reason": routing_reason,
        "execution_time_ms": round(elapsed_ms, 1),
        "rated_fps": model_fps,
    }

    response_headers = {
        "X-Selected-Model": selected_model,
        "X-Execution-Time-Ms": f"{elapsed_ms:.1f}",
        "X-Rated-FPS": f"{model_fps:.2f}",
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
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }
    params = dict(request.query_params)
    body = await request.body()

    client = httpx.AsyncClient(timeout=120.0)
    try:
        client_req = client.build_request(
            method=request.method, url=url, headers=headers, params=params, content=body
        )
        upstream_resp = await client.send(client_req, stream=True)
    except httpx.TimeoutException as exc:
        await client.aclose()
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Upstream MCP server timed out: {exc}",
        )
    except httpx.RequestError as exc:
        await client.aclose()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unable to connect to upstream MCP server: {exc}",
        )

    content_type = upstream_resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        return SafeStreamingResponse(
            sse_stream_bridge(request, upstream_resp, client=client),
            status_code=upstream_resp.status_code,
            headers=upstream_resp.headers,
            media_type="text/event-stream",
        )
    else:
        try:
            content = await upstream_resp.aread()
            clean_headers = sanitize_upstream_headers(upstream_resp.headers)
            return Response(
                content=content,
                status_code=upstream_resp.status_code,
                headers=clean_headers,
                media_type=content_type or "application/json",
            )
        finally:
            await upstream_resp.aclose()
            await client.aclose()


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
                        "Evaluates state against typed questions (choice, noul/boolean, score) "
                        "without generating text. Choose model based on difficulty: 'decider' "
                        "(default, high accuracy, ~1.6s) for complex logic/triage; 'laya' "
                        "(sub-350ms, ~3.3 FPS) for simple keyword classification or edge routing."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "state": {
                                "type": "string",
                                "description": "The text or document state to evaluate.",
                            },
                            "sla": {
                                "type": "string",
                                "enum": ["smart", "fast", "accurate"],
                                "description": (
                                    "Performance SLA: 'smart' (decider, 1.6s), "
                                    "'fast' (laya, 300ms)."
                                ),
                            },
                            "model": {
                                "type": "string",
                                "enum": ["auto", "decider", "laya", "kev:4b"],
                                "description": "Specific model override (default: auto).",
                            },
                            "questions": {
                                "type": "object",
                                "description": "Dictionary of typed questions (e.g. choice, noul).",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["state", "questions"],
                    },
                },
            }
        ]
    }
