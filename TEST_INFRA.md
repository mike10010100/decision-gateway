# Test Infrastructure: Decision Gateway

## Overview & Architecture

The `decision-gateway` test suite is designed as an independent, opaque-box, offline-hermetic test harness built with `pytest`, `pytest-asyncio`, `respx`, and `httpx`. It validates all functional, resilience, streaming, and integration contracts specified in `ORIGINAL_REQUEST.md` and `PROJECT.md` without requiring running Docker containers, live accelerator GPUs, or physical network connectivity.

```
+-------------------------------------------------------------------------+
|                              pytest Runner                              |
+------------------------------------+------------------------------------+
                                     |
                +--------------------+--------------------+
                |                                         |
                v                                         v
    FastAPI Gateway Application               respx Mock Router
    - httpx.ASGITransport                     - 127.0.0.1:11435 (Inference & Tags)
    - FastAPI TestClient                      - 127.0.0.1:11436 (MCP SSE Server)
```

## 4-Tier Test Derivation Methodology

Every test case in the suite is derived directly from user requirements and interface contracts across four rigorous tiers:

| Tier | Name | Target Scope | Focus Areas |
|------|------|--------------|-------------|
| **Tier 1** | Feature Coverage | Features F1 to F10 | Happy paths, schema conformance, routing accuracy, model catalog, SSE handshake. |
| **Tier 2** | Boundary & Corner Cases | Features F1 to F10 | Empty strings, whitespace inputs, non-dict payloads, type mismatches, corrupt `details: null`, null sizes, 0ms latency budgets. |
| **Tier 3** | Cross-Feature Combinations | Pairwise & multi-feature interactions | SLA routing + upstream timeouts (504), max latency budget + network disconnects (502), concurrent heterogeneous SLAs, SSE streaming + concurrent decisions. |
| **Tier 4** | Real-World Application Scenarios | End-to-end multi-question pipelines | Customer support triage, edge spam/moderation filtering, incident response failover, financial fraud evaluation, full model lifecycle, agentic MCP tool calling. |

## Modular Test Suite Structure

The test suite is organized into modular files under `tests/`:

```
tests/
├── conftest.py                     # Offline hermetic mock fixtures & test clients
├── test_e2e_validation.py          # Tier 1 & Tier 2: Strict schema validation & 400 Bad Request
├── test_e2e_upstream_resilience.py # Tier 1 & Tier 2: Upstream fault tolerance (502/504 status codes)
├── test_e2e_capabilities.py        # Tier 1 & Tier 2: Dynamic capabilities, corrupt data isolation
├── test_e2e_streaming.py           # Tier 1 & Tier 2: SSE /mcp proxy, client disconnect, NDJSON pull
├── test_e2e_cross_feature.py       # Tier 3: Concurrency, SLA routing + failure combinations
├── test_e2e_scenarios.py           # Tier 4: Realistic real-world production decision pipelines
└── test_gateway.py                 # Baseline router and manifest unit tests
```

### Module Responsibilities

1. **`tests/conftest.py`**:
   - `async_client`: Asynchronous test client using `httpx.ASGITransport(app=app)`.
   - `sync_client`: Synchronous `TestClient(app)`.
   - `mock_ollaya_inference`: Dynamic inference mock mirroring requested model and generating typed question answers.
   - `mock_ollaya_tags`: Standard Ollaya `/api/tags` catalog.
   - `mock_ollaya_tags_corrupt`: Corrupt tags simulation (`details: None`, null sizes).
   - `mock_ollaya_mcp_sse`: Streaming SSE mock on port 11436 with 15s keepalive pings (`:\n\n`).
   - `mock_ollaya_pull` / `mock_ollaya_pull_stream`: Buffered and NDJSON streaming model download mocks.
   - `mock_ollaya_delete`: Model deletion mock.
   - `simulate_network_disconnect`: Helper simulating `httpx.ConnectError`.
   - `simulate_timeout`: Helper simulating `httpx.ReadTimeout` and `httpx.ConnectTimeout`.
   - `simulate_upstream_crashed`: Helper simulating upstream 500 / 503 crashed inference.
   - `simulate_missing_model`: Helper simulating upstream 404 missing model.
   - `sample_payloads`: Fixture providing valid choice, score, noul, and multi-question payloads.
   - `hermetic_upstream_baseline`: Autouse baseline mock ensuring any unmocked test runs 100% offline.

2. **`tests/test_e2e_validation.py` (35 test cases)**:
   - Validates strict schema requirements for `/v1/auto`, `/v1/systemone`, `/v1/models/pull`.
   - Tests missing fields, empty strings, whitespace states, non-dict questions, invalid SLA enums, negative/zero latency budgets, malformed JSON syntax, JSON arrays, and primitive root payloads.
   - Asserts all validation errors return HTTP `400 Bad Request` with structured error details (not 422 or 500).

3. **`tests/test_e2e_upstream_resilience.py` (24 test cases)**:
   - Validates upstream status code mapping and error shields.
   - Tests upstream `ConnectError` -> `502 Bad Gateway`.
   - Tests upstream `ReadTimeout` / `ConnectTimeout` -> `504 Gateway Timeout`.
   - Tests upstream 500 / 503 CUDA crash -> `502 Bad Gateway`.
   - Tests upstream 404 missing model -> structured error without unhandled crash.
   - Tests streaming model pull failure boundaries and error non-masking.

4. **`tests/test_e2e_capabilities.py` (17 test cases)**:
   - Validates `/v1/capabilities`, `/`, `/v1/models`, and `/v1/tools`.
   - Tests partial upstream failures: corrupt model with `"details": null` is isolated without crashing discovery for valid models.
   - Tests fallback to static cached capabilities when upstream `/api/tags` is completely unreachable (ConnectError, timeout, 500).
   - Validates hardware manifest specs and SLA profile definitions.

5. **`tests/test_e2e_streaming.py` (18 test cases)**:
   - Validates SSE `/mcp` proxy and NDJSON pull streaming.
   - Tests SSE keepalive ping forwarding (`:\n\n`), `mcp-session-id` forwarding, and query parameter passthrough.
   - Tests mid-stream upstream restarts (`RemoteProtocolError`) emitting structured SSE error frames.
   - Tests client disconnect cleanup and resource deallocation.
   - Tests hop-by-hop header stripping from proxy responses.

6. **`tests/test_e2e_cross_feature.py` (13 test cases)**:
   - Tests pairwise cross-feature interactions: SLA fast routing + timeout (504), SLA smart routing + 500 crash (502), max latency budget + network disconnect.
   - Tests explicit model override precedence over SLA.
   - Tests 20+ concurrent heterogeneous SLA requests without header leakage.
   - Tests concurrency under partial upstream failure conditions.
   - Tests `/healthz` isolation during complete upstream outage.
   - Tests router type safety invariants.

7. **`tests/test_e2e_scenarios.py` (8 test cases)**:
   - Tests realistic production end-to-end pipelines:
     - Customer support incident triage with choice, score, and noul questions.
     - High-frequency edge spam/moderation filter with sub-500ms SLA.
     - Autonomous incident response failover with fallback model routing.
     - Content policy guardrail evaluation via `qwen3guard` and span extraction via `kev:4b`.
     - Financial fraud alert scoring with strict latency budget.
     - Full model lifecycle (discovery -> pull -> inference -> delete).
     - Multi-turn Agentic MCP tool calling workflow.
     - Multi-tenant burst traffic with header isolation.

## Execution Guide

### Run Full Test Suite
```bash
pytest
```

### Run Specific Test Modules
```bash
pytest tests/test_e2e_validation.py -v
pytest tests/test_e2e_upstream_resilience.py -v
pytest tests/test_e2e_capabilities.py -v
pytest tests/test_e2e_streaming.py -v
pytest tests/test_e2e_cross_feature.py -v
pytest tests/test_e2e_scenarios.py -v
```

### Run by Tier Markers / Keywords
```bash
# Run validation tests
pytest -k "validation or auto or pull"

# Run upstream error resilience tests
pytest -k "upstream or timeout or connect_error"

# Run streaming tests
pytest -k "mcp or sse or streaming"

# Run scenario tests
pytest -k "scenario"
```

## Hermetic Offline Guarantees
- All tests execute without an internet connection or local Docker daemons.
- Ports 11435 and 11436 are intercepted in-memory by `respx`.
- Async operations are executed on the local event loop via `httpx.ASGITransport` directly against the FastAPI ASGI application.
