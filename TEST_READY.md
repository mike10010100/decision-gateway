# TEST READY: Opaque-Box E2E Test Suite Sign-Off

**Date**: 2026-09-26  
**Agent**: `test_writer_e2e` (`teamwork_preview_test_writer`)  
**Status**: **READY** (Test Infrastructure & Tiers 1-4 Test Suites Complete)  
**Test Harness**: `pytest` + `pytest-asyncio` + `respx` + `httpx`  
**Execution Mode**: 100% Offline & Hermetic  

---

## 1. Test Suite Summary

The independent, opaque-box E2E test suite has been designed, implemented, and verified against the interface contracts in `ORIGINAL_REQUEST.md` and `PROJECT.md`.

- **Total Test Cases in Repository**: **226 tests** (exceeds requirement of ~115 test cases)
- **New Modular E2E Test Cases**: **115 tests** across 6 dedicated test modules
- **Current Baseline Status**: **208 PASSED**, **18 FAILED** (clean execution in 10.59s, 0 crashes)
- **Failure Analysis**: All 18 failing tests are verified implementation gaps in unhardened code, serving as exact acceptance criteria for Milestone M2 (Upstream Fault Tolerance) and Milestone M3 (Stream Lifecycle).

---

## 2. Test File Inventory & Feature Mapping

| Test File | Target Features | Tier Breakdown | Test Count | Pass | Fail | Milestone Dependency |
|-----------|-----------------|----------------|------------|------|------|----------------------|
| `tests/test_e2e_validation.py` | **F1**: Strict Schema Validation & 400 Bad Request | Tier 1 (10), Tier 2 (25) | 35 | 35 | 0 | **M1** (VERIFIED PASS) |
| `tests/test_e2e_capabilities.py` | **F3**: Capabilities Resilience & Corrupt Data Isolation | Tier 1 (10), Tier 2 (7) | 17 | 17 | 0 | **M2** (VERIFIED PASS) |
| `tests/test_e2e_upstream_resilience.py` | **F2**: Upstream Fault Tolerance & Status Code Mapping | Tier 1 (14), Tier 2 (10) | 24 | 10 | 14 | **M2** (Pending implementation) |
| `tests/test_e2e_streaming.py` | **F4, F5, F6**: SSE Proxy, Disconnect Cleanup, Mid-Stream Errors | Tier 1 (10), Tier 2 (8) | 18 | 17 | 1 | **M3** (Pending hop-by-hop stripping) |
| `tests/test_e2e_cross_feature.py` | **F7, F8, F10**: Cross-Feature Combinations & Concurrency | Tier 3 (13) | 13 | 11 | 2 | **M2, M3** (Pending 502/504 mapping) |
| `tests/test_e2e_scenarios.py` | **F10**: Real-World Production Decision Workloads | Tier 4 (8) | 8 | 7 | 1 | **M2** (Pending 502 mapping) |
| `tests/test_gateway.py` | Baseline unit router and manifest tests | Unit | 11 | 11 | 0 | Baseline |
| `tests/test_router.py` | Unit SLA routing and latency logic | Unit | 23 | 23 | 0 | Baseline |
| `tests/test_validation.py` | Unit validation checks | Unit | 77 | 77 | 0 | Baseline |
| **TOTAL** | **Features F1 to F10** | **Tiers 1-4** | **226** | **208** | **18** | **Ready for M2/M3 completion** |

---

## 3. Four-Tier Coverage Breakdown

### Tier 1: Feature Coverage (Features F1 through F10)
- Valid decision evaluations across all question types (`choice`, `score`, `noul`, `multi`).
- Dynamic model resolution via SLA (`fast` -> `laya`, `smart`/`accurate` -> `decider`, `pointer` -> `kev:4b`).
- Latency budget routing (`<= 500ms` -> `laya`, `501-2500ms` -> `decider`).
- Explicit model overrides taking precedence over SLA.
- Dynamic capabilities discovery via `GET /v1/capabilities` and `GET /v1/models`.
- Function calling tool definitions via `GET /v1/tools`.
- Streaming SSE MCP proxy on `/mcp` with keepalive pings (`:\n\n`).
- Model download trigger via `POST /v1/models/pull` (buffered and streaming NDJSON).
- Model removal via `DELETE /v1/models/{model_name}`.

### Tier 2: Boundary & Corner Cases (Features F1 through F10)
- Empty payload bodies (`{}`).
- Empty string states (`""`) and whitespace-only states (`"   "`).
- Non-dict payload roots (arrays `[1, 2, 3]`, primitive strings `"string"`).
- Non-string states (integers `123`, lists `["a", "b"]`).
- Empty questions dictionaries (`{}`).
- Missing question types and unrecognized question types.
- Choice questions with missing criteria or criteria with fewer than 2 choices.
- Score questions with missing criteria anchor lists.
- Invalid SLA types (integer `123`, boolean `True`) and unrecognized SLA values (`"ultra_fast"`).
- Non-integer or string max latency budgets (`"fast"`), zero budgets (`0`), and negative budgets (`-50`).
- Upstream model metadata corruption: `details: null` (isolated without crash), null sizes, missing fields.
- Non-boolean `stream` parameters on `/v1/models/pull` (`"not_a_bool"`).
- Large SSE stream chunks (32KB data frames) and empty streams (0 bytes).

### Tier 3: Cross-Feature Combinations
- SLA routing combined with upstream timeouts (expected 504).
- SLA routing combined with upstream 500 server crashes (expected 502).
- Max latency budget routing combined with upstream network disconnects (expected 502).
- High concurrency: 20 simultaneous requests with mixed SLAs (`fast`, `smart`, `accurate`, `pointer`) asserting header and routing isolation.
- Concurrency under partial upstream failures: simultaneous successes, timeouts, and network disconnects handled without process termination.
- Gateway health check probe (`GET /healthz`) remains 200 OK during total upstream inference outage.
- Concurrent streaming SSE on `/mcp` while dispatching decision queries on `/v1/auto`.

### Tier 4: Real-World Application Scenarios
1. **Customer Support Incident Triage**: End-to-end evaluation of enterprise SSO outage ticket with multi-question payload (category choice, priority score, manager escalation noul) using default SLA (`decider`).
2. **High-Frequency Edge Spam & Moderation Filter**: Sub-500ms edge pipeline evaluating comment spam and profanity routed to `laya`.
3. **Autonomous Incident Response Failover**: Primary `decider` model crashes with 500 CUDA OOM; gateway shields error as 502; client falls back to `laya` and completes triage.
4. **Policy Guardrail & Candidate Span Extraction**: Stage 1 safety check via `qwen3guard` model override; Stage 2 candidate span extraction via SLA `pointer` (`kev:4b`).
5. **Financial Fraud Risk Scoring Pipeline**: Multi-question fraud alert with choice risk tier, 0-100 score, and card freeze boolean under strict 300ms latency budget.
6. **Full Model Lifecycle**: Discovery (`/v1/models`) -> Pull (`/v1/models/pull`) -> Inference (`/v1/auto`) -> Cleanup (`DELETE /v1/models/{model}`).
7. **Multi-Turn Remote Agent MCP Tool Calling**: Tool discovery via `/v1/tools`, handshake on `/mcp`, and decision evaluation via `/v1/auto`.
8. **Multi-Tenant Burst Traffic**: 24 concurrent requests across 3 tenants with different SLAs asserting `X-Selected-Model`, `X-Execution-Time-Ms`, and `X-Rated-FPS` response headers.

---

## 4. Implementation Gap Backlog (Failing Tests to Pass)

The following 18 tests currently fail against the unhardened implementation. They serve as the definitive acceptance checklist for the implementation track:

### Milestone M2 (Upstream Fault Tolerance & Status Code Mapping)
1. `tests/test_e2e_upstream_resilience.py::test_auto_upstream_read_timeout_returns_504` (Expected 504, got 502)
2. `tests/test_e2e_upstream_resilience.py::test_systemone_upstream_read_timeout_returns_504` (Expected 504, got 502)
3. `tests/test_e2e_upstream_resilience.py::test_auto_upstream_connect_timeout_returns_504` (Expected 504, got 502)
4. `tests/test_e2e_upstream_resilience.py::test_auto_upstream_500_crashed_returns_502` (Expected 502, got 500)
5. `tests/test_e2e_upstream_resilience.py::test_auto_upstream_503_unavailable_returns_502` (Expected 502, got 503)
6. `tests/test_e2e_upstream_resilience.py::test_pull_sync_upstream_read_timeout_returns_504` (Expected 504, got 502)
7. `tests/test_e2e_upstream_resilience.py::test_pull_sync_upstream_connect_timeout_returns_504` (Expected 504, got 502)
8. `tests/test_e2e_upstream_resilience.py::test_pull_sync_upstream_500_returns_502` (Expected 502, got 500)
9. `tests/test_e2e_upstream_resilience.py::test_delete_upstream_timeout_returns_504` (Expected 504, got 502)
10. `tests/test_e2e_upstream_resilience.py::test_delete_upstream_500_returns_502` (Expected 502, got 500)
11. `tests/test_e2e_upstream_resilience.py::test_pull_streaming_upstream_connect_error_handled` (Unhandled connect error crash)
12. `tests/test_e2e_upstream_resilience.py::test_pull_streaming_upstream_timeout_handled` (Unhandled timeout crash)
13. `tests/test_e2e_upstream_resilience.py::test_pull_streaming_upstream_500_not_masked_as_200` (Masking upstream 500 as 200)
14. `tests/test_e2e_upstream_resilience.py::test_auto_upstream_malformed_html_500_handled` (Expected 502, got 500)
15. `tests/test_e2e_cross_feature.py::test_sla_fast_routes_to_laya_and_handles_upstream_timeout` (Expected 504, got 502)
16. `tests/test_e2e_cross_feature.py::test_sla_smart_routes_to_decider_and_handles_upstream_500` (Expected 502, got 500)
17. `tests/test_e2e_scenarios.py::test_scenario_incident_response_failover` (Expected 502 on upstream 500, got 500)

### Milestone M3 (Stream Lifecycle & Header Sanitization)
18. `tests/test_e2e_streaming.py::test_mcp_hop_by_hop_headers_stripped` (Hop-by-hop `connection: close` not stripped from proxy response)

---

## 5. Verification Commands

To run the entire suite offline:
```bash
pytest
```

To run only the E2E test files:
```bash
pytest tests/test_e2e_validation.py tests/test_e2e_upstream_resilience.py tests/test_e2e_capabilities.py tests/test_e2e_streaming.py tests/test_e2e_cross_feature.py tests/test_e2e_scenarios.py -v
```

To verify Milestone M1 pass:
```bash
pytest tests/test_e2e_validation.py -v
```

The test infrastructure is fully hermetic, reproducible, and ready to guide implementation through 100% pass rate in Milestone M5.
