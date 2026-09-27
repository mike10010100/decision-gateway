"""
E2E Real-World Application Scenarios (Tier 4, Feature F10).
Tests complete realistic end-to-end multi-question pipelines:
- Customer support incident triage & routing
- High-frequency edge spam and moderation filtering
- Autonomous incident response failover workflows
- Content policy guardrail evaluation
- Financial fraud and risk scoring pipeline
- Full model lifecycle (discovery -> pull -> inference -> delete)
- Multi-turn Agentic MCP tool calling workflow
- Multi-tenant burst traffic with header isolation
- Zero-shot taxonomy classification
"""

import sys
import os
import asyncio
import pytest
import httpx
import respx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.config import OLLAYA_URL, OLLAYA_MCP_URL

# ===========================================================================
# Scenario 1: Customer Support Incident Triage & Automated Routing
# ===========================================================================


@pytest.mark.asyncio
async def test_scenario_customer_support_triage(
    async_client: httpx.AsyncClient, mock_ollaya_inference
):
    """
    Scenario 1: End-to-end customer support ticket triage.
    Evaluates incident description against multiple typed questions:
    - category (choice)
    - priority_score (score 1-5)
    - requires_manager_escalation (noul)
    """
    ticket_payload = {
        "state": (
            "URGENT: All single sign-on logins via Okta are failing with SAML response "
            "signature verification errors across the enterprise. 5000+ employees blocked."
        ),
        "sla": "smart",
        "questions": {
            "incident_category": {
                "type": "choice",
                "criteria": {
                    "identity_auth": "Authentication, SSO, SAML, or IAM failure",
                    "billing": "Invoice, credit card, or payment issues",
                    "ui_bug": "Cosmetic or minor display issue",
                },
            },
            "priority_score": {
                "type": "score",
                "criteria": ["1 - minor", "5 - critical enterprise outage"],
                "description": "1 is minor, 5 is critical enterprise outage",
            },
            "requires_manager_escalation": {
                "type": "noul",
                "description": "True if severity requires immediate leadership page",
            },
        },
    }

    resp = await async_client.post("/v1/auto", json=ticket_payload)
    assert resp.status_code == 200
    data = resp.json()

    # Verify routing metadata
    assert data["routing"]["selected_model"] == "decider"
    assert resp.headers.get("x-selected-model") == "decider"

    # Verify answers returned for all 3 questions
    answers = data["answers"]
    assert "incident_category" in answers
    assert "priority_score" in answers
    assert "requires_manager_escalation" in answers


# ===========================================================================
# Scenario 2: High-Frequency Edge Spam & Moderation Filter
# ===========================================================================


@pytest.mark.asyncio
async def test_scenario_edge_spam_moderation_pipeline(
    async_client: httpx.AsyncClient, mock_ollaya_inference
):
    """
    Scenario 2: High-frequency edge loop requiring sub-500ms latency.
    Routes to 'laya' (ModernBERT-large, ~3.3 rated FPS).
    """
    comment_payload = {
        "state": "Click this link to claim free crypto tokens right now: http://bit.ly/scam123",
        "sla": "fast",
        "questions": {
            "is_spam": {
                "type": "noul",
                "description": "True if the text is promotional spam or phishing",
            },
            "has_profanity": {
                "type": "noul",
                "description": "True if offensive language is detected",
            },
        },
    }

    resp = await async_client.post("/v1/auto", json=comment_payload)
    assert resp.status_code == 200
    assert resp.headers.get("x-selected-model") == "laya"
    data = resp.json()
    assert data["routing"]["selected_model"] == "laya"
    assert "is_spam" in data["answers"]
    assert "has_profanity" in data["answers"]


# ===========================================================================
# Scenario 3: Autonomous Incident Response Failover Pipeline
# ===========================================================================


@pytest.mark.asyncio
async def test_scenario_incident_response_failover(
    async_client: httpx.AsyncClient, respx_mock: respx.MockRouter
):
    """
    Scenario 3: Failover pipeline when primary inference model fails.
    Step 1: Primary 'decider' fails with 500 CUDA crash.
    Step 2: Gateway returns 502 Bad Gateway cleanly.
    Step 3: Client falls back to explicit model 'laya'.
    Step 4: Upstream laya succeeds, returning valid triage.
    """
    # Step 1: decider fails
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").respond(
        status_code=500, json={"error": "CUDA out of memory on device 0"}
    )
    req_decider = {
        "state": "Database connection pool exhausted on prod-db-01",
        "sla": "smart",
        "questions": {"severity": {"type": "noul"}},
    }
    resp1 = await async_client.post("/v1/auto", json=req_decider)
    assert resp1.status_code == 502

    # Step 2: Client retries with fallback model 'laya'
    respx_mock.post(f"{OLLAYA_URL}/v1/systemone").respond(
        status_code=200, json={"answers": {"severity": True}, "model": "laya"}
    )
    req_fallback = {
        "state": "Database connection pool exhausted on prod-db-01",
        "model": "laya",
        "questions": {"severity": {"type": "noul"}},
    }
    resp2 = await async_client.post("/v1/auto", json=req_fallback)
    assert resp2.status_code == 200
    assert resp2.headers.get("x-selected-model") == "laya"
    assert resp2.json()["answers"]["severity"] is True


# ===========================================================================
# Scenario 4: Content Policy Guardrail & Pointer-Head Span Evaluation
# ===========================================================================


@pytest.mark.asyncio
async def test_scenario_policy_guardrail_and_pointer_head(
    async_client: httpx.AsyncClient, mock_ollaya_inference
):
    """
    Scenario 4: Multi-stage pipeline:
    Stage 1: Safety check via qwen3guard model override.
    Stage 2: Candidate span extraction via SLA 'pointer' (kev:4b).
    """
    # Stage 1: Safety check
    safety_req = {
        "state": "Candidate query text for moderation",
        "model": "qwen3guard",
        "questions": {"is_safe": {"type": "noul", "description": "True if safe"}},
    }
    resp_safety = await async_client.post("/v1/auto", json=safety_req)
    assert resp_safety.status_code == 200
    assert resp_safety.headers.get("x-selected-model") == "qwen3guard"

    # Stage 2: Pointer evaluation
    pointer_req = {
        "state": "Select the optimal delivery window: [Morning 8-12, Afternoon 12-4, Evening 4-8]",
        "sla": "pointer",
        "questions": {
            "selected_window": {
                "type": "choice",
                "criteria": {"morning": "8-12", "afternoon": "12-4", "evening": "4-8"},
            }
        },
    }
    resp_pointer = await async_client.post("/v1/auto", json=pointer_req)
    assert resp_pointer.status_code == 200
    assert resp_pointer.headers.get("x-selected-model") == "kev:4b"


# ===========================================================================
# Scenario 5: Financial Fraud Risk Scoring Pipeline
# ===========================================================================


@pytest.mark.asyncio
async def test_scenario_financial_fraud_evaluation(
    async_client: httpx.AsyncClient, mock_ollaya_inference
):
    """
    Scenario 5: Multi-question financial fraud evaluation with strict latency budget.
    """
    fraud_payload = {
        "state": (
            "Cardholder normally spends in Austin, TX. Sudden international transaction "
            "of $3,200.00 originating from Lagos, Nigeria at 3:14 AM local time."
        ),
        "max_latency_ms": 300,
        "questions": {
            "risk_tier": {
                "type": "choice",
                "criteria": {
                    "low": "Expected cardholder pattern",
                    "medium": "Unusual location but regular amount",
                    "critical": "High-value cross-border transaction with location mismatch",
                },
            },
            "risk_score": {
                "type": "score",
                "criteria": ["0 - expected cardholder pattern", "100 - critical fraud likelihood"],
                "description": "Fraud likelihood score 0-100",
            },
            "freeze_card": {"type": "noul", "description": "Immediate card block required"},
        },
    }

    resp = await async_client.post("/v1/systemone", json=fraud_payload)
    assert resp.status_code == 200
    # Budget <= 500ms maps to laya
    assert resp.headers.get("x-selected-model") == "laya"
    data = resp.json()
    assert "risk_tier" in data["answers"]
    assert "risk_score" in data["answers"]
    assert "freeze_card" in data["answers"]


# ===========================================================================
# Scenario 6: Full Model Lifecycle & Inference Loop
# ===========================================================================


@pytest.mark.asyncio
async def test_scenario_model_lifecycle_pipeline(
    async_client: httpx.AsyncClient,
    mock_ollaya_tags,
    mock_ollaya_pull,
    mock_ollaya_delete,
    mock_ollaya_inference,
):
    """
    Scenario 6: Full model lifecycle:
    1. Inspect model catalog (GET /v1/models)
    2. Pull model 'gliclass' (POST /v1/models/pull)
    3. Execute decision using pulled model (POST /v1/auto)
    4. Delete model to free disk space (DELETE /v1/models/{model})
    """
    # 1. Inspect catalog
    r_list = await async_client.get("/v1/models")
    assert r_list.status_code == 200

    # 2. Pull model
    r_pull = await async_client.post("/v1/models/pull", json={"model": "gliclass"})
    assert r_pull.status_code == 200
    assert r_pull.json()["status"] == "success"

    # 3. Decision with model
    r_dec = await async_client.post(
        "/v1/auto",
        json={
            "model": "gliclass",
            "state": "Natural language topic classification text",
            "questions": {
                "topic": {
                    "type": "choice",
                    "criteria": {"tech": "Technology", "finance": "Finance"},
                }
            },
        },
    )
    assert r_dec.status_code == 200
    assert r_dec.headers.get("x-selected-model") == "gliclass"

    # 4. Delete model
    r_del = await async_client.delete("/v1/models/gliclass")
    assert r_del.status_code == 200
    assert r_del.json()["status"] == "success"


# ===========================================================================
# Scenario 7: Multi-Turn Remote Agent MCP Tool Calling Loop
# ===========================================================================


@pytest.mark.asyncio
async def test_scenario_agent_tool_calling_workflow(
    async_client: httpx.AsyncClient, mock_ollaya_mcp_sse, mock_ollaya_inference
):
    """
    Scenario 7: Remote AI agent discovers tools, initiates MCP handshake,
    and executes decisions through the gateway.
    """
    # Step 1: Discover tools
    r_tools = await async_client.get("/v1/tools")
    assert r_tools.status_code == 200
    tool_def = r_tools.json()["tools"][0]
    assert tool_def["function"]["name"] == "system_one_decision"

    # Step 2: MCP handshake
    r_handshake = await async_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "initialize", "id": 1},
        headers={"Accept": "application/json"},
    )
    assert r_handshake.status_code == 200

    # Step 3: Execute decision via tool invocation
    r_decision = await async_client.post(
        "/v1/auto",
        json={
            "state": "User requested flight cancellation due to family emergency",
            "questions": {
                "waive_cancellation_fee": {"type": "noul", "description": "Waive policy exception"}
            },
        },
    )
    assert r_decision.status_code == 200
    assert "waive_cancellation_fee" in r_decision.json()["answers"]


# ===========================================================================
# Scenario 8: Multi-Tenant Burst Traffic with Header Isolation
# ===========================================================================


@pytest.mark.asyncio
async def test_scenario_multi_tenant_burst_traffic(
    async_client: httpx.AsyncClient, mock_ollaya_inference
):
    """
    Scenario 8: Burst of 24 concurrent requests across 3 tenants with different SLAs.
    Verifies that response headers and execution metadata are isolated per request.
    """
    requests_meta = [
        {"tenant": "tenant_fast", "sla": "fast", "expected_model": "laya"},
        {"tenant": "tenant_smart", "sla": "smart", "expected_model": "decider"},
        {"tenant": "tenant_pointer", "sla": "pointer", "expected_model": "kev:4b"},
    ] * 8  # 24 requests

    async def run_tenant_request(meta):
        payload = {
            "state": f"Tenant {meta['tenant']} incoming request payload data.",
            "sla": meta["sla"],
            "questions": {"is_valid": {"type": "noul"}},
        }
        resp = await async_client.post("/v1/auto", json=payload)
        return resp, meta

    results = await asyncio.gather(*(run_tenant_request(m) for m in requests_meta))

    for resp, meta in results:
        assert resp.status_code == 200
        assert resp.headers.get("x-selected-model") == meta["expected_model"]
        body = resp.json()
        assert body["routing"]["selected_model"] == meta["expected_model"]
        assert float(resp.headers.get("x-execution-time-ms", 0)) >= 0
