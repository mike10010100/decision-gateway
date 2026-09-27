"""
Comprehensive Test Suite for Decision Gateway
Tests:
  - Pattern 1: Tool definitions and MCP proxy
  - Pattern 2: Capabilities and SLA manifest
  - Pattern 3: SLA-driven Auto-routing and model resolution
"""

import sys
import os
import unittest
from fastapi.testclient import TestClient

# Ensure app package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.router import resolve_model

client = TestClient(app)


class TestRouterLogic(unittest.TestCase):
    def test_default_resolution(self):
        model, reason = resolve_model()
        self.assertEqual(model, "decider")
        self.assertIn("decider", reason)

    def test_sla_fast_resolution(self):
        model, reason = resolve_model(sla="fast")
        self.assertEqual(model, "laya")
        self.assertIn("laya", reason)

    def test_sla_smart_resolution(self):
        model, reason = resolve_model(sla="smart")
        self.assertEqual(model, "decider")

    def test_latency_budget_resolution(self):
        model, _ = resolve_model(max_latency_ms=400)
        self.assertEqual(model, "laya")

        model, _ = resolve_model(max_latency_ms=2000)
        self.assertEqual(model, "decider")

    def test_explicit_model_override(self):
        model, _ = resolve_model(model="kev:4b", sla="fast")
        self.assertEqual(model, "kev:4b")


class TestCapabilitiesAPI(unittest.TestCase):
    def test_healthz(self):
        resp = client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_get_capabilities_manifest(self):
        resp = client.get("/v1/capabilities")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("models", data)
        self.assertIn("decider", data["models"])
        self.assertIn("laya", data["models"])
        self.assertIn("throughput_fps", data["models"]["decider"])
        self.assertIn("avg_latency_ms", data["models"]["decider"])
        self.assertIn("sla_profiles", data)

    def test_list_models(self):
        resp = client.get("/v1/models")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("data", data)
        model_ids = [m["id"] for m in data["data"]]
        self.assertIn("decider", model_ids)
        self.assertIn("laya", model_ids)

    def test_agent_tools_schema(self):
        resp = client.get("/v1/tools")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("tools", data)
        self.assertEqual(data["tools"][0]["function"]["name"], "system_one_decision")


class TestEndToEndDecisionRouting(unittest.TestCase):
    def test_auto_routing_default(self):
        payload = {
            "state": "The user password reset email failed to send.",
            "questions": {
                "intent": {
                    "type": "choice",
                    "criteria": {"email_issue": "Email delivery failure", "auth": "Login failure"},
                }
            },
        }
        resp = client.post("/v1/systemone", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(resp.headers["x-selected-model"], "decider")
        self.assertIn("routing", data)
        self.assertEqual(data["routing"]["selected_model"], "decider")
        self.assertIn("answers", data)

    def test_auto_routing_sla_fast(self):
        payload = {
            "sla": "fast",
            "state": "The user password reset email failed to send.",
            "questions": {
                "intent": {
                    "type": "choice",
                    "criteria": {"email_issue": "Email delivery failure", "auth": "Login failure"},
                }
            },
        }
        resp = client.post("/v1/auto", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["x-selected-model"], "laya")
        data = resp.json()
        self.assertEqual(data["routing"]["selected_model"], "laya")


if __name__ == "__main__":
    unittest.main()
