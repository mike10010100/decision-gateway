# Decision Gateway

A unified, SLA-driven API gateway and capabilities discovery service for **System-One Decision Models** (Kev, Decider, Laya), running locally with zero cloud cost.

---

## Architecture Overview

```
                      [ Remote AI Agents / Microservices / Apps ]
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     │            Decision Gateway               │
                     │         (Port 8000 on localhost)          │
                     └─────────────────────┬─────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         │                                 │                                 │
   Pattern 1 (MCP)              Pattern 2 (Capabilities)          Pattern 3 (Auto-Routing)
  SSE / Streamable MCP             Live Model Manifest               SLA & Latency Budget
  Tool definitions (/tools)        FPS / Latency Profiles            Smart Router (/auto)
         │                                 │                                 │
         └─────────────────────────────────┼─────────────────────────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     │          Ollaya Inference Server          │
                     │   decider (0.60 FPS) │ laya (3.29 FPS)   │
                     │   kev:4b (0.23 FPS)  │ kev (1.50 FPS)    │
                     └───────────────────────────────────────────┘
```

---

## Advertised Models & Performance

| Model | Parameters | VRAM | Avg Latency | Throughput (FPS) | Reasoning Tier & Best Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`decider`** *(default)* | 1.9B | 3.8 GB | **~1,670 ms** | **0.60 FPS** | **`deep_causal`**: High accuracy triage, customer frustration, implicit severity. |
| **`laya`** | 421M | 850 MB | **~304 ms** | **3.29 FPS** | **`surface_keyword`**: High-throughput loops, spam/profanity, simple keyword routing. |
| **`kev:4b`** | 4.0B | 9.5 GB | **~4,330 ms** | **0.23 FPS** | **`deep_pointer_head`**: Jared Palmer pointer-head span extraction workflows. |
| **`kev`** *(0.8B)* | 0.76B | 1.8 GB | **~667 ms** | **1.50 FPS** | **`lightweight_pointer_head`**: Experimental pointer-head evaluation. |

---

## Pattern 1: For Remote AI Agents (MCP & Tool Specs)

### 1. Model Context Protocol (MCP)
Remote agents (Claude, Cursor, LangChain, AutoGen) can connect directly to the streamable SSE endpoint:
```text
http://<server-ip>:8000/mcp
# or via mDNS:
http://localhost:8000/mcp
```

### 2. OpenAI / Anthropic Function Calling Schema
Fetch standard JSON tool definitions that any LLM can plug directly into `tools=[...]`:
```bash
curl http://localhost:8000/v1/tools | jq .
```

---

## Pattern 2: Capabilities & SLA Manifest API

Remote microservices can inspect available models and hardware benchmarks on startup to automatically calibrate their requests:

```bash
# Full manifest with hardware stats, benchmarks, and routing rules:
curl http://localhost:8000/v1/capabilities | jq .

# Standard model catalog list:
curl http://localhost:8000/v1/models | jq .
```

---

## Pattern 3: SLA-Driven Auto-Router

Remote callers do not need to hardcode model names; they can declare an **SLA profile** or **latency budget**:

### 1. Smart Triage (Default / `sla: "smart"`)
Routes to **`decider`** for high accuracy and context awareness:
```bash
curl -s -X POST http://localhost:8000/v1/auto \
  -H "Content-Type: application/json" \
  -d '{
    "sla": "smart",
    "state": "The checkout service crashed with an out of memory error, and customers are angry.",
    "questions": {
      "intent": {
        "type": "choice",
        "criteria": {
          "tech_support": "Software error requiring engineering assistance",
          "billing": "Payment questions"
        }
      },
      "urgent": {
        "type": "noul",
        "criteria": {"yes": "Severe production outage", "no": "Routine request"}
      }
    }
  }' | jq .
```

### 2. High-Throughput Edge Routing (`sla: "fast"`)
Routes to **`laya`** for sub-350ms execution:
```bash
curl -s -X POST http://localhost:8000/v1/auto \
  -H "Content-Type: application/json" \
  -d '{
    "sla": "fast",
    "state": "Spam message promoting cheap watches",
    "questions": {
      "is_spam": {
        "type": "noul",
        "criteria": {"yes": "Promotional or scam spam", "no": "Legitimate communication"}
      }
    }
  }' | jq .
```

### 3. Latency Budget Routing (`max_latency_ms`)
If `max_latency_ms <= 500`, the gateway automatically selects `laya`; otherwise it selects `decider`.

### 4. Telemetry Headers Returned
Every response returns telemetry headers for client logging:
* `X-Selected-Model`: `decider`
* `X-Execution-Time-Ms`: `1648.5`
* `X-Rated-FPS`: `0.60`

---

## Running the Service

### Start the Gateway:
```bash
./run.sh
```

### Run Tests:
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```
