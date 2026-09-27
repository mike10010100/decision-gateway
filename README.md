# Decision Gateway

A unified, SLA-driven API gateway, dynamic capabilities discovery, and model management service for **System-One Decision Models** (Julia, Decider, Laya, Kev, GLiClass, Qwen3Guard, NLI), running locally with zero cloud cost.

GitHub Repository: **[https://github.com/mike10010100/decision-gateway](https://github.com/mike10010100/decision-gateway)** (Public)

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
         ┌───────────────────┬─────────────┴─────────────┬───────────────────┐
         │                   │                           │                   │
   Pattern 1 (MCP)     Pattern 2 (Capabilities)    Pattern 3 (Auto)    Model Management
  SSE Streamable MCP   Live Model Manifest         SLA & Latency       Pull/Delete APIs
  Tool definitions     FPS & Latency Profiles      Smart Auto-Router   Auto-Preload
         │                   │                           │                   │
         └───────────────────┴─────────────┬─────────────┴───────────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     │          Ollaya Inference Server          │
                     │  Port 11435 (REST) │ Port 11436 (MCP)     │
                     │  Model Storage: ${MODELS_PATH:-./data/models} │
                     └───────────────────────────────────────────┘
```

---

## Interactive Documentation & OpenAPI

* **Swagger UI (Interactive API Explorer):** `http://<server-ip>:8000/docs` or `http://localhost:8000/docs`
* **ReDoc (Reference Documentation):** `http://<server-ip>:8000/redoc` or `http://localhost:8000/redoc`
* **Raw OpenAPI 3.1 Spec:** `http://<server-ip>:8000/openapi.json`
* **Live Capabilities Manifest:** `http://<server-ip>:8000/v1/capabilities`

---

## Advertised Models & Performance

| Model | Parameters | VRAM | Avg Latency | Throughput (FPS) | Reasoning Tier & Best Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`julia`** | 144M | 380 MB | **~100 ms** | **10.0 FPS** | **`multilingual_compact`**: Ultra-fast edge loops, 52-locale classification, emotion/sentiment routing. |
| **`decider`** *(default)* | 1.9B | 3.8 GB | **~1,670 ms** | **0.60 FPS** | **`deep_causal`**: High accuracy triage, customer frustration, implicit severity. |
| **`laya`** | 421M | 850 MB | **~304 ms** | **3.29 FPS** | **`surface_keyword`**: High-throughput edge loops, spam/profanity, simple keyword routing. |
| **`gliclass`** | 300M | 650 MB | **~280 ms** | **3.57 FPS** | **`zero_shot_classification`**: High-throughput arbitrary taxonomy classification. |
| **`nli`** | 400M | 880 MB | **~380 ms** | **2.63 FPS** | **`entailment_classification`**: Zero-shot premise/hypothesis testing. |
| **`qwen3guard`** | 1.5B | 1.5 GB | **~950 ms** | **1.05 FPS** | **`safety_guardrail`**: Safety moderation, content policy guardrails. |
| **`kev:4b`** | 4.0B | 9.5 GB | **~4,330 ms** | **0.23 FPS** | **`deep_pointer_head`**: Jared Palmer pointer-head span extraction workflows. |
| **`kev`** *(0.8B)* | 0.76B | 1.8 GB | **~667 ms** | **1.50 FPS** | **`lightweight_pointer_head`**: Experimental pointer-head evaluation. |

---

## Quick Model Management & Automated Downloads

### 1. Using the REST API (Any Remote Client)
Download and install any model from the registry over HTTP:
```bash
# Pull a model (buffered)
curl -s -X POST http://localhost:8000/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"model": "gliclass"}' | jq .

# Pull a model with streaming NDJSON progress
curl -N -X POST http://localhost:8000/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3guard", "stream": true}'

# Remove a model from local storage
curl -s -X DELETE http://localhost:8000/v1/models/gliclass | jq .
```

### 2. Using the CLI Helper Script (`./scripts/models.sh`)
```bash
# List all installed models and their live FPS ratings
./scripts/models.sh list

# Pull a new model
./scripts/models.sh pull gliclass

# Delete a model
./scripts/models.sh rm gliclass

# Run a quick test decision
./scripts/models.sh test decider
```

### 3. Startup Automated Preloading
In `docker-compose.yml`, configure the `PRELOAD_MODELS` environment variable:
```yaml
environment:
  - PRELOAD_MODELS=decider,laya,kev:4b,gliclass
```
On boot, the gateway inspects installed models and automatically downloads any missing entries in the background.

---

## Pattern 1: For Remote AI Agents (MCP & Tool Specs)

### 1. Model Context Protocol (MCP)
Remote agents (Claude Desktop, Cursor, LangChain, AutoGen) can connect directly to the streamable SSE endpoint:
```text
http://localhost:8000/mcp
```

### 2. OpenAI / Anthropic Function Calling Schema
Fetch standard JSON tool definitions that any LLM can plug directly into `tools=[...]`:
```bash
curl http://localhost:8000/v1/tools | jq .
```

---

## Pattern 2: Capabilities & SLA Manifest API

Remote microservices can inspect available models and hardware benchmarks on startup:
```bash
# Full manifest with hardware stats, benchmarks, and routing rules:
curl http://localhost:8000/v1/capabilities | jq .

# Standard model catalog list:
curl http://localhost:8000/v1/models | jq .
```

---

## Pattern 3: SLA-Driven Auto-Router

Remote callers declare an **SLA profile** or **latency budget**:

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

### 3. Ultra-Fast Multilingual Edge Routing (`sla: "ultrafast"`)
Routes to **`julia`** for sub-150ms execution (~10 FPS) across 52 languages:
```bash
curl -s -X POST http://localhost:8000/v1/auto \
  -H "Content-Type: application/json" \
  -d '{
    "sla": "ultrafast",
    "state": "Eu gostaria de alterar a data do meu voo para amanhã cedo.",
    "questions": {
      "department": {
        "type": "choice",
        "criteria": {
          "flights": "Reservas e alterações de voos",
          "luggage": "Bagagem extraviada",
          "support": "Atendimento geral"
        }
      }
    }
  }' | jq .
```

---

## Docker Compose Management

The whole stack auto-starts on boot with `restart: unless-stopped`.

```bash
cd ~/decision-gateway

# Check container status
docker compose ps

# View live logs
docker compose logs -f

# Rebuild and restart
docker compose up -d --build
```
