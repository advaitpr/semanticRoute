# Production Readiness Guide 🚀

What needs to change to take this learning project from a prototype to a production-grade semantic router.

---

## 1. Replace Mock Endpoints with Real LLM APIs

### Current State
Simulated endpoints with `time.sleep()` and fake responses.

### What to Do
- Integrate with real APIs (OpenAI, Anthropic, Google, etc.) using their SDKs
- Implement **async calls** (`asyncio` / `aiohttp`) — synchronous `time.sleep()` blocks the event loop
- Add **retry logic** with exponential backoff for transient API failures
- Implement **circuit breakers** — if an endpoint fails repeatedly, temporarily remove it from the pool
- Add **streaming support** for long responses

### Example
```python
import openai
import anthropic

class OpenAIEndpoint(LLMEndpoint):
    async def generate(self, prompt: str) -> dict:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return {"response": response.choices[0].message.content, ...}
```

---

## 2. Upgrade the Embedding Model

### Current State
Using `all-MiniLM-L6-v2` (384 dims, ~80MB). Good for learning, but limited.

### What to Do
- **Use a hosted embedding API** (OpenAI `text-embedding-3-small`, Cohere, Voyage) to avoid loading models in memory
- **Cache embeddings** — store pre-computed intent embeddings in Redis or a file, don't recompute on every startup
- **Consider a vector database** (Pinecone, Qdrant, Weaviate) if you scale to hundreds of intent categories
- **Use larger models** (`all-mpnet-base-v2`, 768 dims) for better accuracy if latency allows

### Why It Matters
Our confidence scores were 0.1–0.3. A better model would give 0.5–0.8, making the intent boundaries sharper and reducing misclassifications.

---

## 3. Improve the Complexity Scorer

### Current State
Rule-based with hardcoded keyword lists and fixed weights.

### What to Do
- **Learn the weights from data** — log real prompts with human-labeled complexity, then train a small regression model to find optimal weights
- **Add more signals**:
  - Nested clause depth (parse tree complexity)
  - Named entity count (more entities = more complex)
  - Code block detection (presence of code snippets)
  - Language detection (multi-lingual prompts may need specific models)
- **Use an LLM for meta-scoring** — for ambiguous cases, call a cheap LLM to estimate complexity (meta-routing)

---

## 4. Add Observability & Logging

### Current State
`print()` statements only. No persistence, no metrics.

### What to Do
- **Structured logging** — use `structlog` or `loguru` to log every routing decision as JSON
- **Metrics collection** — track via Prometheus/Grafana or Datadog:
  - Requests per intent category
  - Route distribution (% to each endpoint)
  - Latency per endpoint (actual, not simulated)
  - Cost per request and cumulative spend
  - Intent confidence distribution
- **Tracing** — OpenTelemetry spans for the full pipeline (classify → score → route → execute)
- **Alerting** — alert if intent confidence drops below a threshold (may indicate new, unseen intents)

### Example Metrics Dashboard
```
┌─────────────────────────────────────────────────┐
│ Route Distribution (last 24h)                   │
│   fast-lite: ████████████████░░░░░ 62%          │
│   power-pro: ████████░░░░░░░░░░░░ 38%          │
│                                                 │
│ Avg Latency: fast-lite=142ms  power-pro=823ms   │
│ Daily Cost:  $12.40 (saved $9.80 vs all-pro)    │
│ Intent Confidence: avg=0.45, min=0.08 ⚠️        │
└─────────────────────────────────────────────────┘
```

---

## 5. Add a Feedback Loop

### Current State
No way to know if the routing decision was good or bad.

### What to Do
- **Collect user feedback** — thumbs up/down on responses
- **Track quality signals** — response regeneration rate, follow-up questions (signal of unsatisfactory answer)
- **A/B testing** — randomly route some prompts to both endpoints and compare quality
- **Automatic retraining** — use feedback data to:
  - Add new examples to intent categories
  - Adjust capability floors
  - Tune complexity weights

### Why It Matters
Without feedback, you're flying blind. A router that sends complex queries to cheap models (saving money but giving bad answers) will go undetected.

---

## 6. Add More Endpoint Tiers

### Current State
Only 2 endpoints (cheap and expensive).

### What to Do
Add a **mid-tier** model to fill the gap:

```python
mid_model = LLMEndpoint(
    name="balanced-mid",
    cost_per_token=0.03,
    latency_ms=400,
    capability=0.80,
)
```

Production-grade systems often have 3–5 tiers:
| Tier | Use Case | Example |
|------|----------|---------|
| Nano | Trivial classification, yes/no | GPT-4o-mini |
| Lite | Simple Q&A, summarization | Claude Haiku |
| Mid | Code, moderate reasoning | GPT-4o |
| Pro | Complex analysis, long context | Claude Opus |
| Specialized | Domain-specific fine-tuned models | Custom |

---

## 7. Handle Edge Cases & Safety

### Current State
No input validation, no fallback, no rate limiting.

### What to Do
- **Input validation** — reject empty prompts, enforce max length, sanitize input
- **Fallback routing** — if the preferred endpoint is down, fall back to the next-best
- **Rate limiting** — per-user and per-endpoint throttling
- **Content safety** — run a moderation check before routing (e.g., OpenAI Moderation API)
- **Timeout handling** — if an endpoint doesn't respond within `max_latency_ms`, cancel and reroute
- **Graceful degradation** — if all endpoints are down, return a cached response or error message

---

## 8. Make It a Service

### Current State
Python scripts run from the command line.

### What to Do
- **Wrap in a FastAPI/Flask server** with REST endpoints:
  ```
  POST /route    → returns routing decision
  POST /generate → routes and executes
  GET  /health   → health check
  GET  /metrics  → Prometheus metrics
  ```
- **Containerize** with Docker
- **Deploy** behind a load balancer (Nginx, AWS ALB)
- **Add authentication** — API keys or OAuth for access control
- **Configuration management** — move weights, thresholds, and intent definitions to a config file or database so they can be updated without redeployment

---

## 9. Performance Optimization

### Current State
Synchronous, single-threaded, loads model into memory on every startup.

### What to Do
- **Async throughout** — `asyncio` for non-blocking I/O
- **Connection pooling** — reuse HTTP connections to LLM APIs
- **Model caching** — load the embedding model once and share across requests (singleton pattern)
- **Batch embedding** — if multiple requests arrive close together, batch them for efficiency
- **Pre-warm** — embed intent examples at deployment time, not at request time
- **Response caching** — cache responses for identical/similar prompts (semantic cache using embedding similarity)

### Semantic Caching Example
```
User: "What is the capital of France?"  → cache miss → call LLM → cache response
User: "What's France's capital city?"   → cosine similarity > 0.95 → cache hit → return cached response
```

---

## 10. Testing for Production

### Current State
12 hardcoded test cases.

### What to Do
- **Unit tests** — test each component in isolation with `pytest`
- **Integration tests** — test the full pipeline with mocked API responses
- **Load testing** — use `locust` or `k6` to simulate concurrent users
- **Regression tests** — every time you change weights or intent definitions, run the full suite
- **Shadow mode** — run the router alongside your existing system, compare decisions without affecting users
- **Chaos testing** — simulate endpoint failures, high latency, malformed responses

---

## Priority Roadmap

If you're going to production, tackle these in order:

| Priority | Enhancement | Impact | Effort |
|----------|------------|--------|--------|
| 🔴 P0 | Real LLM API integration | Core functionality | Medium |
| 🔴 P0 | Error handling & fallbacks | Reliability | Low |
| 🟡 P1 | Structured logging & metrics | Observability | Low |
| 🟡 P1 | FastAPI wrapper | Deployability | Medium |
| 🟡 P1 | Add mid-tier endpoint | Better routing | Low |
| 🟢 P2 | Feedback loop | Continuous improvement | High |
| 🟢 P2 | Semantic caching | Cost reduction | Medium |
| 🟢 P2 | Hosted embeddings API | Scalability | Low |
| 🔵 P3 | A/B testing framework | Optimization | High |
| 🔵 P3 | Vector database | Scale to 100+ intents | Medium |

