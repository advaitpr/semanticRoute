# Semantic Router 🧭

A lightweight, dynamic semantic router in Python that intelligently routes incoming requests to simulated LLM endpoints based on **Intent**, **Complexity**, **Latency**, and **Cost**.

Instead of sending every prompt to the most expensive model, the router analyzes what the user is asking and how hard it is, then picks the cheapest model that can handle the job — saving up to **45% in costs**.

## Architecture

```
User Prompt
     │
     ├──→ Intent Analyzer (embeddings + cosine similarity)
     │         └─→ intent = "code_generation", confidence = 0.12
     │
     ├──→ Complexity Scorer (rule-based heuristics)
     │         └─→ complexity = 0.22, label = "low"
     │
     └──→ Routing Engine
               ├─→ required_capability = 0.77
               ├─→ Score fast-lite  → 0.35 (underpowered)
               ├─→ Score power-pro  → 0.60 (capable ✓)
               └─→ ROUTE TO: power-pro
```

## Project Structure

```
semanticRoute/
├── llm_endpoints.py       # Mock LLM endpoints (cheap vs expensive)
├── intent_analyzer.py     # Embedding-based intent classification
├── complexity_scorer.py   # Rule-based complexity scoring
├── router.py              # Routing engine (combines all signals)
├── test_suite.py          # End-to-end test suite
├── LEARNING_NOTES.md      # Concepts, Q&A, and lessons learned
├── PRODUCTION_READY.md    # Production readiness guide
└── README.md              # This file
```

## Prerequisites

- Python 3.9+
- macOS / Linux / Windows

## Setup

**1. Clone the repository**
```bash
git clone <repo-url>
cd semanticRoute
```

**2. Create a virtual environment**
```bash
python3 -m venv venv
```

**3. Activate the virtual environment**
```bash
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**4. Install dependencies**
```bash
pip install sentence-transformers numpy
```

> **Note**: First run will download the `all-MiniLM-L6-v2` model (~80MB). Subsequent runs use the cached version.

## Usage

### Run Individual Components

Each module can be run standalone to verify it works:

```bash
# Test mock LLM endpoints
python llm_endpoints.py

# Test intent analyzer (requires model download on first run)
python intent_analyzer.py

# Test complexity scorer
python complexity_scorer.py

# Test the routing engine
python router.py
```

### Run the Full Test Suite

```bash
python test_suite.py
```

Expected output:
```
RESULTS SUMMARY
======================================================================

  Intent Classification Accuracy : 12/12 (100%)
  Routing Decision Accuracy      : 11/12 (92%)

COST ANALYSIS (if all prompts were executed)
======================================================================

  Routed to fast-lite : 6 prompts (cheap)
  Routed to power-pro: 6 prompts (expensive)
  Cost savings        : 45%
```

### Use in Your Own Code

```python
from router import SemanticRouter

router = SemanticRouter()

# Basic routing
result = router.route("What is the capital of France?")
print(result["routed_to"])       # "fast-lite"
print(result["analysis"])        # intent, complexity, required capability

# Routing with constraints
result = router.route(
    "Explain quantum computing in detail",
    max_cost=0.05,               # Budget constraint
    max_latency_ms=500,          # Latency constraint
)

# Route and execute (calls the chosen mock endpoint)
result = router.route_and_execute("Write a haiku about rain")
print(result["response"])        # Simulated LLM response
```

## Available Endpoints

| Model | Cost/1K Tokens | Latency | Capability | Use Case |
|-------|---------------|---------|------------|----------|
| `fast-lite` | $0.01 | ~150ms | 0.6 | Simple Q&A, summarization, simple creative |
| `power-pro` | $0.10 | ~800ms | 0.95 | Complex reasoning, code generation, detailed tasks |

## Intent Categories

| Intent | Description | Capability Floor |
|--------|-------------|-----------------|
| `simple_qa` | Factual questions | 0.3 |
| `summarization` | Condensing text | 0.4 |
| `creative_writing` | Stories, poems, copy | 0.5 |
| `code_generation` | Writing/debugging code | 0.7 |
| `complex_reasoning` | Analysis, comparison | 0.8 |

## How Routing Works

1. **Classify intent** — Embeddings + cosine similarity match the prompt to an intent category
2. **Score complexity** — 6 weighted heuristics (word count, keywords, etc.) produce a 0–1 score
3. **Compute required capability** — `floor + (1 - floor) × complexity`
4. **Score each endpoint** — Capable endpoints score 0.60–1.00; underpowered ones score 0.00–0.50
5. **Pick the winner** — Highest score wins, with hard constraints (cost/latency) as vetoes

## Learning Notes

See [LEARNING_NOTES.md](LEARNING_NOTES.md) for detailed explanations of:
- How embeddings and cosine similarity work
- Why we chose rule-based complexity scoring over ML
- The scoring evolution from naive weights to hard-gate approach
- Key concepts and lessons learned at each step

