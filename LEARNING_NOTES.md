# Semantic Router — Learning Notes & Key Concepts

A reference document for concepts, doubts, and explanations encountered while building the semantic router.

---

## Step 1: Mock LLM Endpoints

### Key Concept: Why Mock Endpoints?
- In production, a semantic router sits between the user and multiple LLM APIs (e.g., GPT-4o-mini, GPT-4o, Claude).
- We simulate these with Python objects that mimic real-world properties: **cost**, **latency**, and **capability**.
- This lets us develop and test the routing logic without spending money on API calls.

### Key Concept: Latency Jitter
- Real network calls don't take the exact same time every time. We add ±20% random jitter (`random.uniform(0.8, 1.2)`) to simulate this.
- This is a common technique in system simulations and load testing.

---

## Step 2: Intent Analyzer (Embeddings + Cosine Similarity)

### Q: Where are the embeddings stored after they're computed?

**In memory (RAM)**, inside a Python dictionary called `self.intent_embeddings`.

```python
self.intent_embeddings = {
    "simple_qa":          numpy array of shape (6, 384),
    "complex_reasoning":  numpy array of shape (6, 384),
    "code_generation":    numpy array of shape (6, 384),
    "creative_writing":   numpy array of shape (6, 384),
    "summarization":      numpy array of shape (6, 384),
}
```

- Each intent has **6 example phrases**.
- Each phrase is converted into a **384-dimensional vector** (a list of 384 numbers).
- So each intent's examples become a **matrix of shape (6, 384)** — 6 rows (examples) × 384 columns (dimensions).
- These are **not persisted to disk**. When the program exits, they're gone. On next startup, they're re-computed (which is fast — milliseconds).

### Q: How does cosine similarity work between the prompt and intent definitions?

We do **NOT** compare the prompt against the intent as a whole. We compare against **every individual example phrase**, then **average** the scores.

**Step-by-step flow:**

1. **Embed the prompt** → produces a single vector of 384 numbers.
2. **For each intent**, compute cosine similarity between the prompt vector and **each** of the 6 example vectors → gives 6 similarity scores.
3. **Average** those 6 scores → one mean similarity per intent.
4. The intent with the **highest mean** wins.

**Visual flow:**
```
Prompt: "What is the speed of light?"
                    │
                    ▼
            Embed → [0.025, -0.110, ..., 0.061]  (384 dims)
                    │
        ┌───────────┼───────────┬───────────┬───────────┐
        ▼           ▼           ▼           ▼           ▼
   simple_qa   complex_re..  code_gen   creative   summarize
   (6 examples) (6 examples) (6 examples)(6 examples)(6 examples)
        │           │           │           │           │
   6 cosines    6 cosines   6 cosines   6 cosines   6 cosines
        │           │           │           │           │
     mean()      mean()      mean()      mean()      mean()
        │           │           │           │           │
      0.73        0.31        0.28        0.35        0.40
        │
        ▼
   WINNER! → intent = "simple_qa"
```

### Key Concept: What is an Embedding?

An embedding is a **numerical representation of text** — a list of numbers (a vector) that captures the **semantic meaning** of a sentence.

- The model `all-MiniLM-L6-v2` maps any sentence to a **384-dimensional vector**.
- Sentences with **similar meanings** produce vectors that **point in similar directions**.
- Example: "What is the capital of France?" and "What is the speed of light?" produce vectors that are close together because they're both simple factual questions.

> Think of it as coordinates in "meaning space" — related sentences are nearby neighbors.

### Key Concept: What is Cosine Similarity?

Cosine similarity measures the **angle** between two vectors, ignoring their magnitude (length).

**Formula:**

$$\text{cosine}(A, B) = \frac{\sum_{i=1}^{n} A_i \times B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \;\times\; \sqrt{\sum_{i=1}^{n} B_i^2}}$$

**In plain English:** Multiply each pair of corresponding numbers, sum them up, then divide by both vector lengths.

**Score interpretation:**
| Score | Meaning |
|-------|---------|
| 1.0   | Identical direction → same meaning |
| 0.5–0.8 | Similar meaning |
| ~0.0  | Perpendicular → unrelated |
| -1.0  | Opposite direction → opposite meaning |

**Why cosine and not Euclidean distance?**
- Cosine ignores vector magnitude — it only cares about **direction**.
- Two sentences can have different word counts (different magnitude) but the same meaning (same direction). Cosine handles this correctly.

### Key Concept: Pre-computation Optimization

We embed all example phrases **once at startup** (in `_precompute_embeddings()`), not on every request.

- Startup cost: embed 30 phrases (6 per intent × 5 intents) → ~100ms
- Per-request cost: embed just 1 prompt + compute 30 cosine similarities → ~10ms

This is a common pattern in production ML systems — **precompute everything you can**.

### Key Concept: Why Average Over Multiple Examples?

A single example per intent would be fragile. If your only `simple_qa` example is "What is the capital of France?" and someone asks "Define photosynthesis", the cosine similarity might be low even though it's the same intent.

By having **6 diverse examples**, the average captures the **general shape** of what that intent looks like, not just one specific phrasing.

### Observation: Low Confidence Scores Are Normal

Our test showed confidence scores of 0.1–0.3. This is expected because:
- We're using a **lightweight** model (MiniLM, 384 dims) — not a large model.
- Short sentences have less information to encode → lower absolute similarity.
- What matters is **relative ranking** — the correct intent should have the **highest** score, even if the absolute number is small.

---

## Step 3: Complexity Scorer

### Key Concept: Why Rule-Based and Not ML?

The Intent Analyzer (Step 2) uses ML (embeddings). The Complexity Scorer uses **simple rules**. Why the difference?

- **Intent** is about *meaning* — "What is X?" and "Define X" mean the same thing but look different. ML excels at this.
- **Complexity** is about *structure* — word count, keyword presence, question count. Rules handle this perfectly.
- Rule-based scoring is **interpretable**: the `breakdown` dict shows exactly *why* a score is high. With ML, you'd get a number but not know why.

### Key Concept: Weighted Signal Combination

The complexity score is a **weighted sum** of 6 independent signals:

```
Final = 0.20 × word_count
      + 0.15 × sentence_count
      + 0.15 × question_count
      + 0.20 × reasoning_keywords    ← "compare", "analyze", "trade-offs"
      + 0.20 × technical_keywords    ← "algorithm", "API", "microservice"
      + 0.10 × constraint_keywords   ← "step-by-step", "in detail"
```

Each signal is independently scored 0.0→1.0, then combined. The weights are **tunable** — in production you'd adjust them based on real-world performance data.

### Key Concept: Signal Independence

Each signal captures a different "axis" of complexity:
- A prompt can be **short but technical** (few words, many technical keywords)
- A prompt can be **long but simple** (many words, no reasoning keywords)
- Multiple questions raise complexity even if each question is simple

This multi-signal approach avoids over-relying on any single heuristic.

---

## Step 4: Routing Engine

### Key Concept: Required Capability Formula

The router computes how capable a model needs to be using:

```
required_capability = floor + (1 - floor) × complexity
```

Where `floor` comes from the intent type (e.g., `simple_qa` = 0.3, `code_generation` = 0.7).

**Example**: `code_generation` (floor=0.7) with complexity=0.5:
- `0.7 + (1 - 0.7) × 0.5 = 0.7 + 0.15 = 0.85`
- Need at least 0.85 capability → expensive model

**Example**: `simple_qa` (floor=0.3) with complexity=0.1:
- `0.3 + (1 - 0.3) × 0.1 = 0.3 + 0.07 = 0.37`
- Need only 0.37 capability → cheap model is fine

### Lesson Learned: Naive Weighted Scoring Fails with Few Endpoints

**Problem we hit**: Our first scoring approach was a simple weighted sum:
```
score = 0.50 × capability + 0.30 × cost_efficiency + 0.20 × latency_efficiency
```

With only 2 endpoints, `fast-lite` always scored 1.0 on both cost AND latency efficiency (since it's the cheapest and fastest), giving it a permanent +0.50 bonus. Even when underpowered, it beat `power-pro` every time.

**Fix**: We switched to a **hard gate** approach:
```
If capable (meets requirement):
    score = 0.60 + 0.25 × cost_eff + 0.15 × latency_eff    → range: 0.60–1.00

If underpowered:
    score = (capability/required)² × 0.50                    → range: 0.00–0.50
```

The key insight: **an underpowered endpoint should never beat a capable one**, regardless of cost. The hard gate guarantees this by capping underpowered scores at 0.50, while capable endpoints always score ≥ 0.60.

### Key Concept: Quadratic Penalty

When an endpoint is underpowered, we use `ratio²` instead of just `ratio`:
- `capability=0.6, required=0.86 → ratio=0.70 → penalty=0.49`
- `capability=0.6, required=0.95 → ratio=0.63 → penalty=0.40`

Squaring makes the penalty **grow faster** as the capability gap widens. A small gap is forgivable; a large gap is devastating.

### Key Concept: Hard Constraints vs Soft Scoring

The router has two layers:
1. **Hard constraints** (`max_cost`, `max_latency_ms`): If violated, the endpoint is **excluded** entirely (score = -1). No exceptions.
2. **Soft scoring**: Among non-excluded endpoints, pick the best balance of capability, cost, and latency.

This mirrors real-world systems where budget limits are absolute, but quality preferences are flexible.

---

## Step 5: Testing & Evaluation

### Test Results

| Metric | Score |
|--------|-------|
| Intent Classification | 12/12 (100%) |
| Routing Accuracy | 11/12 (92%) |
| Cost Savings | 45% vs all-expensive routing |

### Lesson Learned: Conservative Routing is a Feature, Not a Bug

The one "miss": `"Fix this: def add(a, b) return a + b"` was routed to `power-pro` instead of `fast-lite`.

**Why it happened**: `code_generation` has a capability floor of 0.7, and `fast-lite` has capability 0.6. Even though the fix is trivial, the router says "this is code → needs at least 0.7 capability."

**Why it's actually fine**: In production, it's better to **over-route** (send a simple task to a powerful model) than to **under-route** (send a hard task to a weak model). Over-routing wastes a little money; under-routing gives the user a bad answer. This is a deliberate design trade-off.

**How to fix if needed**: You could add a sub-intent like `simple_code_fix` with a lower capability floor, or let very low complexity scores override the intent floor.

### Key Concept: Cost Savings Math

```
Without router:  12 prompts × $0.10 = $1.20 per 1K tokens
With router:      6 × $0.01 + 6 × $0.10 = $0.66 per 1K tokens
Savings:          45%
```

In production with thousands of requests, this compounds significantly. Most real-world traffic is simple queries — a good router can route 60–80% to cheap models.

### Final Architecture

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

### Project File Structure

```
semanticRoute/
├── llm_endpoints.py       # Step 1: Mock LLM endpoints
├── intent_analyzer.py     # Step 2: Embedding-based intent classification
├── complexity_scorer.py   # Step 3: Rule-based complexity scoring
├── router.py              # Step 4: Routing engine (combines everything)
├── test_suite.py          # Step 5: End-to-end test suite
├── LEARNING_NOTES.md      # This file — concepts & Q&A reference
└── venv/                  # Python virtual environment
```
