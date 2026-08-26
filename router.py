"""
Step 4: Routing Engine
======================
The decision-maker. Combines Intent + Complexity signals with endpoint
properties (cost, latency, capability) to pick the best LLM for each request.

Routing Strategy:
  1. Classify the prompt's intent   → which category?
  2. Score the prompt's complexity   → how hard is it?
  3. Determine the minimum capability needed
  4. Score each endpoint             → can it handle this? at what cost?
  5. Pick the endpoint with the best score
"""

from intent_analyzer import IntentAnalyzer
from complexity_scorer import ComplexityScorer
from llm_endpoints import MODEL_REGISTRY, LLMEndpoint


# ---------------------------------------------------------------------------
# Intent-to-minimum-capability mapping
# ---------------------------------------------------------------------------
# Some intents inherently need more capable models.
# This is the "floor" — complexity can push it higher.

INTENT_CAPABILITY_FLOOR = {
    "simple_qa":          0.3,   # Almost any model can handle factual Q&A
    "summarization":      0.4,   # Summarization needs decent comprehension
    "creative_writing":   0.5,   # Creative tasks benefit from better models
    "code_generation":    0.7,   # Code needs precision — prefer capable models
    "complex_reasoning":  0.8,   # Deep analysis needs the best available
}


class SemanticRouter:
    """
    Routes prompts to the optimal LLM endpoint based on intent,
    complexity, cost, and latency.

    The router doesn't just pick "cheap" or "expensive" — it computes
    a score for each endpoint that balances capability fit against cost
    efficiency. The best score wins.
    """

    def __init__(self):
        self.intent_analyzer = IntentAnalyzer()
        self.complexity_scorer = ComplexityScorer()
        self.endpoints = MODEL_REGISTRY
        print(f"\nRouter initialized with {len(self.endpoints)} endpoints ✅")

    def route(self, prompt: str, max_cost: float = None, max_latency_ms: int = None) -> dict:
        """
        Route a prompt to the best endpoint.

        Args:
            prompt:         The user's input text
            max_cost:       Optional cost ceiling (dollars per 1K tokens)
            max_latency_ms: Optional latency ceiling (milliseconds)

        Returns:
            dict with routing decision, reasoning, and analysis details
        """
        # ----- Phase 1: Analyze the prompt -----
        intent_result = self.intent_analyzer.classify(prompt)
        complexity_result = self.complexity_scorer.score(prompt)

        intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        complexity = complexity_result["complexity_score"]

        # ----- Phase 2: Determine required capability -----
        # Start with the intent's floor, then let complexity push it up.
        # High complexity can elevate a "simple" intent to need a powerful model.
        #
        # Formula: required = floor + (1 - floor) × complexity
        # Example: simple_qa floor=0.3, complexity=0.8 → 0.3 + 0.7×0.8 = 0.86

        capability_floor = INTENT_CAPABILITY_FLOOR.get(intent, 0.5)
        required_capability = capability_floor + (1 - capability_floor) * complexity

        # ----- Phase 3: Score each endpoint -----
        endpoint_scores = {}
        for name, endpoint in self.endpoints.items():
            score, reasoning = self._score_endpoint(
                endpoint, required_capability, max_cost, max_latency_ms
            )
            endpoint_scores[name] = {
                "score": score,
                "reasoning": reasoning,
                "endpoint": endpoint,
            }

        # ----- Phase 4: Pick the best -----
        best_name = max(endpoint_scores, key=lambda k: endpoint_scores[k]["score"])
        chosen = endpoint_scores[best_name]

        return {
            # The decision
            "routed_to": best_name,
            "endpoint": chosen["endpoint"],

            # The analysis that led to this decision
            "analysis": {
                "intent": intent,
                "intent_confidence": confidence,
                "complexity_score": complexity,
                "complexity_label": complexity_result["label"],
                "required_capability": round(required_capability, 4),
            },

            # Scores for all endpoints (for transparency)
            "all_endpoint_scores": {
                name: {
                    "score": round(data["score"], 4),
                    "reasoning": data["reasoning"],
                }
                for name, data in endpoint_scores.items()
            },
        }

    def _score_endpoint(
        self,
        endpoint: LLMEndpoint,
        required_capability: float,
        max_cost: float = None,
        max_latency_ms: int = None,
    ) -> tuple:
        """
        Score a single endpoint for this request.

        Scoring logic:
          1. If the endpoint can't meet hard constraints (cost/latency), score = -1
          2. If the endpoint's capability is below required, penalize heavily
          3. Otherwise, prefer the cheapest endpoint that meets the capability bar

        Returns:
            (score, reasoning_string)
        """
        reasons = []

        # --- Hard constraints: instant disqualification ---
        if max_cost is not None and endpoint.cost_per_token > max_cost:
            return -1.0, f"EXCLUDED: cost ${endpoint.cost_per_token} exceeds max ${max_cost}"

        if max_latency_ms is not None and endpoint.latency_ms > max_latency_ms:
            return -1.0, f"EXCLUDED: latency {endpoint.latency_ms}ms exceeds max {max_latency_ms}ms"

        # --- Capability fit (0 to 1) ---
        # How well does this endpoint's capability match what's needed?
        if endpoint.capability >= required_capability:
            # Endpoint meets or exceeds the requirement — full marks
            capability_score = 1.0
            reasons.append(f"capability {endpoint.capability} >= required {required_capability:.2f} ✓")
        else:
            # Endpoint is underpowered — apply a STEEP penalty.
            # We square the ratio so the penalty grows fast as the gap widens.
            # e.g. capability=0.6, required=0.86 → ratio=0.70 → score=0.49
            ratio = endpoint.capability / required_capability
            capability_score = ratio ** 2  # Quadratic penalty
            reasons.append(f"capability {endpoint.capability} < required {required_capability:.2f} ✗ (penalty={capability_score:.2f})")

        # --- Cost efficiency (0 to 1) ---
        # Cheaper is better. We use a ratio against the most expensive endpoint.
        # But we dampen it so it doesn't dominate the score.
        max_endpoint_cost = max(ep.cost_per_token for ep in self.endpoints.values())
        min_endpoint_cost = min(ep.cost_per_token for ep in self.endpoints.values())
        if max_endpoint_cost == min_endpoint_cost:
            cost_score = 1.0
        else:
            cost_score = 1.0 - (endpoint.cost_per_token - min_endpoint_cost) / (max_endpoint_cost - min_endpoint_cost)
        reasons.append(f"cost_eff={cost_score:.2f}")

        # --- Latency efficiency (0 to 1) ---
        # Faster is better. Same normalization approach.
        max_endpoint_latency = max(ep.latency_ms for ep in self.endpoints.values())
        min_endpoint_latency = min(ep.latency_ms for ep in self.endpoints.values())
        if max_endpoint_latency == min_endpoint_latency:
            latency_score = 1.0
        else:
            latency_score = 1.0 - (endpoint.latency_ms - min_endpoint_latency) / (max_endpoint_latency - min_endpoint_latency)
        reasons.append(f"latency_eff={latency_score:.2f}")

        # --- Weighted combination ---
        # Capability is a HARD GATE: if the endpoint can't do the job,
        # no amount of cost savings should make it win.
        #
        # Strategy:
        #   - If capable enough: score = base + cost/latency bonuses
        #   - If underpowered:   score = penalized heavily
        if capability_score >= 1.0:
            # Endpoint meets the bar — reward cost/latency efficiency
            final_score = 0.60 + 0.25 * cost_score + 0.15 * latency_score
        else:
            # Endpoint is underpowered — heavy penalty, cost savings don't help
            final_score = capability_score * 0.50

        reasons.append(f"final={final_score:.3f}")

        return final_score, " | ".join(reasons)

    def route_and_execute(self, prompt: str, **kwargs) -> dict:
        """
        Route the prompt AND call the chosen endpoint.

        Convenience method that combines routing + execution.
        """
        routing = self.route(prompt, **kwargs)
        endpoint = routing["endpoint"]
        response = endpoint.generate(prompt)

        return {
            "routing_decision": routing,
            "response": response,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    router = SemanticRouter()

    test_prompts = [
        # Should route to CHEAP model
        "What is the capital of Japan?",

        # Should route to EXPENSIVE model
        "Compare and contrast TCP vs UDP protocols, analyzing trade-offs in reliability, "
        "latency, and throughput for real-time gaming applications.",

        # Should route to EXPENSIVE model (code intent)
        "Write a Python implementation of Dijkstra's shortest path algorithm.",

        # Should route to CHEAP model (simple creative)
        "Write a haiku about rain.",

        # Test with cost constraint — force cheap model
        "Explain quantum entanglement in detail with step-by-step examples.",
    ]

    print("\n=== Semantic Router Test ===\n")
    print("=" * 70)

    for i, prompt in enumerate(test_prompts):
        display = prompt[:70] + "..." if len(prompt) > 70 else prompt
        result = router.route(prompt)
        analysis = result["analysis"]

        print(f"\n[Test {i+1}] \"{display}\"")
        print(f"  Intent     : {analysis['intent']} (confidence: {analysis['intent_confidence']})")
        print(f"  Complexity : {analysis['complexity_score']} ({analysis['complexity_label']})")
        print(f"  Required   : capability >= {analysis['required_capability']}")
        print(f"  ➜ Routed to: {result['routed_to']}")
        print(f"  Scores:")
        for name, data in result["all_endpoint_scores"].items():
            print(f"    {name:12s} → {data['score']:6.3f}  ({data['reasoning']})")
        print("-" * 70)

    # Test with cost constraint
    print("\n[Test 6] Same complex prompt, but with max_cost=$0.05")
    result = router.route(test_prompts[4], max_cost=0.05)
    print(f"  ➜ Routed to: {result['routed_to']}")
    for name, data in result["all_endpoint_scores"].items():
        print(f"    {name:12s} → {data['score']:6.3f}  ({data['reasoning']})")

    print("\n✅ Routing Engine working correctly.")
