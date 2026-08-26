"""
Step 1: Mock LLM Endpoints
==========================
Simulated LLM backends that mimic real-world model tiers.
Each endpoint has different cost, latency, and capability profiles.
"""

import time
import random


class LLMEndpoint:
    """
    Base class representing a single LLM endpoint.

    Every endpoint has:
      - name:          human-readable identifier
      - cost_per_token: simulated cost (in dollars per 1K tokens)
      - latency_ms:     simulated response time (in milliseconds)
      - capability:     a simple quality score from 0.0 to 1.0
    """

    def __init__(self, name: str, cost_per_token: float, latency_ms: int, capability: float):
        self.name = name
        self.cost_per_token = cost_per_token
        self.latency_ms = latency_ms
        self.capability = capability

    def generate(self, prompt: str) -> dict:
        """
        Simulate an LLM call.

        Instead of hitting a real API, we:
          1. Sleep to mimic network + inference latency
          2. Return a mock response with metadata

        Returns a dict with the response text, cost, and latency.
        """
        # Simulate realistic latency with ±20% jitter
        jitter = random.uniform(0.8, 1.2)
        actual_latency_ms = int(self.latency_ms * jitter)
        time.sleep(actual_latency_ms / 1000)  # convert ms → seconds

        # Estimate token count (rough: ~1 token per 4 chars)
        estimated_tokens = len(prompt) / 4
        cost = (estimated_tokens / 1000) * self.cost_per_token

        return {
            "model": self.name,
            "response": f"[{self.name}] Simulated response to: '{prompt[:50]}...'",
            "latency_ms": actual_latency_ms,
            "estimated_cost": round(cost, 6),
            "capability": self.capability,
        }

    def __repr__(self):
        return (
            f"LLMEndpoint(name='{self.name}', "
            f"cost=${self.cost_per_token}/1K tokens, "
            f"latency={self.latency_ms}ms, "
            f"capability={self.capability})"
        )


# ---------------------------------------------------------------------------
# Pre-configured endpoint instances
# ---------------------------------------------------------------------------
# These mirror real-world model tiers you'd see in production.

cheap_model = LLMEndpoint(
    name="fast-lite",
    cost_per_token=0.01,    # Very cheap — like GPT-4o-mini
    latency_ms=150,         # Fast response
    capability=0.6,         # Good for simple tasks
)

expensive_model = LLMEndpoint(
    name="power-pro",
    cost_per_token=0.10,    # 10x more expensive — like GPT-4o / Claude Opus
    latency_ms=800,         # Slower but more capable
    capability=0.95,        # Excellent for complex reasoning
)

# A registry so the router can discover all available endpoints
MODEL_REGISTRY = {
    "fast-lite": cheap_model,
    "power-pro": expensive_model,
}


# ---------------------------------------------------------------------------
# Quick smoke test — run this file directly to verify endpoints work
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Mock LLM Endpoint Test ===\n")

    test_prompt = "Explain the theory of relativity in simple terms."

    for name, endpoint in MODEL_REGISTRY.items():
        print(f"Model: {endpoint}")
        result = endpoint.generate(test_prompt)
        print(f"  Response : {result['response']}")
        print(f"  Latency  : {result['latency_ms']}ms")
        print(f"  Cost     : ${result['estimated_cost']}")
        print()

    print("✅ All endpoints responding correctly.")

