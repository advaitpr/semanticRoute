"""
Step 5: Testing & Evaluation
=============================
End-to-end test suite for the Semantic Router.

Runs diverse prompts through the full pipeline and validates:
  - Intent classification accuracy
  - Complexity scoring reasonableness
  - Routing decision correctness
  - Cost and latency metrics
"""

from router import SemanticRouter


# ---------------------------------------------------------------------------
# Test cases: each has a prompt, expected intent, and expected route
# ---------------------------------------------------------------------------
TEST_CASES = [
    # --- Simple tasks → should route to fast-lite ---
    {
        "prompt": "What is the capital of Japan?",
        "expected_intent": "simple_qa",
        "expected_route": "fast-lite",
        "category": "Simple Q&A",
    },
    {
        "prompt": "Who wrote Romeo and Juliet?",
        "expected_intent": "simple_qa",
        "expected_route": "fast-lite",
        "category": "Simple Q&A",
    },
    {
        "prompt": "What is the boiling point of ethanol?",
        "expected_intent": "simple_qa",
        "expected_route": "fast-lite",
        "category": "Simple Q&A",
    },
    {
        "prompt": "Summarize the plot of 1984 by George Orwell.",
        "expected_intent": "summarization",
        "expected_route": "fast-lite",
        "category": "Summarization",
    },
    {
        "prompt": "Write a haiku about winter.",
        "expected_intent": "creative_writing",
        "expected_route": "fast-lite",
        "category": "Simple Creative",
    },

    # --- Complex tasks → should route to power-pro ---
    {
        "prompt": (
            "Compare and contrast microservices vs monolithic architecture. "
            "Analyze the trade-offs in scalability, deployment, and latency."
        ),
        "expected_intent": "complex_reasoning",
        "expected_route": "power-pro",
        "category": "Complex Reasoning",
    },
    {
        "prompt": (
            "Evaluate the pros and cons of using Kubernetes for a small startup. "
            "Consider infrastructure cost, operational complexity, and scalability. "
            "What are the alternatives?"
        ),
        "expected_intent": "complex_reasoning",
        "expected_route": "power-pro",
        "category": "Complex Reasoning",
    },
    {
        "prompt": "Write a Python implementation of the A* pathfinding algorithm with visualization.",
        "expected_intent": "code_generation",
        "expected_route": "power-pro",
        "category": "Code Generation",
    },
    {
        "prompt": (
            "Implement a REST API in Flask with JWT authentication, rate limiting, "
            "and database connection pooling. Include error handling and logging."
        ),
        "expected_intent": "code_generation",
        "expected_route": "power-pro",
        "category": "Code Generation",
    },
    {
        "prompt": (
            "Write a detailed fantasy short story about a dragon who becomes a librarian. "
            "Include dialogue, world-building, and a plot twist. At least 1000 words."
        ),
        "expected_intent": "creative_writing",
        "expected_route": "power-pro",
        "category": "Complex Creative",
    },

    # --- Edge cases ---
    {
        "prompt": "Fix this: def add(a, b) return a + b",
        "expected_intent": "code_generation",
        "expected_route": "fast-lite",  # Simple fix, low complexity
        "category": "Edge: Simple Code Fix",
    },
    {
        "prompt": "TL;DR this for me.",
        "expected_intent": "summarization",
        "expected_route": "fast-lite",
        "category": "Edge: Minimal Prompt",
    },
]


def run_test_suite():
    """Run all test cases and report results."""
    print("=" * 70)
    print("SEMANTIC ROUTER — FULL TEST SUITE")
    print("=" * 70)

    router = SemanticRouter()

    # Tracking metrics
    intent_correct = 0
    route_correct = 0
    total = len(TEST_CASES)
    total_cost = 0.0
    results = []

    print(f"\nRunning {total} test cases...\n")
    print("-" * 70)

    for i, test in enumerate(TEST_CASES, 1):
        prompt = test["prompt"]
        result = router.route(prompt)
        analysis = result["analysis"]

        # Check correctness
        intent_match = analysis["intent"] == test["expected_intent"]
        route_match = result["routed_to"] == test["expected_route"]

        if intent_match:
            intent_correct += 1
        if route_match:
            route_correct += 1

        # Display
        display_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt
        intent_icon = "✅" if intent_match else "❌"
        route_icon = "✅" if route_match else "❌"

        print(f"[{i:2d}] {test['category']}")
        print(f"     \"{display_prompt}\"")
        print(f"     Intent : {intent_icon} {analysis['intent']:<20s} (expected: {test['expected_intent']})")
        print(f"     Route  : {route_icon} {result['routed_to']:<20s} (expected: {test['expected_route']})")
        print(f"     Detail : complexity={analysis['complexity_score']}, required_cap={analysis['required_capability']}")
        if not intent_match or not route_match:
            print(f"     Scores : {result['all_endpoint_scores']}")
        print("-" * 70)

        results.append({
            "test": test,
            "result": result,
            "intent_correct": intent_match,
            "route_correct": route_match,
        })

    # --- Summary Report ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  Intent Classification Accuracy : {intent_correct}/{total} ({100*intent_correct/total:.0f}%)")
    print(f"  Routing Decision Accuracy      : {route_correct}/{total} ({100*route_correct/total:.0f}%)")

    # Show which tests failed
    intent_failures = [r for r in results if not r["intent_correct"]]
    route_failures = [r for r in results if not r["route_correct"]]

    if intent_failures:
        print(f"\n  ❌ Intent misclassifications:")
        for r in intent_failures:
            prompt = r["test"]["prompt"][:50] + "..."
            print(f"     \"{prompt}\"")
            print(f"       Got: {r['result']['analysis']['intent']}, Expected: {r['test']['expected_intent']}")

    if route_failures:
        print(f"\n  ❌ Routing misroutes:")
        for r in route_failures:
            prompt = r["test"]["prompt"][:50] + "..."
            print(f"     \"{prompt}\"")
            print(f"       Got: {r['result']['routed_to']}, Expected: {r['test']['expected_route']}")

    if not intent_failures and not route_failures:
        print("\n  🎉 All tests passed!")

    # --- Cost Analysis ---
    print("\n" + "=" * 70)
    print("COST ANALYSIS (if all prompts were executed)")
    print("=" * 70)

    fast_lite_count = sum(1 for r in results if r["result"]["routed_to"] == "fast-lite")
    power_pro_count = sum(1 for r in results if r["result"]["routed_to"] == "power-pro")

    print(f"\n  Routed to fast-lite : {fast_lite_count} prompts (cheap)")
    print(f"  Routed to power-pro: {power_pro_count} prompts (expensive)")
    print(f"  Cost savings        : Routing saved {fast_lite_count} calls from the expensive model")

    all_expensive_cost = total * 0.10  # If everything went to power-pro
    routed_cost = fast_lite_count * 0.01 + power_pro_count * 0.10
    savings_pct = (1 - routed_cost / all_expensive_cost) * 100

    print(f"\n  All-expensive cost  : ${all_expensive_cost:.2f}/1K tokens")
    print(f"  Smart-routed cost   : ${routed_cost:.2f}/1K tokens")
    print(f"  Savings             : {savings_pct:.0f}%")

    print("\n" + "=" * 70)
    print("✅ Test suite complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_test_suite()

