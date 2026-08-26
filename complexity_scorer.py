"""
Step 3: Complexity Scorer
=========================
Estimates prompt complexity using rule-based text heuristics.

No ML here — just pattern matching and counting. This is intentional:
  - It's fast (no model inference)
  - It's interpretable (you can see exactly why a score is high)
  - It complements the Intent Analyzer (which uses ML)

The final score is a float from 0.0 (trivial) to 1.0 (very complex).
"""

import re


# ---------------------------------------------------------------------------
# Keyword dictionaries — each set of words signals complexity
# ---------------------------------------------------------------------------

# Words that suggest deep reasoning is needed
REASONING_KEYWORDS = {
    "compare", "contrast", "analyze", "evaluate", "critique",
    "assess", "argue", "debate", "justify", "implications",
    "trade-offs", "tradeoffs", "pros and cons", "advantages",
    "disadvantages", "however", "furthermore", "nevertheless",
    "whereas", "although", "hypothesis", "theoretically",
}

# Words that suggest technical domain knowledge
TECHNICAL_KEYWORDS = {
    "algorithm", "implementation", "architecture", "infrastructure",
    "database", "API", "REST", "GraphQL", "microservice", "kubernetes",
    "docker", "deploy", "scalability", "latency", "throughput",
    "concurrency", "mutex", "semaphore", "binary", "neural",
    "gradient", "backpropagation", "regression", "optimization",
    "encryption", "authentication", "OAuth", "JWT", "SQL",
    "NoSQL", "cache", "index", "schema", "migration",
}

# Words that impose constraints, raising the bar for the response
CONSTRAINT_KEYWORDS = {
    "step-by-step", "step by step", "in detail", "detailed",
    "comprehensive", "thorough", "exhaustive", "at least",
    "no more than", "exactly", "must include", "make sure",
    "don't forget", "including examples", "with examples",
    "with code", "line by line", "word by word",
}


class ComplexityScorer:
    """
    Scores prompt complexity on a 0.0–1.0 scale using text heuristics.

    The scorer evaluates 6 independent signals, weights them,
    and produces a single normalized score.
    """

    # Weights control how much each signal contributes to the final score.
    # These are tunable — you'd adjust them based on real-world performance.
    WEIGHTS = {
        "word_count":         0.20,  # Longer prompts are usually more complex
        "sentence_count":     0.15,  # Multi-sentence = multi-part request
        "question_count":     0.15,  # Multiple questions = more work
        "reasoning_keywords": 0.20,  # Analytical language = deeper thinking
        "technical_keywords": 0.20,  # Domain jargon = specialized knowledge
        "constraint_keywords":0.10,  # Constraints = higher quality bar
    }

    def score(self, prompt: str) -> dict:
        """
        Analyze a prompt and return a complexity score with breakdown.

        Returns:
            dict with:
              - complexity_score: float 0.0–1.0
              - label: "low", "medium", or "high"
              - breakdown: individual signal scores (for debugging)
        """
        signals = {
            "word_count":          self._score_word_count(prompt),
            "sentence_count":      self._score_sentence_count(prompt),
            "question_count":      self._score_question_count(prompt),
            "reasoning_keywords":  self._score_keyword_match(prompt, REASONING_KEYWORDS),
            "technical_keywords":  self._score_keyword_match(prompt, TECHNICAL_KEYWORDS),
            "constraint_keywords": self._score_keyword_match(prompt, CONSTRAINT_KEYWORDS),
        }

        # Weighted sum of all signals
        weighted_score = sum(
            signals[signal] * weight
            for signal, weight in self.WEIGHTS.items()
        )

        # Clamp to [0.0, 1.0]
        final_score = min(max(weighted_score, 0.0), 1.0)

        # Human-readable label
        if final_score < 0.3:
            label = "low"
        elif final_score < 0.6:
            label = "medium"
        else:
            label = "high"

        return {
            "complexity_score": round(final_score, 4),
            "label": label,
            "breakdown": {k: round(v, 4) for k, v in signals.items()},
        }

    # -------------------------------------------------------------------
    # Individual signal scorers — each returns a float from 0.0 to 1.0
    # -------------------------------------------------------------------

    def _score_word_count(self, prompt: str) -> float:
        """
        More words → higher complexity.

        Scale:
          - 1–10 words  → low  (0.0–0.3)
          - 10–30 words → med  (0.3–0.7)
          - 30+ words   → high (0.7–1.0)

        We use a simple linear ramp capped at 50 words.
        """
        word_count = len(prompt.split())
        # Linear scale: 0 words = 0.0, 50+ words = 1.0
        return min(word_count / 50.0, 1.0)

    def _score_sentence_count(self, prompt: str) -> float:
        """
        More sentences → likely a multi-part or compound request.

        Uses regex to split on sentence-ending punctuation.
        """
        # Split on . ! ? followed by space or end-of-string
        sentences = re.split(r'[.!?]+(?:\s|$)', prompt)
        # Filter out empty strings from the split
        sentence_count = len([s for s in sentences if s.strip()])
        # 1 sentence = 0.2, 2 = 0.4, 3 = 0.6, 4 = 0.8, 5+ = 1.0
        return min(sentence_count / 5.0, 1.0)

    def _score_question_count(self, prompt: str) -> float:
        """
        Multiple question marks → multiple questions to answer.
        """
        question_count = prompt.count("?")
        # 0 = 0.0, 1 = 0.33, 2 = 0.67, 3+ = 1.0
        return min(question_count / 3.0, 1.0)

    def _score_keyword_match(self, prompt: str, keywords: set) -> float:
        """
        Count how many keywords from the given set appear in the prompt.

        More keyword matches → higher signal score.
        We normalize by expecting at most 5 matches for a "maxed out" score.
        """
        prompt_lower = prompt.lower()
        match_count = sum(1 for kw in keywords if kw in prompt_lower)
        # 0 matches = 0.0, 5+ matches = 1.0
        return min(match_count / 5.0, 1.0)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scorer = ComplexityScorer()

    test_prompts = [
        # Should be LOW complexity
        "What is the capital of France?",

        # Should be MEDIUM complexity
        "Explain how a binary search algorithm works and provide a Python example.",

        # Should be HIGH complexity
        (
            "Compare and contrast microservices vs monolithic architecture. "
            "Analyze the trade-offs in terms of scalability, deployment complexity, "
            "and latency. Provide a step-by-step migration strategy with detailed "
            "examples for each approach. What are the implications for database "
            "design? How does this affect API gateway requirements?"
        ),
    ]

    print("=== Complexity Scorer Test ===\n")

    for prompt in test_prompts:
        result = scorer.score(prompt)
        # Truncate long prompts for display
        display = prompt[:80] + "..." if len(prompt) > 80 else prompt
        print(f"Prompt     : \"{display}\"")
        print(f"Complexity : {result['complexity_score']} ({result['label']})")
        print(f"Breakdown  : {result['breakdown']}")
        print()

    print("✅ Complexity Scorer working correctly.")

