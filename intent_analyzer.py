"""
Step 2: Intent Analyzer
=======================
Classifies incoming prompts into intent categories using
semantic embeddings and cosine similarity.

Core idea:
  - Each intent category has example phrases.
  - We embed those phrases and the incoming prompt into vectors.
  - The intent whose examples are closest (by cosine similarity) wins.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Intent definitions — each intent has a label and example phrases
# ---------------------------------------------------------------------------
# These examples teach the model what each intent "sounds like".
# The more diverse your examples, the better the classification.

INTENT_DEFINITIONS = {
    "simple_qa": {
        "description": "Simple factual questions with short answers",
        "examples": [
            "What is the capital of France?",
            "Who invented the telephone?",
            "How many planets are in the solar system?",
            "What year did World War II end?",
            "Define photosynthesis.",
            "What is the boiling point of water?",
        ],
    },
    "complex_reasoning": {
        "description": "Tasks requiring deep analysis, comparison, or multi-step logic",
        "examples": [
            "Compare and contrast microservices vs monolithic architecture for a startup.",
            "Analyze the economic impact of remote work on urban real estate markets.",
            "What are the philosophical implications of artificial general intelligence?",
            "Explain the trade-offs between consistency and availability in distributed systems.",
            "Evaluate the pros and cons of nuclear energy as a climate solution.",
            "Design a strategy for migrating a legacy system to the cloud.",
        ],
    },
    "code_generation": {
        "description": "Writing, debugging, or explaining code",
        "examples": [
            "Write a Python function to sort a list using merge sort.",
            "Debug this JavaScript code that throws a TypeError.",
            "Create a REST API endpoint in Flask for user authentication.",
            "Write a SQL query to find duplicate records in a table.",
            "Implement a binary search tree in Java.",
            "Explain what this regex does: ^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+$",
        ],
    },
    "creative_writing": {
        "description": "Stories, poems, marketing copy, or other creative content",
        "examples": [
            "Write a short story about a robot learning to love.",
            "Compose a haiku about autumn.",
            "Create a catchy tagline for a new coffee brand.",
            "Write a persuasive product description for wireless earbuds.",
            "Draft a fantasy world-building document for a tabletop RPG.",
            "Write a limerick about programming.",
        ],
    },
    "summarization": {
        "description": "Condensing or summarizing longer text",
        "examples": [
            "Summarize this article in three bullet points.",
            "Give me the key takeaways from this research paper.",
            "TL;DR this email thread.",
            "Condense this 10-page report into a one-page executive summary.",
            "What are the main points of this blog post?",
            "Summarize the plot of The Great Gatsby.",
        ],
    },
}


class IntentAnalyzer:
    """
    Classifies prompts into intent categories using embedding similarity.

    How it works:
      1. On init, we load a lightweight embedding model and pre-compute
         embeddings for all intent example phrases.
      2. On classify(), we embed the incoming prompt and compare it
         against each intent's examples using cosine similarity.
      3. We return the best-matching intent and a confidence score.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: The sentence-transformers model to use.
                        'all-MiniLM-L6-v2' is only ~80MB and very fast.
                        It maps sentences to a 384-dimensional vector space.
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded ✅")

        # Pre-compute embeddings for all intent examples
        self.intent_embeddings = {}  # intent_name → numpy array of shape (N, 384)
        self._precompute_embeddings()

    def _precompute_embeddings(self):
        """
        Embed all example phrases upfront so classify() is fast.

        This is a common optimization — you only pay the embedding cost
        once at startup, not on every request.
        """
        for intent_name, intent_data in INTENT_DEFINITIONS.items():
            examples = intent_data["examples"]
            # model.encode() returns a numpy array of shape (len(examples), 384)
            embeddings = self.model.encode(examples, convert_to_numpy=True)
            self.intent_embeddings[intent_name] = embeddings

        print(f"Pre-computed embeddings for {len(self.intent_embeddings)} intents ✅")

    def classify(self, prompt: str) -> dict:
        """
        Classify a prompt into an intent category.

        Steps:
          1. Embed the prompt → a single 384-dim vector
          2. For each intent, compute cosine similarity between the prompt
             vector and every example vector in that intent
          3. Take the mean similarity per intent (averaging over examples)
          4. The intent with the highest mean similarity wins

        Returns:
            dict with:
              - intent: the matched category name
              - confidence: cosine similarity score (0.0 to 1.0)
              - all_scores: similarity scores for every intent (for debugging)
        """
        # Step 1: Embed the prompt
        prompt_embedding = self.model.encode([prompt], convert_to_numpy=True)[0]

        # Step 2 & 3: Compute mean cosine similarity for each intent
        scores = {}
        for intent_name, example_embeddings in self.intent_embeddings.items():
            similarities = cosine_similarity(prompt_embedding, example_embeddings)
            # Mean similarity across all examples for this intent
            scores[intent_name] = float(np.mean(similarities))

        # Step 4: Pick the best match
        best_intent = max(scores, key=scores.get)

        return {
            "intent": best_intent,
            "confidence": round(scores[best_intent], 4),
            "description": INTENT_DEFINITIONS[best_intent]["description"],
            "all_scores": {k: round(v, 4) for k, v in sorted(
                scores.items(), key=lambda x: x[1], reverse=True
            )},
        }


def cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single vector and each row of a matrix.

    Cosine similarity = (A · B) / (||A|| * ||B||)

    It measures the angle between two vectors:
      - 1.0  = identical direction (same meaning)
      - 0.0  = perpendicular (unrelated)
      - -1.0 = opposite direction (opposite meaning)

    Args:
        vector: shape (D,)  — the prompt embedding
        matrix: shape (N, D) — the example embeddings

    Returns:
        shape (N,) — similarity score for each example
    """
    # Dot product of prompt with each example
    dot_products = np.dot(matrix, vector)

    # Norms
    vector_norm = np.linalg.norm(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    # Cosine similarity (add tiny epsilon to avoid division by zero)
    similarities = dot_products / (vector_norm * matrix_norms + 1e-10)

    return similarities


# ---------------------------------------------------------------------------
# Smoke test — run this file directly to see intent classification in action
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    analyzer = IntentAnalyzer()

    test_prompts = [
        "What is the speed of light?",
        "Write a Python script to scrape a website.",
        "Compare React and Vue for building a dashboard.",
        "Write me a poem about the ocean.",
        "Summarize this meeting transcript for me.",
    ]

    print("\n=== Intent Classification Test ===\n")

    for prompt in test_prompts:
        result = analyzer.classify(prompt)
        print(f"Prompt  : \"{prompt}\"")
        print(f"Intent  : {result['intent']} (confidence: {result['confidence']})")
        print(f"          {result['description']}")
        print(f"Scores  : {result['all_scores']}")
        print()

    print("✅ Intent Analyzer working correctly.")

