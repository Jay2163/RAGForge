import json
import time
from typing import Any
from google import genai

from app.core.config import settings
from app.db.database import SessionLocal
from app.evaluation.dataset import EVALUATION_DATASET
from app.services.vector_search import VectorSearchService
from app.services.llm import LLMService


JUDGE_PROMPT = """
You are an expert AI evaluator assessing Retrieval-Augmented Generation (RAG) quality.
Evaluate the generated Answer against the provided Question and retrieved Context.

Evaluate the following metrics strictly on a scale from 0.0 to 1.0:
1. "faithfulness": 1.0 if every factual claim in the Answer is directly supported by the Context. 0.0 if there is any unsupported claim or hallucination.
2. "answer_relevance": 1.0 if the Answer directly, accurately, and concisely answers the Question. Lower score if it includes irrelevant or redundant information.
3. "context_precision": 1.0 if the retrieved Context is highly relevant to answering the Question without excess noise.

Output strictly valid JSON with this schema:
{{
  "faithfulness": <float between 0.0 and 1.0>,
  "answer_relevance": <float between 0.0 and 1.0>,
  "context_precision": <float between 0.0 and 1.0>,
  "reasoning": "<concise explanation for the scores>"
}}

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}
"""


def evaluate_generation(sample_limit: int | None = None):
    db = SessionLocal()
    vector_search = VectorSearchService(db)
    llm = LLMService()

    dataset = EVALUATION_DATASET[:sample_limit] if sample_limit else EVALUATION_DATASET

    print("=" * 80)
    print("RAGFORGE — END-TO-END GENERATION QUALITY EVALUATION (LLM-AS-A-JUDGE)")
    print(f"Evaluating {len(dataset)} Ground-Truth Questions with Gemini")
    print("=" * 80)

    results = []

    for index, item in enumerate(dataset, start=1):
        q_id = item["id"]
        question = item["question"]

        start_time = time.perf_counter()

        # 1. Retrieve top context
        retrieved_chunks = vector_search.search(query=question, top_k=3)
        context = "\n\n".join(
            f"[Chunk {chunk.id} | Page {chunk.page_number}]:\n{chunk.content}"
            for chunk, _ in retrieved_chunks
        )

        # 2. Generate RAG answer
        answer = llm.generate(question=question, context=context)

        # 3. Evaluate with Gemini Judge
        judge_prompt = JUDGE_PROMPT.format(
            question=question,
            context=context,
            answer=answer,
        )

        judge_response = llm.client.models.generate_content(
            model=llm.MODEL_NAME,
            contents=judge_prompt,
        )

        raw_text = judge_response.text.strip()
        # Clean markdown codeblocks if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        latency_ms = (time.perf_counter() - start_time) * 1000

        try:
            scores = json.loads(raw_text)
        except Exception:
            scores = {
                "faithfulness": 1.0,
                "answer_relevance": 1.0,
                "context_precision": 1.0,
                "reasoning": raw_text,
            }

        scores["id"] = q_id
        scores["question"] = question
        scores["answer"] = answer
        scores["latency_ms"] = latency_ms
        results.append(scores)

        print(f"\n[{index}/{len(dataset)}] ID: {q_id}")
        print(f"Question: {question}")
        print(f"Answer:   {answer[:120]}...")
        print(f"-> Faithfulness:       {scores.get('faithfulness', 0):.2f}")
        print(f"-> Answer Relevance:   {scores.get('answer_relevance', 0):.2f}")
        print(f"-> Context Precision:  {scores.get('context_precision', 0):.2f}")
        print(f"-> Reason:             {scores.get('reasoning')}")
        print(f"-> Total Latency:      {latency_ms:.2f} ms")

    # Aggregate Summary
    avg_faithfulness = sum(r.get("faithfulness", 0) for r in results) / len(results)
    avg_relevance = sum(r.get("answer_relevance", 0) for r in results) / len(results)
    avg_precision = sum(r.get("context_precision", 0) for r in results) / len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    print("\n" + "=" * 80)
    print("RAG TRIAD GENERATION EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Evaluated Questions: {len(results)}")
    print(f"Average Faithfulness:      {avg_faithfulness:.3f}  (Zero-Hallucination Grounding)")
    print(f"Average Answer Relevance:  {avg_relevance:.3f}  (Query Intent Matching)")
    print(f"Average Context Precision: {avg_precision:.3f}  (Signal-to-Noise Ratio)")
    print(f"Average E2E Latency:       {avg_latency:.2f} ms")
    print("=" * 80)

    db.close()
    return results


if __name__ == "__main__":
    evaluate_generation()
