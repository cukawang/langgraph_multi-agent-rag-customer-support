"""
RAGAS-based automated evaluation for the RAG pipeline (hybrid retrieval + rerank + LLM).
Run from project root: poetry run python evaluation/rag_eval_ragas.py
Requires: OPENAI_API_KEY, and that vectorizer has been run (Qdrant + BM25 index for faq_collection).
"""
import os
import sys

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Sample evaluation set: question -> optional reference answer (for context recall if used)
EVAL_SAMPLES = [
    {
        "question": "Can I change my flight to a different date?",
        "reference": "Flight changes depend on fare conditions. You may change your flight subject to availability and fare rules.",
    },
    {
        "question": "What is the baggage allowance for economy class?",
        "reference": "Baggage allowance varies by fare and route. Check your ticket or our baggage policy.",
    },
    {
        "question": "How do I get a refund for a cancelled booking?",
        "reference": "Refunds are processed according to the fare conditions and cancellation policy.",
    },
    {
        "question": "Is food provided on board?",
        "reference": "Meal service depends on flight duration and route. See our inflight service information.",
    },
    {
        "question": "What documents do I need for check-in?",
        "reference": "You need a valid ID or passport and your booking reference for check-in.",
    },
]


def retrieve_contexts(query: str, top_k: int = 5) -> list[str]:
    """Use hybrid retrieval + rerank to get context strings for the query."""
    from customer_support_chat.app.services.retrieval import hybrid_search_with_rerank
    results = hybrid_search_with_rerank(
        "faq_collection",
        query,
        top_k_dense=20,
        top_k_bm25=20,
        top_k_final=top_k,
        use_rerank=True,
    )
    return [r.get("content", "") for r in results if r.get("content")]


def generate_answer(question: str, contexts: list[str]) -> str:
    """Generate an answer from retrieved contexts using LLM."""
    from langchain_openai import ChatOpenAI
    from customer_support_chat.app.core.settings import get_settings
    settings = get_settings()
    llm = ChatOpenAI(
        model=getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"),
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=getattr(settings, "OPENAI_BASE_URL", None) or None,
        temperature=0,
    )
    context_block = "\n\n---\n\n".join(contexts) if contexts else "No relevant context found."
    prompt = f"""Answer the question based only on the following context. If the context does not contain enough information, say so briefly. Do not invent facts.

Context:
{context_block}

Question: {question}

Answer:"""
    msg = llm.invoke(prompt)
    return msg.content if hasattr(msg, "content") else str(msg)


def run_ragas_evaluation():
    """Build dataset from eval samples (retrieve + generate), then run RAGAS metrics."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
    except ImportError as e:
        print("Missing dependency. Install with: poetry add ragas datasets")
        raise e

    try:
        from ragas.dataset_schema import EvaluationDataset
    except ImportError:
        from ragas import EvaluationDataset

    print("Building evaluation dataset (retrieve + generate)...")
    rows = []
    for s in EVAL_SAMPLES:
        q = s["question"]
        contexts = retrieve_contexts(q, top_k=5)
        answer = generate_answer(q, contexts)
        rows.append({
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": s.get("reference", ""),
        })

    # Ragas 0.2 expects EvaluationDataset (from_list of SingleTurnSample-like dicts)
    try:
        eval_dataset = EvaluationDataset.from_list(rows)
    except Exception:
        # Fallback: use HuggingFace Dataset if column names match
        from datasets import Dataset
        eval_dataset = Dataset.from_list(rows)
    print(f"Dataset size: {len(rows)}")

    metrics = [faithfulness, answer_relevancy, context_precision]
    result = evaluate(dataset=eval_dataset, metrics=metrics)
    print("\n--- RAGAS Evaluation Results ---")
    print(result)
    return result


if __name__ == "__main__":
    run_ragas_evaluation()
