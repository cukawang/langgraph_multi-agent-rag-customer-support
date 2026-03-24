import re
import logging
from typing import List, Dict

from langchain_core.tools import tool

from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.services.retrieval import hybrid_search_with_rerank

logger = logging.getLogger(__name__)
settings = get_settings()


def _parse_faq_entry(content: str, score: float) -> Dict:
    """Parse content into question/answer for FAQ entry."""
    question = "General FAQ Information"
    answer = content
    category = "FAQ"
    question_match = re.search(r"^\d+\. (.+?)(?=\n|$)", content, re.MULTILINE)
    if question_match:
        question = question_match.group(1).strip()
        answer_start = content.find(question) + len(question)
        answer = content[answer_start:].strip()
    elif content.startswith("##"):
        lines = content.split("\n", 1)
        question = lines[0].replace("##", "").strip()
        answer = lines[1] if len(lines) > 1 else "See section content for details."
    return {
        "question": question,
        "answer": answer,
        "category": category,
        "chunk": content,
        "similarity": score,
    }


@tool
def search_faq(
    query: str,
    limit: int = 5,
) -> List[Dict]:
    """Search for FAQ entries using hybrid retrieval (dense + BM25) and Cross-Encoder reranking."""
    results = hybrid_search_with_rerank(
        "faq_collection",
        query,
        top_k_dense=20,
        top_k_bm25=20,
        top_k_final=limit,
        use_rerank=True,
    )
    faq_entries = []
    for r in results:
        content = r.get("content", "")
        score = r.get("score", 0.0)
        faq_entries.append(_parse_faq_entry(content, score))
    return faq_entries

@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted.
    Use this before making any flight changes or performing other 'write' events."""
    faq_results = search_faq.invoke({"query": query, "limit": 2})
    if not faq_results:
        return "Sorry, I couldn't find any relevant policy information. Please contact support for assistance."
    
    policy_info = "\n\n".join([f"Q: {entry['question']}\nA: {entry['answer']}" for entry in faq_results])
    return f"Here's the relevant policy information:\n\n{policy_info}"
