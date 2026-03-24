"""Hybrid retrieval: dense (Qdrant) + sparse (BM25) with RRF merge and Cross-Encoder rerank."""
from typing import List, Dict, Any, Optional

from customer_support_chat.app.core.logger import logger
from .bm25_store import bm25_search
from .reranker import rerank

# Map collection_name -> table_name for vectorizer VectorDB
COLLECTION_TO_TABLE = {
    "faq_collection": "faq",
    "flights_collection": "flights",
    "hotels_collection": "hotels",
    "car_rentals_collection": "car_rentals",
    "excursions_collection": "trip_recommendations",
}


def _rrf_merge(
    dense_items: List[tuple],
    sparse_items: List[tuple],
    k: int = 60,
) -> List[tuple]:
    """
    Reciprocal Rank Fusion. Each item is (point_id, content, score).
    Returns merged list of (point_id, content, rrf_score) sorted by rrf_score desc.
    """
    def rrf_score(rank: int) -> float:
        return 1.0 / (k + rank)

    scores: Dict[str, float] = {}
    content_by_id: Dict[str, str] = {}

    for rank, (pid, content, _) in enumerate(dense_items):
        key = str(pid)
        scores[key] = scores.get(key, 0.0) + rrf_score(rank)
        content_by_id[key] = content

    for rank, (pid, content, _) in enumerate(sparse_items):
        key = str(pid)
        scores[key] = scores.get(key, 0.0) + rrf_score(rank)
        content_by_id[key] = content

    merged = [(pid, content_by_id[pid], sc) for pid, sc in scores.items() if pid in content_by_id]
    merged.sort(key=lambda x: x[2], reverse=True)
    return merged


def hybrid_search_with_rerank(
    collection_name: str,
    query: str,
    top_k_dense: int = 20,
    top_k_bm25: int = 20,
    top_k_final: int = 5,
    use_rerank: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run hybrid retrieval (dense + BM25), RRF merge, then optional Cross-Encoder rerank.
    Returns list of dicts with keys: id, content, score, (and payload if from Qdrant).
    """
    from vectorizer.app.vectordb.vectordb import VectorDB
    table_name = COLLECTION_TO_TABLE.get(collection_name, collection_name.replace("_collection", ""))
    # Dense search
    vectordb = VectorDB(table_name=table_name, collection_name=collection_name, create_collection=False)
    dense_raw = vectordb.search(query, limit=top_k_dense)
    dense_items = [(r.id, r.payload.get("content", ""), r.score or 0.0) for r in dense_raw]

    # Sparse (BM25) search
    sparse_items = bm25_search(collection_name, query, top_k=top_k_bm25)

    if not dense_items and not sparse_items:
        return []

    merged = _rrf_merge(dense_items, sparse_items)
    if not merged:
        return []

    if use_rerank and len(merged) > top_k_final:
        pairs = [(pid, c) for pid, c, _ in merged]
        reranked = rerank(query, pairs, top_k=top_k_final)
        return [
            {"id": pid, "content": c, "score": sc, "payload": {"content": c}}
            for pid, c, sc in reranked
        ]
    # No rerank or few results: take top_k_final from merged
    return [
        {"id": pid, "content": c, "score": sc, "payload": {"content": c}}
        for pid, c, sc in merged[:top_k_final]
    ]
