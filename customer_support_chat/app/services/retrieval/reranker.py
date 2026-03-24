"""Reranker for retrieved chunks using qwen3-rerank via DashScope SDK."""
from typing import List, Tuple
from http import HTTPStatus

import dashscope

from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.core.logger import logger

settings = get_settings()


def _ensure_dashscope_api_key() -> bool:
    """Ensure dashscope.api_key is set (prefer DASHSCOPE_API_KEY, fallback to OPENAI_API_KEY)."""
    if dashscope.api_key:
        return True
    # 优先使用官方环境变量，其次复用 OPENAI_API_KEY
    api_key = (
        getattr(settings, "DASHSCOPE_API_KEY", None)
        or getattr(settings, "OPENAI_API_KEY", None)
    )
    if not api_key:
        logger.warning("DASHSCOPE_API_KEY / OPENAI_API_KEY 未配置，qwen3-rerank 将跳过。")
        return False
    dashscope.api_key = api_key
    return True


def rerank(
    query: str,
    pairs: List[Tuple[str, str]],
    top_k: int = 5,
) -> List[Tuple[str, str, float]]:
    """
    pairs: list of (point_id, content).
    Returns list of (point_id, content, score) top_k by relevance, scored by qwen3-rerank。
    """
    if not pairs:
        return []

    if not _ensure_dashscope_api_key():
        # 无法获取 key，直接按原顺序返回
        return [(pid, c, 0.0) for pid, c in pairs[:top_k]]

    documents = [c for _, c in pairs]
    try:
        resp = dashscope.TextReRank.call(
            model="qwen3-rerank",
            query=query,
            documents=documents,
            top_n=min(top_k, len(documents)),
            return_documents=False,
            # 可选指令：搜索/FAQ 场景更贴近你的用途
            instruct=(
                "Given a user query, score each document by how well it answers the query. "
                "Return a higher score for more relevant documents."
            ),
        )
    except Exception as e:
        logger.warning(f"调用 dashscope TextReRank 失败，将跳过重排: {e}")
        return [(pid, c, 0.0) for pid, c in pairs[:top_k]]

    if resp.status_code != HTTPStatus.OK:
        logger.warning(f"dashscope TextReRank 返回非 200（{resp.status_code}），将按原顺序返回。")
        return [(pid, c, 0.0) for pid, c in pairs[:top_k]]

    # 参考 dashscope 返回格式，从 items 中取 index + score
    results = getattr(resp, "output", None)
    items = getattr(results, "results", None) if results is not None else None
    if not items:
        logger.warning("dashscope TextReRank 返回结果为空，将按原顺序返回。")
        return [(pid, c, 0.0) for pid, c in pairs[:top_k]]

    score_by_idx = {}
    for item in items:
        idx = getattr(item, "index", None) or item.get("index")
        score = getattr(item, "relevance_score", None) or item.get("relevance_score") or item.get("score", 0.0)
        if isinstance(idx, int):
            score_by_idx[idx] = float(score)

    indexed: List[Tuple[str, str, float]] = []
    for i, (pid, content) in enumerate(pairs):
        sc = score_by_idx.get(i, 0.0)
        indexed.append((pid, content, sc))

    indexed.sort(key=lambda x: x[2], reverse=True)
    return indexed[:top_k]
