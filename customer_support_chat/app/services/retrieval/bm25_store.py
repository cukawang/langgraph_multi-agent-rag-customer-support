"""Load BM25 index built by vectorizer (pickle)."""
import os
import re
import pickle
from typing import List, Tuple, Optional
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.core.logger import logger

settings = get_settings()
_CACHE = {}


def _tokenize(text: str) -> List[str]:
    """Must match vectorizer.app.vectordb.vectordb.VectorDB._tokenize_for_bm25."""
    if not text or not isinstance(text, str):
        return []
    return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower())


def load_bm25_index(collection_name: str) -> Optional[dict]:
    """Load BM25 index for collection from BM25_INDEX_DIR. Cached per collection."""
    global _CACHE
    if collection_name in _CACHE:
        return _CACHE[collection_name]
    index_dir = getattr(settings, "BM25_INDEX_DIR", "./customer_support_chat/data/bm25")
    path = os.path.join(index_dir, f"{collection_name}.pkl")
    if not os.path.isfile(path):
        logger.warning(f"BM25 index not found: {path}")
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        _CACHE[collection_name] = data
        return data
    except Exception as e:
        logger.error(f"Failed to load BM25 index {path}: {e}")
        return None


def bm25_search(
    collection_name: str,
    query: str,
    top_k: int = 20,
) -> List[Tuple[str, str, float]]:
    """
    Returns list of (point_id, content, score) for top_k BM25 results.
    If index missing, returns [].
    """
    data = load_bm25_index(collection_name)
    if not data:
        return []
    doc_ids = data["doc_ids"]
    contents = data["contents"]
    bm25 = data["bm25"]
    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []
    scores = bm25.get_scores(tokenized_query)
    # top-k indices by score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(doc_ids[i], contents[i], float(scores[i])) for i in top_indices if scores[i] > 0]
