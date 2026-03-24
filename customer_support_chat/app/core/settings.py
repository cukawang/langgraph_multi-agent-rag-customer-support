from os import environ
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY: str = environ.get("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = environ.get("OPENAI_BASE_URL", "")
    
    # Model configuration for cost optimization
    OPENAI_MODEL: str = environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
    MAX_TOKENS: int = int(environ.get("MAX_TOKENS", "1000"))  # Limit tokens to control costs
    
    DATA_PATH: str = "./customer_support_chat/data"
    LOG_LEVEL: str = environ.get("LOG_LEVEL", "DEBUG")
    SQLITE_DB_PATH: str = environ.get(
        "SQLITE_DB_PATH", "./customer_support_chat/data/travel2.sqlite"
    )
    QDRANT_URL: str = environ.get("QDRANT_URL", "http://localhost:6333")
    QDRANT_KEY: str = environ.get("QDRANT_KEY", "")
    RECREATE_COLLECTIONS: bool = environ.get("RECREATE_COLLECTIONS", "False")
    LIMIT_ROWS: int = environ.get("LIMIT_ROWS", "100")
    
    # Amap (高德地图) MCP service via DashScope
    DASHSCOPE_API_KEY: str = environ.get("DASHSCOPE_API_KEY", "")
    # BM25 index dir for hybrid retrieval (must match vectorizer output)
    BM25_INDEX_DIR: str = environ.get("BM25_INDEX_DIR", "./customer_support_chat/data/bm25")
    # Reranker model (sentence-transformers CrossEncoder)
    RERANKER_MODEL: str = environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")

def get_settings():
    return Config()