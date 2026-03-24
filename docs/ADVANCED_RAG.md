# 进阶 RAG：混合检索、重排序与 RAGAS 评估

## 1. 混合检索（Dense + BM25）

- **Dense**：沿用现有 Qdrant 向量检索（embedding 相似度）。
- **Sparse**：使用 BM25 关键词检索，索引在 **vectorizer** 建库时一并生成并落盘。
- **融合**：对 Dense 与 BM25 的 top-k 结果做 **RRF（Reciprocal Rank Fusion）** 合并，再进入重排序。

### 配置

- 环境变量 `BM25_INDEX_DIR`：BM25 索引目录，默认 `./customer_support_chat/data/bm25`。  
- Vectorizer 在每次为某 collection 建完 Qdrant 索引后，会为该 collection 生成同名 `{collection_name}.pkl` 的 BM25 索引文件。

### 使用

- FAQ 查询已切换为混合检索：`search_faq` → `hybrid_search_with_rerank("faq_collection", ...)`。  
- 其他 collection（flights/hotels/cars/excursions）若已运行 vectorizer 并生成了对应 BM25 索引，也可通过 `hybrid_search_with_rerank(collection_name, query, ...)` 使用同一套流程。

---

## 2. Cross-Encoder 重排序

- 对混合检索得到的候选片段（如 top 20）使用 **Cross-Encoder** 做 query–document 相关性打分，再取 top-k（默认 5）作为最终上下文。
- 默认模型：`BAAI/bge-reranker-base`（通过 `sentence-transformers` 加载）。  
- 环境变量 `RERANKER_MODEL` 可覆盖模型名。

### 依赖

- `sentence-transformers`（已加入 `pyproject.toml`）。首次运行会自动下载模型。

---

## 3. RAGAS 自动化评估

- 脚本：`evaluation/rag_eval_ragas.py`。  
- 流程：对预设问题做 **混合检索 + 重排序** 得到 contexts → 用当前 LLM 生成 answer → 调用 RAGAS 计算 **faithfulness**、**answer_relevancy**、**context_precision**。
- 使用方式与自定义评估集见：`evaluation/README.md`。

---

## 4. 代码结构摘要


| 模块                                                   | 说明                                                                                                 |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `vectorizer/app/vectordb/vectordb.py`                | 建库时构建并保存 BM25 索引（`_build_and_save_bm25_index`）；FAQ 与常规表均支持。                                        |
| `customer_support_chat/app/services/retrieval/`      | 检索增强：`bm25_store` 加载 BM25、`reranker` 封装 Cross-Encoder、`hybrid_retriever` 实现 hybrid + RRF + rerank。 |
| `customer_support_chat/app/services/tools/lookup.py` | `search_faq` 改为调用 `hybrid_search_with_rerank`，返回格式保持不变。                                            |
| `evaluation/rag_eval_ragas.py`                       | RAGAS 评估入口与样例数据。                                                                                   |


