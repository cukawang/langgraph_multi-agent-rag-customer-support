# RAG Evaluation (RAGAS)

Automated evaluation of the RAG pipeline using [RAGAS](https://github.com/explodinggradients/ragas).

## Prerequisites

- Vectorizer has been run at least once (Qdrant collections + BM25 indices under `customer_support_chat/data/bm25/`).
- `OPENAI_API_KEY` (and optional `OPENAI_BASE_URL`) set in `.env`.

## Install

From project root:

```bash
poetry install
```

## Run

```bash
poetry run python evaluation/rag_eval_ragas.py
```

The script will:

1. For each sample question: run **hybrid retrieval** (dense + BM25) + **Cross-Encoder rerank** to get contexts.
2. Generate an answer from those contexts using the configured LLM.
3. Run RAGAS metrics: **faithfulness**, **answer_relevancy**, **context_precision**.
4. Print the evaluation results.

## Customising the dataset

Edit `EVAL_SAMPLES` in `rag_eval_ragas.py` to add or change questions and optional `reference` answers.