# 🛍️ ProductInsight AI

Hybrid (keyword + semantic + visual) product search and a grounded shopping assistant for an Indian ethnic-wear catalogue, **evaluated with ranking metrics**.

**Stack:** Python · Google Gemini (embeddings + LLM) · FAISS · BM25 · CLIP · Pydantic · FastAPI · Streamlit · pytest · Docker · GitHub Actions

## How it works

```
query ─► parse into {color, category, gender, price} + semantic text
         (Gemini JSON-schema output, rule-based fallback)
      ─► hard metadata filter           ← applied before ranking, not left to the LLM
      ─► BM25  ─┐
      ─► dense (Gemini embeddings, cosine, FAISS) ─┼─► Reciprocal Rank Fusion ─► top-k
      ─► CLIP text→image (optional) ───────────────┘
      ─► Gemini answers as JSON {message, recommended_ids}
      ─► every id is checked against the retrieved context; unknown ids are dropped
```

## What I built, and with what

| Problem | Solution | Tech |
|---|---|---|
| Keyword search misses intent ("elegant for a wedding") | Dense retrieval over product text; fused with BM25 | Gemini `gemini-embedding-001` (query/document task types), FAISS `IndexFlatIP` on L2-normalised vectors (cosine), optional HNSW, own BM25 (`src/lexical.py`) |
| LLM ignores constraints ("under ₹3000", "blue") | Query → structured filters applied before ranking; rule-based fallback keeps it working offline | Gemini structured output + Pydantic schema, regex/vocabulary parser (`src/query_parser.py`) |
| LLM hallucinates products | JSON answer with product ids validated against the retrieved context; deterministic "no results" message; prompt treats user text as data | Gemini JSON mode, grounding check (`src/rag_core.py`) |
| Follow-up questions ("cheaper ones") | Multi-turn chat with LLM query rewriting | Streamlit chat UI |
| "Does it actually work?" | 44 labelled queries, P@5 / Recall / MRR / nDCG, threshold sweep, LLM judge | `eval/` |
| Reliability and cost | Batched + disk-cached embeddings, retry with backoff, incremental indexing, per-stage timings, graceful fallback to offline mode | `src/embedding_utils.py`, `src/data_pipeline.py`, `src/factory.py` |
| Serving | REST API, Docker, CI | FastAPI, Dockerfile, GitHub Actions |
| Visual search | CLIP text→image and image→image over a separate FAISS index | `transformers`, `torch` (optional; see limitations) |

## Results (44 queries, real Gemini embeddings)

| config | P@5 | Recall@10 | MRR | nDCG@10 | out-of-catalogue queries rejected |
|---|---|---|---|---|---|
| BM25 | 0.605 | 0.817 | 0.824 | 0.776 | 14% |
| dense | 0.573 | 0.776 | 0.830 | 0.773 | 57% |
| **hybrid + filters** | 0.788 | **0.881** | **0.959** | **0.914** | **100%** |

nDCG@10 by query type (BM25 → hybrid + filters): attribute 0.72 → 1.0, feature 0.99 → 0.92, combined 0.88 → 1.0, semantic 0.13 → 0.46. Full tables: `eval/results/`.

Caveats: attribute/combined ground truth comes from the same catalogue columns the filter uses (partly circular). `MIN_SIMILARITY=0.65` was tuned on this same query set, and with 44 queries over 51 products small differences are noise. An LLM-judge run (`gemini-3.5-flash-lite` judging its own family, 44 queries) gave 5.0/5 helpfulness and 100% id-grounding. That is a weak, saturated signal (16% of answers used the retrieval-only fallback during Gemini 503s).

## Run it

```bash
pip install -r requirements.txt            # + requirements-clip.txt for image features
cp .env.example .env                       # set GEMINI_API_KEY
python build_index.py                      # incremental; --rebuild to start over
streamlit run app.py                       # chat UI
uvicorn api:app --reload                   # REST API: /health, /search, /chat (docs at /docs)
python -m eval.run_eval                    # metrics (also --sweep-threshold, --judge)
pip install -r requirements-dev.txt && ruff check . && pytest      # 45 offline tests (fake embedder / fake LLM)
docker compose up --build
```

Without an API key or index the app runs in offline mode (BM25 + rule-based filters), which is what CI uses.

## Security notes

- Secrets only via `.env` (git-ignored); `.env`, indexes and caches are excluded from git and Docker images.
- API inputs are bounded (query length, `k`, history size) and validated with Pydantic; image uploads are size-limited and saved under a whitelisted extension.
- LLM output is never trusted: ids are validated against the retrieved context, and answers are rendered as plain Markdown (no raw HTML).
- The container runs as a non-root user; CI has read-only permissions.
- The REST API has **no authentication or rate limiting**; put it behind a gateway before exposing it.

## Limitations

- Tiny catalogue (51 women's ethnic-wear products); semantic-query labels are heuristic.
- No product images ship with the repo, so the CLIP/image index is empty and **image search is implemented but untested**. Size charts are displayed, not used for ranking.
- Docker build and CI have not been run end to end. Streamlit calls the retrieval layer directly instead of the REST API.
- Indexes from the older L2 version are rejected; rebuild with `python build_index.py`.
