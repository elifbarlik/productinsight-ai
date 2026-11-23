"""Central configuration. Every value can be overridden via environment / .env."""
import os

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


# ── API / models ──────────────────────────────────────────
gemini_api_key = os.getenv("GEMINI_API_KEY")
embedding_model_name = _get("EMBEDDING_MODEL_NAME", "gemini-embedding-001")
llm_model = _get("LLM_MODEL", "gemini-3.8-flash")
clip_model_name = _get("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")

# ── Embedding dimensions ──────────────────────────────────
text_embedding_dim = int(_get("TEXT_EMBEDDING_DIM", "3072"))
image_embedding_dim = int(_get("IMAGE_EMBEDDING_DIM", "512"))
table_embedding_dim = int(_get("TABLE_EMBEDDING_DIM", "3072"))

# ── Paths ─────────────────────────────────────────────────
product_csv_path = _get("PRODUCT_CSV_PATH", "src/data/product.csv")
faiss_text_path = _get("FAISS_TEXT_PATH", "embeddings/text_index.faiss")
faiss_image_path = _get("FAISS_IMAGE_PATH", "embeddings/image_index.faiss")
faiss_table_path = _get("FAISS_TABLE_PATH", "embeddings/table_index.faiss")
metadata_path = _get("METADATA_PATH", "embeddings/metadata.json")
embedding_cache_dir = _get("EMBEDDING_CACHE_DIR", ".cache/embeddings")

# ── Retrieval ─────────────────────────────────────────────
# Vectors are L2-normalised and searched with inner product, so scores are cosine
# similarities. The previous L2 threshold of 0.75 on unit vectors corresponds to
# cos = 1 - d^2 / 2 ~= 0.72; 0.70 is the starting point; the sweep picked 0.65. Re-calibrate with
# `python -m eval.run_eval --sweep-threshold`.
min_similarity = float(_get("MIN_SIMILARITY", "0.65"))
retrieval_k = int(_get("RETRIEVAL_K", "5"))
candidate_pool = int(_get("CANDIDATE_POOL", "30"))
rrf_k = int(_get("RRF_K", "60"))
default_search_mode = _get("SEARCH_MODE", "hybrid")  # bm25 | dense | hybrid
faiss_index_type = _get("FAISS_INDEX_TYPE", "flat")  # flat | hnsw

# ── LLM ───────────────────────────────────────────────────
llm_temperature = float(_get("LLM_TEMPERATURE", "0.3"))
currency_symbol = _get("CURRENCY_SYMBOL", "₹")
