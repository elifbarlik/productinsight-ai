"""Wiring: build the retriever / RAG with graceful degradation.

* index + API key present  -> hybrid retrieval, Gemini query parsing and answers
* otherwise                -> BM25-only retrieval with the rule-based parser (offline mode)
"""
import os
from typing import Optional

from src.catalog import ProductCatalog
from src.config import (
    default_search_mode,
    faiss_image_path,
    faiss_table_path,
    faiss_text_path,
    gemini_api_key,
    metadata_path,
    product_csv_path,
)
from src.embedding_utils import ClipEmbedder, GeminiEmbedder, clip_available
from src.logging_utils import get_logger
from src.query_parser import GeminiQueryParser, RuleBasedParser
from src.rag_core import ProductRAG
from src.retriever import Retriever
from src.vector_store import IndexFormatError, MultimodalVectorStore

log = get_logger(__name__)


def build_retriever(
    mode: Optional[str] = None,
    use_llm_parser: bool = True,
    allow_offline: bool = True,
) -> Retriever:
    mode = mode or default_search_mode
    llm = _maybe_llm()
    parser = GeminiQueryParser(llm) if (llm and use_llm_parser) else RuleBasedParser()

    store = None
    if gemini_api_key and os.path.exists(metadata_path):
        try:
            store = MultimodalVectorStore.load(
                faiss_text_path, faiss_image_path, faiss_table_path, metadata_path
            )
        except IndexFormatError as exc:
            log.warning("%s", exc)

    if store is not None:
        clip = ClipEmbedder() if clip_available() else None
        return Retriever(
            store.metadata, store=store, embedder=GeminiEmbedder(), parser=parser,
            clip_embedder=clip, mode=mode,
        )

    if not allow_offline:
        raise RuntimeError("Dense retrieval unavailable: set GEMINI_API_KEY and run build_index.py")
    log.warning("running in OFFLINE mode (BM25 + rule-based parser); no usable index/API key")
    catalog = ProductCatalog(product_csv_path)
    return Retriever(catalog.products, parser=parser, mode="bm25")


def _maybe_llm():
    if not gemini_api_key:
        return None
    from src.llm import GeminiLLM

    return GeminiLLM()


def build_rag(mode: Optional[str] = None) -> ProductRAG:
    llm = _maybe_llm()
    return ProductRAG(build_retriever(mode), llm=llm)
