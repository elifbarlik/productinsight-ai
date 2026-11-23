"""Catalog -> embeddings -> vector store (batched, incremental)."""
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.catalog import ProductCatalog
from src.logging_utils import get_logger
from src.vector_store import MultimodalVectorStore

log = get_logger(__name__)


class DataPipeline:
    def __init__(
        self,
        csv_path: str,
        embedder=None,
        clip_embedder=None,
        store: Optional[MultimodalVectorStore] = None,
        table_embed_fn=None,
    ):
        """
        ``store``: pass an existing store to index only *new* products (incremental).
        ``clip_embedder``: optional; without it image vectors are left as zeros.
        """
        self.catalog = ProductCatalog(csv_path)
        self.embedder = embedder
        self.clip_embedder = clip_embedder
        self.vector_store = store if store is not None else MultimodalVectorStore()
        self.table_embed_fn = table_embed_fn

    def _table_embeddings(self, products) -> Dict[str, np.ndarray]:
        """Size charts are shared between products: embed each distinct file once."""
        if self.table_embed_fn is None:
            return {}
        out: Dict[str, np.ndarray] = {}
        for p in products:
            if p.size_chart_path and str(p.size_chart_path) not in out and Path(p.size_chart_path).exists():
                out[str(p.size_chart_path)] = self.table_embed_fn(str(p.size_chart_path))
        return out

    def build_index(self) -> MultimodalVectorStore:
        new_products = [
            p for p in self.catalog.products if not self.vector_store.has_product(p.product_id)
        ]
        log.info("%d products in catalog, %d new to index", len(self.catalog), len(new_products))
        if not new_products:
            return self.vector_store

        text_embs = self.embedder.embed_documents([p.embedding_text() for p in new_products])
        tables = self._table_embeddings(new_products)

        for product, text_emb in zip(new_products, text_embs):
            image_emb = None
            if self.clip_embedder is not None and product.image_path and Path(product.image_path).exists():
                try:
                    image_emb = self.clip_embedder.embed_image(str(product.image_path))
                except Exception as exc:  # noqa: BLE001 - one bad image must not stop the build
                    log.warning("image embedding failed for %s: %s", product.product_id, exc)
            table_emb = tables.get(str(product.size_chart_path)) if product.size_chart_path else None
            self.vector_store.add_product(product, text_emb, image_emb, table_emb)

        log.info("indexed %d products (total %d)", len(new_products), len(self.vector_store))
        return self.vector_store
