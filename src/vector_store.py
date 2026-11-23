"""FAISS-backed multimodal vector store (text / image / size-chart indexes).

All indexes use inner product over L2-normalised vectors, i.e. cosine similarity
(higher is better). Row ``i`` of every index belongs to ``metadata[i]``.
"""
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np

from src.config import (
    faiss_index_type,
    image_embedding_dim,
    table_embedding_dim,
    text_embedding_dim,
)
from src.embedding_utils import normalize
from src.models import Product

FORMAT_VERSION = 2


class IndexFormatError(RuntimeError):
    """Raised when an index on disk was built by an incompatible version."""


class MultimodalVectorStore:
    def __init__(
        self,
        text_dim: Optional[int] = None,
        image_dim: Optional[int] = None,
        table_dim: Optional[int] = None,
        index_type: Optional[str] = None,
    ):
        self.text_dim = text_dim or text_embedding_dim
        self.image_dim = image_dim or image_embedding_dim
        self.table_dim = table_dim or table_embedding_dim
        self.index_type = index_type or faiss_index_type

        self.text_index = self._make_index(self.text_dim)
        self.image_index = self._make_index(self.image_dim)
        self.table_index = self._make_index(self.table_dim)

        self.metadata: List[Product] = []
        self.id_map: Dict[int, int] = {}

    # ── construction ──────────────────────────────────────
    def _make_index(self, dim: int):
        if self.index_type == "hnsw":
            return faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        if self.index_type == "flat":
            return faiss.IndexFlatIP(dim)
        raise ValueError(f"Unknown index type: {self.index_type!r} (use 'flat' or 'hnsw')")

    def _vec(self, emb: Optional[np.ndarray], dim: int) -> np.ndarray:
        if emb is None:
            return np.zeros((1, dim), dtype="float32")
        arr = normalize(np.asarray(emb, dtype="float32").reshape(1, -1))
        if arr.shape[1] != dim:
            raise ValueError(f"Embedding has dim {arr.shape[1]}, index expects {dim}")
        return arr

    def has_product(self, product_id: int) -> bool:
        return product_id in self.id_map

    def add_product(
        self,
        product: Product,
        text_embedding: np.ndarray,
        image_embedding: Optional[np.ndarray] = None,
        table_embedding: Optional[np.ndarray] = None,
    ) -> bool:
        """Add a product. Returns False (and does nothing) if the id is already indexed."""
        if self.has_product(product.product_id):
            return False
        text_vec = self._vec(text_embedding, self.text_dim)
        image_vec = self._vec(image_embedding, self.image_dim)
        table_vec = self._vec(table_embedding, self.table_dim)

        self.text_index.add(text_vec)
        self.image_index.add(image_vec)
        self.table_index.add(table_vec)

        self.id_map[product.product_id] = len(self.metadata)
        self.metadata.append(product)
        return True

    # ── search ────────────────────────────────────────────
    @staticmethod
    def _search(
        index,
        query: np.ndarray,
        k: int,
        allowed: Optional[Iterable[int]] = None,
        min_score: Optional[float] = None,
    ) -> List[Tuple[int, float]]:
        n = index.ntotal
        if n == 0 or k <= 0:
            return []
        q = normalize(np.asarray(query, dtype="float32").reshape(1, -1))
        k = min(k, n)

        if allowed is None:
            scores, ids = index.search(q, k)
        else:
            allowed_ids = np.array(sorted(set(allowed)), dtype="int64")
            if allowed_ids.size == 0:
                return []
            sel = faiss.IDSelectorBatch(allowed_ids)
            params = (
                faiss.SearchParametersHNSW()
                if isinstance(index, faiss.IndexHNSW)
                else faiss.SearchParameters()
            )
            params.sel = sel
            scores, ids = index.search(q, k, params=params)

        results = [
            (int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0
        ]
        if min_score is not None:
            results = [(i, s) for i, s in results if s >= min_score]
        return results

    def search_text(self, query_embedding, k=5, allowed=None, min_score=None):
        return self._search(self.text_index, query_embedding, k, allowed, min_score)

    def search_image(self, query_embedding, k=5, allowed=None, min_score=None):
        return self._search(self.image_index, query_embedding, k, allowed, min_score)

    def search_table(self, query_embedding, k=5, allowed=None, min_score=None):
        return self._search(self.table_index, query_embedding, k, allowed, min_score)

    # ── persistence ───────────────────────────────────────
    def save(self, text_path: str, image_path: str, table_path: str, metadata_path: str):
        for path in (text_path, image_path, table_path, metadata_path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(self.text_index, text_path)
        faiss.write_index(self.image_index, image_path)
        faiss.write_index(self.table_index, table_path)

        payload = {
            "version": FORMAT_VERSION,
            "metric": "inner_product",
            "products": [p.to_dict() for p in self.metadata],
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, text_path: str, image_path: str, table_path: str, metadata_path: str):
        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict) or payload.get("version") != FORMAT_VERSION:
            raise IndexFormatError(
                "Index was built with an older version (L2 / unnormalised vectors). "
                "Rebuild it with: python build_index.py"
            )

        store = cls.__new__(cls)
        store.text_index = faiss.read_index(text_path)
        store.image_index = faiss.read_index(image_path)
        store.table_index = faiss.read_index(table_path)
        store.text_dim = store.text_index.d
        store.image_dim = store.image_index.d
        store.table_dim = store.table_index.d
        store.index_type = "hnsw" if isinstance(store.text_index, faiss.IndexHNSW) else "flat"

        store.metadata = [Product.from_dict(d) for d in payload["products"]]
        store.id_map = {p.product_id: i for i, p in enumerate(store.metadata)}
        return store

    def __len__(self):
        return len(self.metadata)
