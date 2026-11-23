"""Hybrid retrieval: structured filter -> (BM25 | dense | CLIP text-to-image) -> RRF fusion."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.config import candidate_pool, min_similarity, rrf_k
from src.lexical import BM25Index, tokenize
from src.logging_utils import get_logger, timed
from src.models import Product
from src.query_parser import ParsedQuery, QueryParser, RuleBasedParser
from src.vector_store import MultimodalVectorStore

log = get_logger(__name__)

MODES = ("bm25", "dense", "hybrid")


@dataclass
class SearchHit:
    product: Product
    score: float
    sources: Dict[str, float] = field(default_factory=dict)  # e.g. {"dense": 0.81, "bm25": 7.2}


@dataclass
class SearchResult:
    query: str
    parsed: ParsedQuery
    hits: List[SearchHit]
    mode: str
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def products(self) -> List[Product]:
        return [h.product for h in self.hits]


def reciprocal_rank_fusion(
    rankings: Dict[str, Sequence[int]],
    weights: Optional[Dict[str, float]] = None,
    k: int = 60,
) -> List[tuple]:
    """Fuse several ranked id lists. score(d) = sum_r w_r / (k + rank_r(d)), rank from 1."""
    fused: Dict[int, float] = {}
    for name, ranked in rankings.items():
        w = (weights or {}).get(name, 1.0)
        for rank, doc in enumerate(ranked, start=1):
            fused[doc] = fused.get(doc, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


class Retriever:
    """
    ``products`` is the aligned list (row i of the vector store == products[i]).
    ``store``/``embedder`` are optional: without them only BM25 mode is available,
    which keeps the system usable offline, in CI and without an API key.
    """

    def __init__(
        self,
        products: Sequence[Product],
        store: Optional[MultimodalVectorStore] = None,
        embedder=None,
        parser: Optional[QueryParser] = None,
        clip_embedder=None,
        mode: str = "hybrid",
        use_filter: bool = True,
        min_sim: float = min_similarity,
        pool: int = candidate_pool,
        rrf_constant: int = rrf_k,
        image_weight: float = 0.5,
    ):
        self.products = list(products)
        self.store = store
        self.embedder = embedder
        self.clip_embedder = clip_embedder
        self.parser = parser or RuleBasedParser()
        self.mode = mode
        self.use_filter = use_filter
        self.min_sim = min_sim
        self.pool = pool
        self.rrf_constant = rrf_constant
        self.image_weight = image_weight
        self.bm25 = BM25Index([p.embedding_text() for p in self.products])

    @property
    def dense_available(self) -> bool:
        return self.store is not None and self.embedder is not None

    @property
    def image_search_available(self) -> bool:
        return self.store is not None and self.clip_embedder is not None

    def _allowed(self, parsed: ParsedQuery) -> Optional[set]:
        if not parsed.has_filters:
            return None
        return {i for i, p in enumerate(self.products) if parsed.matches(p)}

    def search(
        self,
        query: str,
        k: int = 5,
        mode: Optional[str] = None,
        use_filter: Optional[bool] = None,
        use_image: bool = False,
        parsed: Optional[ParsedQuery] = None,
    ) -> SearchResult:
        mode = mode or self.mode
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode != "bm25" and not self.dense_available:
            raise RuntimeError(f"mode '{mode}' needs a vector store and an embedder")
        use_filter = self.use_filter if use_filter is None else use_filter

        timings: Dict[str, float] = {}
        with timed(timings, "parse_ms"):
            parsed = parsed or self.parser.parse(query)
        semantic = parsed.semantic_query if use_filter else query

        allowed = None
        if use_filter:
            allowed = self._allowed(parsed)
            if allowed is not None and not allowed:
                return SearchResult(query, parsed, [], mode, timings)

        rankings: Dict[str, List[int]] = {}
        raw_scores: Dict[str, Dict[int, float]] = {}

        if mode in ("bm25", "hybrid"):
            # Match only the words the filters did not already consume; if nothing is left
            # (e.g. "black kurta"), rank the filtered candidates with the full query.
            lexical = parsed.lexical_query if use_filter and parsed.lexical_query is not None else semantic
            if not tokenize(lexical):
                lexical = semantic
            with timed(timings, "bm25_ms"):
                found = self.bm25.search(lexical, k=self.pool, allowed=allowed)
            rankings["bm25"] = [i for i, _ in found]
            raw_scores["bm25"] = dict(found)

        if mode in ("dense", "hybrid"):
            with timed(timings, "embed_ms"):
                q_emb = self.embedder.embed_query(semantic)
            with timed(timings, "dense_ms"):
                found = self.store.search_text(
                    q_emb, k=self.pool, allowed=allowed, min_score=self.min_sim
                )
            rankings["dense"] = [i for i, _ in found]
            raw_scores["dense"] = dict(found)

        if use_image and self.image_search_available and mode != "bm25":
            with timed(timings, "image_ms"):
                clip_q = self.clip_embedder.embed_text(semantic)
                found = self.store.search_image(clip_q, k=self.pool, allowed=allowed)
            rankings["image"] = [i for i, _ in found]
            raw_scores["image"] = dict(found)

        with timed(timings, "fuse_ms"):
            if len(rankings) == 1:
                (name, ranked), = rankings.items()
                fused = [(i, raw_scores[name][i]) for i in ranked]
            else:
                fused = reciprocal_rank_fusion(
                    rankings, {"image": self.image_weight}, self.rrf_constant
                )

        hits = [
            SearchHit(
                self.products[i],
                float(score),
                {name: scores[i] for name, scores in raw_scores.items() if i in scores},
            )
            for i, score in fused[:k]
        ]
        return SearchResult(query, parsed, hits, mode, timings)

    def search_by_image(self, image_path: str, k: int = 5) -> List[SearchHit]:
        """Image-to-image similarity through the CLIP image index."""
        if not self.image_search_available:
            raise RuntimeError("image search needs CLIP (pip install -r requirements-clip.txt)")
        emb = self.clip_embedder.embed_image(image_path)
        found = self.store.search_image(emb, k=k)
        return [
            SearchHit(self.products[i], s, {"image": s})
            for i, s in found
            if float(np.linalg.norm(self.store.image_index.reconstruct(i))) > 0
        ]
