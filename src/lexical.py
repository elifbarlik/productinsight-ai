"""Dependency-free BM25 (Okapi) for the lexical half of hybrid retrieval."""
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "the", "for", "with", "of", "in", "on", "to", "me", "i", "my",
    "is", "are", "be", "some", "any", "that", "this", "it", "or", "show", "find",
    "want", "need", "looking", "look", "get", "please", "give", "have", "do", "you",
    "something", "under", "below", "above", "over", "than", "less", "more", "women",
    "woman", "womens", "men", "mens", "ladies", "wear", "s",
    # generic shopping chatter that carries no catalogue signal
    "outfit", "outfits", "clothes", "clothing", "cheap", "affordable", "budget", "best",
    "good", "nice", "beautiful", "pretty", "stylish", "latest", "trendy", "collection",
    "options", "item", "items", "product", "products", "cost", "price", "priced", "rs",
    "inr", "rupees", "set", "sets",
}
# Light normalisation of catalogue vocabulary variants.
_ALIASES = {"kurti": "kurta", "kurtis": "kurta", "kurtas": "kurta"}


def stem(token: str) -> str:
    token = _ALIASES.get(token, token)
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    tokens = (stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS)
    return [t for t in tokens if t not in _STOPWORDS]


class BM25Index:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d) for d in documents]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.term_freqs = [Counter(t) for t in self.doc_tokens]

        df: Counter = Counter()
        for tf in self.term_freqs:
            df.update(tf.keys())
        n = len(documents)
        self.idf: Dict[str, float] = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def score(self, query: str, doc_idx: int) -> float:
        tf = self.term_freqs[doc_idx]
        dl = self.doc_len[doc_idx]
        total = 0.0
        for term in tokenize(query):
            if term not in tf:
                continue
            f = tf[term]
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            total += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
        return total

    def search(
        self, query: str, k: int = 10, allowed: Optional[Iterable[int]] = None
    ) -> List[Tuple[int, float]]:
        """Return up to ``k`` (doc_idx, score) pairs with score > 0, best first."""
        candidates = range(len(self.doc_tokens)) if allowed is None else sorted(set(allowed))
        scored = [(i, self.score(query, i)) for i in candidates]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
