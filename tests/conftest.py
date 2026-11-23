import hashlib
import re

import numpy as np
import pytest

from src.embedding_utils import normalize
from src.lexical import tokenize
from src.models import Product
from src.retriever import Retriever
from src.vector_store import MultimodalVectorStore

DIM = 64


def make_products():
    rows = [
        (1, "Anouk Women Black Printed Kurta", "Black printed straight kurta cotton", 1500, "Black", "Kurta"),
        (2, "Libas Women Navy Blue Kurta with Palazzos", "Navy blue floral printed kurta set pure cotton", 3600, "Navy Blue", "Kurta Set"),
        (3, "Soch Women Red Anarkali Kurta", "Red thread work georgette anarkali kurta", 3500, "Red", "Kurta"),
        (4, "KALINI Women Pink Sharara Set", "Pink embroidered gotta patti top with sharara and dupatta", 5000, "Pink", "Sharara Set"),
        (5, "Varanga Women Mustard Embroidered Kurta", "Mustard yellow yoke embroidered straight kurta", 2000, "Mustard", "Kurta"),
        (6, "Anubhutee Women Teal Kurta Set", "Teal printed kurta with trousers and dupatta", 5300, "Teal", "Kurta Set"),
    ]
    return [Product(i, n, d, float(p), c, cat, "Women") for i, n, d, p, c, cat in rows]


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: shared words => higher cosine."""

    def __init__(self, dim: int = DIM):
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        return normalize(v)

    def embed_query(self, text):
        return self._embed(text)

    def embed_documents(self, texts):
        return np.vstack([self._embed(t) for t in texts])


class FakeLLM:
    """Scripted LLM: returns queued JSON payloads / raises queued exceptions."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, system, user, schema, **kwargs):
        self.calls.append((system, user, schema))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def generate_text(self, system, user):
        raise NotImplementedError


@pytest.fixture
def products():
    return make_products()


@pytest.fixture
def store(products):
    s = MultimodalVectorStore(text_dim=DIM, image_dim=8, table_dim=8, index_type="flat")
    emb = FakeEmbedder()
    for p, vec in zip(products, emb.embed_documents([p.embedding_text() for p in products])):
        s.add_product(p, vec)
    return s


@pytest.fixture
def retriever(products, store):
    return Retriever(products, store=store, embedder=FakeEmbedder(), min_sim=0.0)


@pytest.fixture
def bm25_retriever(products):
    return Retriever(products, mode="bm25")


def ids(result):
    return [p.product_id for p in result.products]


def slug(text):
    return re.sub(r"\W+", "-", text.lower())
