"""Unit tests: embeddings utils, vector store, BM25, query parser, pipeline."""
import numpy as np
import pytest

from src.data_pipeline import DataPipeline
from src.embedding_utils import normalize, with_retry
from src.lexical import BM25Index, tokenize
from src.query_parser import GeminiQueryParser, RuleBasedParser, color_matches
from src.vector_store import IndexFormatError, MultimodalVectorStore
from tests.conftest import DIM, FakeEmbedder, FakeLLM

# ── embedding utils ───────────────────────────────────────


def test_normalize_unit_length_and_zero_rows():
    out = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert np.all(out[1] == 0)


def test_with_retry_recovers_then_raises(monkeypatch):
    monkeypatch.setattr("src.embedding_utils.time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429")
        return "ok"

    assert with_retry(flaky) == "ok" and calls["n"] == 3
    with pytest.raises(ValueError):
        with_retry(lambda: (_ for _ in ()).throw(RuntimeError("down")), attempts=2)


# ── vector store ──────────────────────────────────────────


def test_search_returns_cosine_scores_best_first(store, products):
    q = FakeEmbedder().embed_query(products[3].embedding_text())
    results = store.search_text(q, k=3)
    assert results[0][0] == 3 and results[0][1] == pytest.approx(1.0, abs=1e-5)
    assert [s for _, s in results] == sorted((s for _, s in results), reverse=True)


def test_search_respects_allowed_ids_and_min_score(store, products):
    q = FakeEmbedder().embed_query("kurta")
    assert {i for i, _ in store.search_text(q, k=6, allowed={1, 2})} <= {1, 2}
    assert store.search_text(q, k=6, allowed=set()) == []
    assert store.search_text(q, k=6, min_score=0.999) == []


def test_duplicate_product_not_added_twice(store, products):
    assert store.add_product(products[0], np.ones(DIM)) is False
    assert len(store) == len(products) == store.text_index.ntotal


def test_dimension_mismatch_rejected(products):
    s = MultimodalVectorStore(text_dim=DIM, image_dim=8, table_dim=8)
    with pytest.raises(ValueError):
        s.add_product(products[0], np.ones(DIM + 1))


def test_save_load_roundtrip(store, tmp_path):
    paths = [str(tmp_path / n) for n in ("t.faiss", "i.faiss", "b.faiss", "meta.json")]
    store.save(*paths)
    loaded = MultimodalVectorStore.load(*paths)
    assert len(loaded) == len(store)
    assert loaded.metadata[0].to_dict() == store.metadata[0].to_dict()
    q = FakeEmbedder().embed_query("red anarkali")
    assert loaded.search_text(q, 3) == store.search_text(q, 3)


def test_legacy_index_format_is_rejected(tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_text("[]")  # old format: bare list
    with pytest.raises(IndexFormatError):
        MultimodalVectorStore.load("a", "b", "c", str(meta))


def test_hnsw_index_works_with_filter(products):
    s = MultimodalVectorStore(text_dim=DIM, image_dim=8, table_dim=8, index_type="hnsw")
    emb = FakeEmbedder()
    for p, v in zip(products, emb.embed_documents([p.embedding_text() for p in products])):
        s.add_product(p, v)
    hits = s.search_text(emb.embed_query("kurta"), k=3, allowed={0, 1, 2})
    assert hits and all(i in {0, 1, 2} for i, _ in hits)


# ── BM25 ──────────────────────────────────────────────────


def test_tokenizer_drops_stopwords_and_stems():
    assert tokenize("Show me the Kurtis for women") == ["kurta"]
    assert "palazzo" in tokenize("Palazzos")


def test_bm25_ranks_matching_doc_first_and_skips_zero_scores():
    idx = BM25Index(["red anarkali kurta", "blue cotton kurta", "black jacket"])
    res = idx.search("anarkali", k=5)
    assert [i for i, _ in res] == [0]
    assert idx.search("zzz", k=5) == []
    assert [i for i, _ in idx.search("kurta", k=5, allowed={1})] == [1]


# ── query parser ──────────────────────────────────────────


@pytest.mark.parametrize(
    "query,lo,hi",
    [
        ("kurta under 2500", None, 2500),
        ("kurta below ₹1,999", None, 1999),
        ("kurta over 5k", 5000, None),
        ("kurta between 3000 and 4000", 3000, 4000),
        ("kurta 2000-3000", 2000, 3000),
        ("kurta", None, None),
    ],
)
def test_price_parsing(query, lo, hi):
    p = RuleBasedParser().parse(query)
    assert (p.min_price, p.max_price) == (lo, hi)


def test_attribute_parsing_and_semantic_residual():
    p = RuleBasedParser().parse("navy blue kurta set for women under 4000")
    assert p.colors == ["navy"] and p.categories == ["Kurta Set"]
    assert p.gender == "Women" and p.max_price == 4000
    assert "4000" not in p.semantic_query
    assert RuleBasedParser().parse("red anarkali").lexical_query == "anarkali"


def test_women_is_not_parsed_as_men():
    assert RuleBasedParser().parse("kurta for women").gender == "Women"
    assert RuleBasedParser().parse("kurta for men").gender == "Men"


def test_color_families():
    assert color_matches("Navy Blue", ["blue"]) and color_matches("Teal", ["blue"])
    assert not color_matches("Navy Blue", ["red"])
    assert color_matches("Mustard", ["yellow"]) and not color_matches("Yellow", ["mustard"])


def test_filter_matching(products):
    p = RuleBasedParser().parse("black kurta under 2000")
    assert [x.product_id for x in products if p.matches(x)] == [1]


def test_llm_parser_uses_schema_and_validates():
    llm = FakeLLM({"semantic_query": "cheap kurta", "colors": ["blue", "unicorn"], "category": "kurta set",
                   "gender": "women", "min_price": 5000, "max_price": 1000})
    p = GeminiQueryParser(llm).parse("cheap blue kurta set")
    assert p.parser == "llm" and p.colors == ["blue"]  # unknown colour dropped
    assert p.categories == ["Kurta Set"] and p.gender == "Women"
    assert (p.min_price, p.max_price) == (1000, 5000)  # swapped back into order


def test_llm_parser_falls_back_to_rules_on_failure():
    p = GeminiQueryParser(FakeLLM(RuntimeError("boom"))).parse("red kurta under 3000")
    assert p.parser == "rules" and p.max_price == 3000


# ── data pipeline ─────────────────────────────────────────


def test_pipeline_is_incremental(tmp_path, products):
    import pandas as pd

    csv = tmp_path / "p.csv"
    rows = [
        {**p.to_dict(), "image_path": "", "size_chart_path": ""} for p in products[:3]
    ]
    pd.DataFrame(rows).to_csv(csv, index=False)

    store = MultimodalVectorStore(text_dim=DIM, image_dim=8, table_dim=8)
    DataPipeline(str(csv), embedder=FakeEmbedder(), store=store).build_index()
    assert len(store) == 3

    pd.DataFrame([{**p.to_dict(), "image_path": "", "size_chart_path": ""} for p in products]).to_csv(csv, index=False)
    DataPipeline(str(csv), embedder=FakeEmbedder(), store=store).build_index()
    assert len(store) == len(products) and store.text_index.ntotal == len(products)
