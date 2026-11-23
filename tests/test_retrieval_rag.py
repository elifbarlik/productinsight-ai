"""Retriever, RAG orchestration, API and eval-regression tests (no network, no API key)."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import create_app
from eval.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from src.rag_core import ProductRAG, check_grounding
from src.retriever import Retriever, reciprocal_rank_fusion
from tests.conftest import FakeLLM, ids

# ── retriever ─────────────────────────────────────────────


def test_rrf_rewards_agreement_between_lists():
    fused = reciprocal_rank_fusion({"a": [1, 2, 3], "b": [3, 2, 9]}, k=60)
    assert {d for d, _ in fused[:2]} == {2, 3}  # 2 and 3 appear in both lists
    weighted = reciprocal_rank_fusion({"a": [1], "b": [2]}, {"b": 5.0}, k=60)
    assert weighted[0][0] == 2


def test_filter_is_applied_before_ranking(bm25_retriever):
    r = bm25_retriever.search("kurta under 2000", k=10)
    assert set(ids(r)) <= {1, 5}


def test_filter_toggle(bm25_retriever):
    with_f = bm25_retriever.search("red anarkali kurta under 2000", use_filter=True)
    without = bm25_retriever.search("red anarkali kurta under 2000", use_filter=False)
    assert ids(with_f) == [] and 3 in ids(without)


def test_unmatched_term_after_filters_returns_nothing(bm25_retriever):
    assert ids(bm25_retriever.search("black leather jacket")) == []
    assert ids(bm25_retriever.search("kurta for men")) == []


def test_hybrid_and_dense_modes_work(retriever):
    for mode in ("bm25", "dense", "hybrid"):
        res = retriever.search("red anarkali kurta", k=3, mode=mode)
        assert 3 in ids(res)[:2], mode
    hybrid = retriever.search("red anarkali kurta", k=3, mode="hybrid")
    assert set(hybrid.hits[0].sources) == {"bm25", "dense"}


def test_dense_threshold_rejects_weak_matches(products, store):
    from tests.conftest import FakeEmbedder

    strict = Retriever(products, store, FakeEmbedder(), min_sim=0.99)
    assert ids(strict.search("zzzz qqqq", mode="dense", use_filter=False)) == []


def test_dense_mode_without_store_raises(bm25_retriever):
    with pytest.raises(RuntimeError):
        bm25_retriever.search("kurta", mode="dense")


def test_timings_are_recorded(retriever):
    t = retriever.search("kurta", mode="hybrid").timings
    assert {"parse_ms", "bm25_ms", "embed_ms", "dense_ms", "fuse_ms"} <= set(t)


# ── RAG ───────────────────────────────────────────────────


def test_grounding_check_separates_hallucinated_ids(products):
    assert check_grounding([1, 99, 2], products) == ([1, 2], [99])


def test_rag_offline_without_llm(bm25_retriever):
    resp = ProductRAG(bm25_retriever, llm=None).answer("red anarkali")
    assert not resp.llm_used and ids(resp.search) == [3] and "Soch" in resp.message


def test_rag_no_results_explains_filters_without_calling_llm(bm25_retriever):
    llm = FakeLLM()
    resp = ProductRAG(bm25_retriever, llm).answer("kurta under 500")
    assert resp.products == [] and "price <= 500" in resp.message and llm.calls == []


def test_rag_structured_answer_drops_hallucinated_ids(bm25_retriever):
    llm = FakeLLM({"message": "Try the Soch kurta", "recommended_ids": [3, 999]})
    resp = ProductRAG(bm25_retriever, llm).answer("red anarkali")
    assert resp.llm_used and not resp.grounded and resp.dropped_ids == [999]
    assert [p.product_id for p in resp.products] == [3]


def test_rag_falls_back_when_llm_fails(bm25_retriever):
    resp = ProductRAG(bm25_retriever, FakeLLM(RuntimeError("quota"))).answer("red anarkali")
    assert not resp.llm_used and ids(resp.search) == [3]


def test_rag_prompt_wraps_user_query_as_data(bm25_retriever):
    llm = FakeLLM({"message": "ok", "recommended_ids": []})
    ProductRAG(bm25_retriever, llm).answer("red anarkali")
    system, user, _ = llm.calls[0]
    assert "<customer_question>" in user and "untrusted" in system


def test_rag_rewrites_follow_up_with_history(bm25_retriever):
    llm = FakeLLM(
        {"standalone_query": "red anarkali"},
        {"message": "Here", "recommended_ids": [3]},
    )
    history = [{"role": "user", "content": "show me anarkali kurtas"}, {"role": "assistant", "content": "..."}]
    resp = ProductRAG(bm25_retriever, llm).answer("in red please", history=history)
    assert resp.standalone_query == "red anarkali" and [p.product_id for p in resp.products] == [3]


# ── API ───────────────────────────────────────────────────


@pytest.fixture
def client(bm25_retriever):
    with TestClient(create_app(ProductRAG(bm25_retriever, llm=None))) as c:
        yield c


def test_api_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["products"] == 6 and not body["llm_enabled"]


def test_api_search_and_validation(client):
    r = client.post("/search", json={"query": "black kurta under 2000", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert [x["product_id"] for x in body["results"]] == [1]
    assert body["parsed"]["max_price"] == 2000
    assert client.post("/search", json={"query": "", "k": 3}).status_code == 422
    assert client.post("/search", json={"query": "x", "k": 99}).status_code == 422
    assert client.post("/search", json={"query": "kurta", "mode": "dense"}).status_code == 400


def test_api_chat_offline(client):
    body = client.post("/chat", json={"query": "red anarkali", "history": []}).json()
    assert body["products"][0]["product_id"] == 3 and body["llm_used"] is False


# ── eval metrics ──────────────────────────────────────────


def test_ranking_metrics():
    ranked, rel = [5, 1, 2, 9], {1, 2, 7}
    assert precision_at_k(ranked, rel, 2) == 0.5
    assert recall_at_k(ranked, rel, 4) == pytest.approx(2 / 3)
    assert reciprocal_rank(ranked, rel) == 0.5
    assert ndcg_at_k([1, 2, 7], rel, 3) == pytest.approx(1.0)
    assert ndcg_at_k([], rel, 3) == 0.0


# ── regression guard on the real catalogue (offline, BM25 + filters) ──


def test_eval_set_is_consistent_with_catalog():
    queries = [json.loads(l) for l in Path("eval/queries.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(queries) >= 40
    assert {q["type"] for q in queries} >= {"attribute", "feature", "combined", "semantic", "negative"}
    assert all(q["relevant"] for q in queries if q["type"] != "negative")


def test_offline_retrieval_quality_does_not_regress():
    from eval.run_eval import evaluate, load_queries
    from src.catalog import ProductCatalog

    catalog = ProductCatalog("src/data/product.csv")
    retriever = Retriever(catalog.products, mode="bm25")
    summary = evaluate(retriever, load_queries(), "bm25+filter")["summary"]
    assert summary["overall"]["ndcg10"] >= 0.85
    assert summary["neg_reject_rate"] == 1.0
