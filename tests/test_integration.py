"""Live tests against Gemini + a built index. Skipped without GEMINI_API_KEY / index.

    python build_index.py && pytest -m integration
"""
import os

import pytest

from src.config import gemini_api_key, metadata_path

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (gemini_api_key and os.path.exists(metadata_path)),
        reason="needs GEMINI_API_KEY and a built index",
    ),
]


@pytest.fixture(scope="module")
def rag():
    from src.factory import build_rag

    return build_rag(mode="hybrid")


def test_semantic_query_returns_products(rag):
    resp = rag.answer("something elegant for a wedding")
    assert resp.products and resp.grounded


def test_filters_are_enforced(rag):
    resp = rag.answer("pink kurta under 2500")
    assert all(p.price <= 2500 and "pink" in p.color.lower() for p in resp.products)


def test_out_of_catalogue_query_is_rejected(rag):
    assert rag.answer("black leather jacket").products == []
