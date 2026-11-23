import os
import tempfile

import pandas as pd
import streamlit as st

from src.config import currency_symbol
from src.factory import build_rag

MAX_UPLOAD_BYTES = 5 * 1024 * 1024

st.set_page_config(page_title="ProductInsight AI", page_icon="🛍️", layout="wide")


@st.cache_resource
def load_rag():
    return build_rag()


rag = load_rag()
retriever = rag.retriever

# ── sidebar: search controls ──────────────────────────────
st.sidebar.header("Search settings")
modes = ["hybrid", "dense", "bm25"] if retriever.dense_available else ["bm25"]
mode = st.sidebar.selectbox("Retrieval mode", modes, help="hybrid = BM25 + dense fused with RRF")
use_filter = st.sidebar.checkbox("Parse query into filters", value=True,
                                 help="Colour / category / gender / price become hard filters")
use_image = st.sidebar.checkbox(
    "Also match images (CLIP)", value=False, disabled=not retriever.image_search_available,
    help="Requires torch + transformers and an index with image vectors",
)
k = st.sidebar.slider("Results", 1, 10, 5)
show_debug = st.sidebar.checkbox("Show debug info", value=False)

if not retriever.dense_available:
    st.sidebar.warning("Offline mode: BM25 only (no API key / index). Answers are not LLM-generated.")
elif rag.llm is None:
    st.sidebar.warning("No LLM configured; showing retrieval-only answers.")

if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

st.title("🛍️ ProductInsight AI")
st.caption("Conversational product search for ethnic wear: ask follow-ups like “cheaper ones” or “in blue”.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def explain(hit) -> str:
    """Why was this product returned?"""
    bits = []
    if "dense" in hit.sources:
        bits.append(f"semantic similarity {hit.sources['dense']:.2f}")
    if "bm25" in hit.sources:
        bits.append(f"keyword score {hit.sources['bm25']:.1f}")
    if "image" in hit.sources:
        bits.append(f"image similarity {hit.sources['image']:.2f}")
    return ", ".join(bits) or "matched your filters"


def render_products(hits, key_prefix: str):
    if not hits:
        return
    cols = st.columns(min(3, len(hits)))
    for i, hit in enumerate(hits):
        p = hit.product
        with cols[i % len(cols)]:
            with st.container(border=True):
                if p.image_path and os.path.exists(p.image_path):
                    st.image(str(p.image_path), use_container_width=True)
                st.markdown(f"**{p.name}**")
                st.markdown(f"{currency_symbol}{p.price:g} · {p.color} · {p.category}")
                st.caption(f"Why: {explain(hit)}")
                if p.size_chart_path and os.path.exists(p.size_chart_path):
                    with st.expander("Size chart"):
                        st.dataframe(pd.read_csv(p.size_chart_path), hide_index=True)


def render_debug(info: dict):
    with st.expander("Debug"):
        st.json(info)


# ── image upload: "find similar items" ────────────────────
with st.sidebar.expander("Find similar to a photo", expanded=False):
    if not retriever.image_search_available:
        st.caption("Needs `pip install -r requirements-clip.txt` and an index built with CLIP.")
    else:
        upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if upload is not None and upload.size > MAX_UPLOAD_BYTES:
            st.error("Image too large (max 5 MB).")
        elif upload is not None and st.button("Search by image"):
            # suffix is taken from a fixed whitelist, never from the user-supplied file name
            suffix = ".png" if upload.type == "image/png" else ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(upload.getbuffer())
            try:
                hits = retriever.search_by_image(tmp.name, k=k)
            except Exception:  # noqa: BLE001 - corrupt / unsupported image
                st.error("Could not read that image.")
                hits = None
            finally:
                os.unlink(tmp.name)
            if hits is not None:
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Items visually similar to your photo:", "hits": hits}
                )
                st.rerun()

# ── conversation ──────────────────────────────────────────
for n, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_products(msg.get("hits", []), f"m{n}")
        if show_debug and msg.get("debug"):
            render_debug(msg["debug"])

if query := st.chat_input("What are you looking for? e.g. “red anarkali under ₹6000”"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching catalog…"):
            resp = rag.answer(query, history=history, k=k, mode=mode, use_filter=use_filter, use_image=use_image)
        st.markdown(resp.message)
        render_products(resp.hits, "new")
        debug = {
            "standalone_query": resp.standalone_query,
            "filters": resp.search.parsed.describe(),
            "parser": resp.search.parsed.parser,
            "mode": resp.search.mode,
            "grounded": resp.grounded,
            "timings_ms": {**resp.search.timings, **resp.timings},
        }
        if show_debug:
            render_debug(debug)
    st.session_state.messages.append(
        {"role": "assistant", "content": resp.message, "hits": resp.hits, "debug": debug}
    )

st.markdown("<div style='text-align:center;color:#777;font-size:0.8rem;padding-top:2rem'>"
            "Built with Gemini · FAISS · BM25 · CLIP · Streamlit</div>", unsafe_allow_html=True)
