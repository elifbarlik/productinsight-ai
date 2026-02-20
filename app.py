# app.py
import os

import streamlit as st
from src.rag_core import ProductRAG

# Page config
st.set_page_config(
    page_title="ProductInsight AI",
    page_icon="🛍️",
    layout="centered",
)

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    .block-container {
        max-width: 720px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* ── Header ── */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .app-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .app-header p {
        color: #888;
        font-size: 0.95rem;
        margin-top: 0.25rem;
    }

    /* ── Product Card ── */
    .product-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s;
    }
    .product-card:hover {
        border-color: #6c63ff;
    }
    .product-card h4 {
        margin: 0.5rem 0 0.25rem;
        font-size: 1rem;
    }
    .product-card .price {
        color: #6c63ff;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .product-card .meta {
        color: #aaa;
        font-size: 0.85rem;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        color: #555;
        font-size: 0.8rem;
        padding-top: 2rem;
    }

    /* ── Misc ── */
    div[data-testid="stTextInput"] input {
        border-radius: 10px;
    }
    button[kind="primary"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🛍️ ProductInsight AI</h1>
    <p>AI-Powered E-Commerce Assistant — ask anything about our catalog</p>
</div>
""", unsafe_allow_html=True)

# ── Load RAG (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_rag():
    return ProductRAG()

rag = load_rag()

# ── Search Bar ──────────────────────────────────────────────────────────
user_query = st.text_input(
    "🔍 What are you looking for?",
    placeholder="E.g. 'black jacket under $100'",
    label_visibility="collapsed",
)

search_clicked = st.button("Search", type="primary", use_container_width=True)

# ── Results ─────────────────────────────────────────────────────────────
if user_query and search_clicked:
    with st.spinner("Searching catalog…"):
        response, products = rag.query(user_query)

    # AI response
    st.markdown("#### 🤖 Assistant")
    st.info(response)

    # Products
    if products:
        st.markdown(f"#### 📦 Found {len(products)} product(s)")

        cols = st.columns(min(3, len(products)))
        for i, product in enumerate(products):
            with cols[i % 3]:
                if os.path.exists(product.image_path):
                    st.image(product.image_path, use_container_width=True)

                st.markdown(f"""
                <div class="product-card">
                    <h4>{product.name}</h4>
                    <span class="price">${product.price}</span><br>
                    <span class="meta">Color: {product.color}</span>
                    <p class="meta">{product.description}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No products found for your query.")

# ── Footer ──────────────────────────────────────────────────────────────
st.markdown('<div class="app-footer">Built with Gemini + FAISS + Streamlit</div>', unsafe_allow_html=True)