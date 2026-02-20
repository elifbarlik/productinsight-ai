import os

import streamlit as st
from src.rag_core import ProductRAG
from src.config import relevance_threshold

st.set_page_config(
    page_title="ProductInsight AI",
    page_icon="🛍️",
    layout="centered",
)


def distance_to_score(distance: float, max_distance: float = relevance_threshold * 2) -> int:
    score = max(0, min(100, int((1 - distance / max_distance) * 100)))
    return score



st.markdown("""
<style>
    .block-container {
        max-width: 720px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

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
    .product-card .relevance-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    .app-footer {
        text-align: center;
        color: #555;
        font-size: 0.8rem;
        padding-top: 2rem;
    }

    div[data-testid="stTextInput"] input {
        border-radius: 10px;
    }
    button[kind="primary"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <h1>🛍️ ProductInsight AI</h1>
    <p>AI-Powered E-Commerce Assistant — ask anything about our catalog</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag():
    return ProductRAG()

rag = load_rag()

user_query = st.text_input(
    "🔍 What are you looking for?",
    placeholder="E.g. 'black jacket under $100'",
    label_visibility="collapsed",
)

search_clicked = st.button("Search", type="primary", use_container_width=True)

if user_query and search_clicked:
    with st.spinner("Searching catalog…"):
        response, products, distances = rag.query(user_query)

    st.markdown("#### 🤖 Assistant")
    st.info(response)

    if products:

        st.markdown(f"#### 📦 Found {len(products)} product(s)")

        cols = st.columns(min(3, len(products)))
        for i, product in enumerate(products):
            with cols[i % len(cols)]:
                dist = distances[i] if i < len(distances) else 1.0
                score = distance_to_score(dist)

                if score >= 80:
                    score_color = "#2ECC71"
                elif score >= 50:
                    score_color = "#F39C12"
                else:
                    score_color = "#E74C3C"

                if os.path.exists(product.image_path):
                    st.image(product.image_path, use_container_width=True)

                st.markdown(f"""
                <div class="product-card">
                    <h4>{product.name}</h4>
                    <span class="price">₹{product.price}</span><br>
                    <span class="relevance-badge" style="background:{score_color}22; color:{score_color};">
                        Match: {score}%
                    </span><br>
                    <div style="margin: 5px 0;">
                        <span class="meta" style="background: #333; padding: 2px 6px; border-radius: 4px; margin-right: 4px;">{product.gender}</span>
                        <span class="meta" style="background: #333; padding: 2px 6px; border-radius: 4px;">{product.category}</span>
                    </div>
                    <span class="meta">Color: {product.color}</span>
                    <p class="meta" style="margin-top: 5px;">{product.description[:100]}...</p>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.warning("No products found for your query.")

st.markdown('<div class="app-footer">Built with Gemini + FAISS + Streamlit</div>', unsafe_allow_html=True)
