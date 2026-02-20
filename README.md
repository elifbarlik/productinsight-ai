# 🛍️ ProductInsight AI

> AI-powered semantic product search for fashion & clothing e-commerce — built with Gemini, FAISS, CLIP, and Streamlit.

---

## Overview

**ProductInsight AI** is a Retrieval-Augmented Generation (RAG) system designed for fashion e-commerce product discovery. It lets customers describe what they're looking for in natural language and returns semantically relevant clothing items, along with a conversational AI response — all powered by Google Gemini and a multimodal vector store.

---

## Problem

Traditional keyword-based product search fails when customers describe items conversationally:

- *"Something elegant for a wedding, preferably blue"*
- *"A comfortable summer outfit under ₹2000"*
- *"A casual jacket that goes well with jeans"*

Keyword search misses intent. Filters are rigid. Customers bounce.

---

## Solution

ProductInsight AI combines **semantic text embeddings**, **visual image embeddings**, and **structured size-chart embeddings** into a single multimodal FAISS vector store. When a customer queries the system:

1. The query is embedded using Gemini's `gemini-embedding-001` model.
2. The FAISS index retrieves the most semantically similar products.
3. Retrieved products are passed to a Gemini LLM as context.
4. The LLM generates a helpful, catalog-grounded response.

---

## Usage Modes

### Local Analysis (Streamlit UI)
Run the interactive web interface to query the product catalog:
```bash
streamlit run app.py
```

### Index Building (CLI)
Build or rebuild the FAISS vector index from the product CSV:
```bash
python build_index.py
```

### Testing
Run end-to-end and retrieval tests from the `tests/` directory:
```bash
python tests/test_rag.py
python tests/test_retrieval.py
python tests/test_quality.py
```

---

## Key Features

### Search Types

#### 🔤 Semantic Text Search
Embeds the customer's natural language query and retrieves products by meaning — not just keyword overlap. Handles color, category, gender, and price intent.

#### 🖼️ Visual Similarity Search
Each product image is embedded via CLIP (`openai/clip-vit-base-patch32`). The image index enables future visual query support (e.g., upload a photo to find similar items).

#### 📐 Size Chart Embedding
Product size charts (CSV files) are embedded as structured text. This enables size-aware retrieval for customers describing fit or dimensions.

#### 💬 LLM-Grounded Response
Gemini generates a natural-language response strictly grounded in retrieved products — no hallucinated items, no off-catalog recommendations.

#### 🎯 Relevance Filtering
Results are filtered by a configurable distance threshold (`RETRIEVAL_THRESHOLD`) before being passed to the LLM, ensuring only genuinely relevant products are surfaced.

---

## Architecture Overview

### Core Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `ProductRAG` | `src/rag_core.py` | Orchestrates retrieval + generation |
| `MultimodalVectorStore` | `src/vector_store.py` | FAISS text / image / table indexes |
| `DataPipeline` | `src/data_pipeline.py` | Embeds products and builds index |
| `ProductCatalog` | `src/catalog.py` | Loads and parses the product CSV |
| `Product` | `src/models.py` | Core data model |
| `embed_*` | `src/embedding_utils.py` | Gemini text + CLIP image + table embedders |
| `app.py` | — | Streamlit UI |
| `build_index.py` | — | CLI entry point for index building |

### Processing Stages

```
Product CSV
    │
    ▼
ProductCatalog.load_from_csv()
    │
    ▼
DataPipeline.prepare_product()
    ├─ embed_text()     → Gemini text embedding (3072-dim)
    ├─ embed_image()    → CLIP image embedding (512-dim)
    └─ embed_table()    → Gemini table embedding (3072-dim)
    │
    ▼
MultimodalVectorStore.add_product()
    ├─ text_index  (FAISS IndexFlatL2)
    ├─ image_index (FAISS IndexFlatL2)
    └─ table_index (FAISS IndexFlatL2)
    │
    ▼
Saved to embeddings/ directory
```

### Parsing / Validation Layer

The `Product` dataclass (`src/models.py`) validates all incoming product records:
- Price must be a positive value
- Image and size chart paths are verified; missing files emit warnings without crashing the pipeline

### External Service Integration

| Service | Purpose |
|---------|---------|
| Google Gemini (`gemini-embedding-001`) | Text and table embedding |
| Google Gemini (`gemini-2.5-flash`) | LLM response generation |
| HuggingFace CLIP (`openai/clip-vit-base-patch32`) | Image embedding |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend / UI** | [Streamlit](https://streamlit.io/) |
| **AI / Embedding** | [Google Gemini API](https://ai.google.dev/) (`gemini-embedding-001`) |
| **LLM** | Google Gemini (`gemini-2.5-flash`) |
| **Image Embedding** | [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) via HuggingFace Transformers |
| **Vector Search** | [FAISS](https://github.com/facebookresearch/faiss) (CPU) |
| **Data Processing** | Pandas, NumPy |
| **Image Processing** | Pillow, PyTorch |
| **Language / Runtime** | Python 3.10+ |

---

## Getting Started

### Prerequisites
- Python 3.10+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Option 1: Local Development

**1. Clone the repository**
```bash
git clone https://github.com/your-username/productinsight-ai.git
cd productinsight-ai
```

**2. Create and activate a virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

**5. Build the FAISS index**
```bash
python build_index.py
```

**6. Launch the app**
```bash
streamlit run app.py
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values:

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | — |
| `EMBEDDING_MODEL_NAME` | Gemini embedding model | `gemini-embedding-001` |
| `LLM_MODEL` | Gemini chat model | `gemini-2.5-flash` |
| `CLIP_MODEL_NAME` | HuggingFace CLIP model | `openai/clip-vit-base-patch32` |
| `TEXT_EMBEDDING_DIM` | Text embedding dimension | `3072` |
| `IMAGE_EMBEDDING_DIM` | Image embedding dimension | `512` |
| `TABLE_EMBEDDING_DIM` | Table embedding dimension | `3072` |
| `FAISS_TEXT_PATH` | Path to text FAISS index | `embeddings/text_index.faiss` |
| `FAISS_IMAGE_PATH` | Path to image FAISS index | `embeddings/image_index.faiss` |
| `FAISS_TABLE_PATH` | Path to table FAISS index | `embeddings/table_index.faiss` |
| `METADATA_PATH` | Path to product metadata JSON | `embeddings/metadata.json` |
| `PRODUCT_CSV_PATH` | Path to product catalog CSV | `src/data/product.csv` |
| `RETRIEVAL_THRESHOLD` | Max L2 distance for relevance filtering | `0.75` |
| `RETRIEVAL_K` | Number of products to retrieve | `5` |
| `LLM_TEMPERATURE` | LLM response temperature | `0.3` |

---

## Data Format

### `src/data/product.csv`

| Column | Type | Description |
|--------|------|-------------|
| `product_id` | int | Unique product identifier |
| `name` | str | Product name |
| `description` | str | Product description |
| `price` | float | Price in ₹ |
| `color` | str | Primary color |
| `category` | str | e.g., T-Shirt, Jacket, Dress, Jeans |
| `gender` | str | Women / Men / Unisex |
| `image_path` | str | Relative path to product image |
| `size_chart_path` | str | Relative path to size chart CSV (optional) |

---

## Testing

```bash
# End-to-end RAG system test
python tests/test_rag.py

# Retrieval and ranking test
python tests/test_retrieval.py

# Response quality evaluation
python tests/test_quality.py
```

---

## Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request


<p align="center">Built with ❤️ using Gemini · FAISS · CLIP · Streamlit</p>
