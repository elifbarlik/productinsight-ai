"""Build (or incrementally update) the FAISS index.

    python build_index.py            # add only products missing from the index
    python build_index.py --rebuild  # rebuild from scratch
"""
import argparse
import os

from src.config import (
    faiss_image_path,
    faiss_table_path,
    faiss_text_path,
    metadata_path,
    product_csv_path,
)
from src.data_pipeline import DataPipeline
from src.embedding_utils import ClipEmbedder, GeminiEmbedder, clip_available, embed_table
from src.logging_utils import get_logger
from src.vector_store import IndexFormatError, MultimodalVectorStore

log = get_logger("build_index")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="ignore any existing index")
    parser.add_argument("--no-clip", action="store_true", help="skip image embeddings")
    args = parser.parse_args()

    store = None
    if not args.rebuild and os.path.exists(metadata_path):
        try:
            store = MultimodalVectorStore.load(
                faiss_text_path, faiss_image_path, faiss_table_path, metadata_path
            )
            log.info("loaded existing index with %d products", len(store))
        except IndexFormatError as exc:
            log.warning("%s -> rebuilding from scratch", exc)

    use_clip = clip_available() and not args.no_clip
    if not use_clip:
        log.warning("CLIP/torch not available or disabled: image index will be empty")

    pipeline = DataPipeline(
        product_csv_path,
        embedder=GeminiEmbedder(),
        clip_embedder=ClipEmbedder() if use_clip else None,
        store=store,
        table_embed_fn=embed_table,
    )
    vector_store = pipeline.build_index()
    vector_store.save(faiss_text_path, faiss_image_path, faiss_table_path, metadata_path)
    log.info("saved index: %d products -> %s", len(vector_store), faiss_text_path)


if __name__ == "__main__":
    main()
