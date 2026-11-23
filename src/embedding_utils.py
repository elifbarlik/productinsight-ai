"""Embedding backends: Gemini (text/table), CLIP (image / text-to-image).

* All vectors are L2-normalised so inner product == cosine similarity.
* Gemini calls are retried with exponential backoff and cached on disk
  (query embeddings are the hot path; document embeddings are cached too so
  rebuilding the index is cheap).
* CLIP / torch are imported lazily: the rest of the system works without them.
"""
import hashlib
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from src.config import (
    clip_model_name,
    embedding_cache_dir,
    embedding_model_name,
    gemini_api_key,
)
from src.logging_utils import get_logger

log = get_logger(__name__)

_BATCH_SIZE = 50
_client = None
_clip_model = None
_clip_processor = None


# ── helpers ───────────────────────────────────────────────
def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows (a single 1-D vector is also accepted). Zero rows stay zero."""
    v = np.asarray(vectors, dtype="float32")
    single = v.ndim == 1
    v = np.atleast_2d(v)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out = v / norms
    return out[0] if single else out


def with_retry(fn: Callable, attempts: int = 4, base_delay: float = 1.0):
    """Call ``fn`` retrying with exponential backoff (rate limits, transient errors)."""
    last: Optional[Exception] = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - SDK raises many exception types
            last = exc
            if i < attempts - 1:
                delay = base_delay * (2**i)
                log.warning("API call failed (%s); retry %d in %.1fs", exc, i + 1, delay)
                time.sleep(delay)
    raise ValueError(f"API call failed after {attempts} attempts: {last}") from last


def _get_client():
    global _client
    if _client is None:
        if not gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai

        _client = genai.Client(api_key=gemini_api_key)
    return _client


# ── disk cache ────────────────────────────────────────────
def _cache_path(task: str, text: str) -> Path:
    digest = hashlib.sha256(f"{embedding_model_name}|{task}|{text}".encode()).hexdigest()
    return Path(embedding_cache_dir) / f"{digest}.npy"


def _cache_get(task: str, text: str) -> Optional[np.ndarray]:
    path = _cache_path(task, text)
    if path.exists():
        try:
            return np.load(path)
        except Exception:  # noqa: BLE001 - corrupt cache entry, recompute
            return None
    return None


def _cache_put(task: str, text: str, vec: np.ndarray) -> None:
    try:
        path = _cache_path(task, text)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vec)
    except OSError as exc:
        log.warning("could not write embedding cache: %s", exc)


# ── Gemini text embeddings ────────────────────────────────
def _embed_gemini(texts: List[str], task_type: str) -> np.ndarray:
    from google.genai import types

    client = _get_client()
    config = types.EmbedContentConfig(task_type=task_type)
    response = with_retry(
        lambda: client.models.embed_content(
            model=embedding_model_name, contents=texts, config=config
        )
    )
    return normalize(np.array([e.values for e in response.embeddings], dtype="float32"))


def embed_text(text: str, task_type: str = "RETRIEVAL_QUERY") -> np.ndarray:
    """Embed one string. Use RETRIEVAL_QUERY for queries, RETRIEVAL_DOCUMENT for documents."""
    cached = _cache_get(task_type, text)
    if cached is not None:
        return cached
    vec = _embed_gemini([text], task_type)[0]
    _cache_put(task_type, text, vec)
    return vec


def embed_texts_batch(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """Embed many strings in API-sized batches, using the disk cache where possible."""
    if not texts:
        return np.empty((0, 0), dtype="float32")

    results: List[Optional[np.ndarray]] = [_cache_get(task_type, t) for t in texts]
    missing = [i for i, r in enumerate(results) if r is None]
    for start in range(0, len(missing), _BATCH_SIZE):
        chunk_idx = missing[start : start + _BATCH_SIZE]
        vecs = _embed_gemini([texts[i] for i in chunk_idx], task_type)
        for i, vec in zip(chunk_idx, vecs):
            results[i] = vec
            _cache_put(task_type, texts[i], vec)
    return np.vstack(results)


def table_to_text(csv_path: str) -> str:
    import pandas as pd

    df = pd.read_csv(csv_path)
    return "Size Chart:\n" + df.to_string(index=False)


def embed_table(csv_path: str) -> np.ndarray:
    return embed_text(table_to_text(csv_path), task_type="RETRIEVAL_DOCUMENT")


# ── CLIP (optional dependency) ────────────────────────────
def clip_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor

        _clip_model = CLIPModel.from_pretrained(clip_model_name)
        _clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        _clip_model.eval()
    return _clip_model, _clip_processor


def _as_tensor(out, *attrs):
    """transformers versions differ: feature getters return a tensor or an output object."""
    if hasattr(out, "norm"):
        return out
    for attr in attrs:
        if hasattr(out, attr):
            return getattr(out, attr)
    raise TypeError(f"Unexpected CLIP output type: {type(out)}")


def embed_image(image_path: str) -> np.ndarray:
    import torch
    from PIL import Image

    model, processor = _get_clip()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        feat = _as_tensor(model.get_image_features(**inputs), "image_embeds", "pooler_output")
    return normalize(feat[0].cpu().numpy())


def embed_clip_text(text: str) -> np.ndarray:
    """Embed text into the CLIP *image* space so text queries can search the image index."""
    import torch

    model, processor = _get_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        feat = _as_tensor(model.get_text_features(**inputs), "text_embeds", "pooler_output")
    return normalize(feat[0].cpu().numpy())


# ── Embedder objects (dependency-injection friendly) ──────
class GeminiEmbedder:
    """Default text embedder. Tests substitute their own object with the same two methods."""

    def embed_query(self, text: str) -> np.ndarray:
        return embed_text(text, "RETRIEVAL_QUERY")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return embed_texts_batch(texts, "RETRIEVAL_DOCUMENT")


class ClipEmbedder:
    def embed_text(self, text: str) -> np.ndarray:
        return embed_clip_text(text)

    def embed_image(self, path: str) -> np.ndarray:
        return embed_image(path)


@lru_cache(maxsize=1)
def default_embedder() -> GeminiEmbedder:
    return GeminiEmbedder()
