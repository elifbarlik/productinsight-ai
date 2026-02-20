import pandas as pd
import torch
from PIL import Image
from google import genai
from src.config import gemini_api_key, embedding_model_name, clip_model_name
from transformers import CLIPProcessor, CLIPModel
import numpy as np

client = genai.Client(api_key=gemini_api_key)

_clip_model = None
_clip_processor = None

def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained(clip_model_name)
        _clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    return _clip_model, _clip_processor

def embed_text(text: str) -> np.ndarray:
    try:
        response = client.models.embed_content(
            model=embedding_model_name,
            contents=text
        )
        embedding = response.embeddings[0].values
        return np.array(embedding, dtype='float32')
    except Exception as e:
        raise ValueError(f"Text embedding failed: {e}") from e

def embed_texts_batch(texts: list) -> np.ndarray:
    if not texts:
        return np.array([], dtype='float32')

    try:
        response = client.models.embed_content(
            model=embedding_model_name,
            contents=texts
        )
        embedding = [item.values for item in response.embeddings]
        return np.array(embedding, dtype='float32')
    except Exception as e:
        raise ValueError(f"Batch embedding failed: {e}") from e

def embed_image(image_path: str) -> np.ndarray:
    clip_model, clip_processor = _get_clip()
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        pooled = vision_out.pooler_output
        feat = clip_model.visual_projection(pooled)

    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].cpu().numpy().astype("float32")

def embed_table(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    table_text = "Size Chart: \n"
    table_text += df.to_string(index=False)
    return embed_text(table_text)

if __name__ == "__main__":
    text_emb = embed_text("Black leather jacket")

    try:
        img_emb = embed_image("data/images/image_1.jpg")
    except Exception as e:
        pass

