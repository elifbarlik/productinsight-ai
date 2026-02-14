import pandas as pd
import torch
from PIL import Image
from google import genai
from src.config import gemini_api_key, embedding_model_name,clip_model_name
from transformers import CLIPProcessor, CLIPModel
import numpy as np

client = genai.Client(api_key=gemini_api_key)

clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

def embed_text(text:str)->np.ndarray:
    response = client.models.embed_content(
        model=embedding_model_name,
        contents=text
    )
    embedding = response.embeddings[0].values
    return np.array(embedding, dtype='float32')

def embed_texts_batch(texts:list)->np.ndarray:
    response = client.models.embed_content(
        model=embedding_model_name,
        contents=texts
    )
    embedding = [item.embedding for item in response.embeddings]
    return np.array(embedding, dtype='float32')

def embed_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        pooled = vision_out.pooler_output              # [1, hidden]
        feat = clip_model.visual_projection(pooled)    # [1, 512]

    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].cpu().numpy().astype("float32")

def embed_table(csv_path: str) -> np.ndarray:

    df = pd.read_csv(csv_path)
    table_text = "Size Chart: \n"
    table_text += df.to_string(index=False)

    return embed_text(table_text)

if __name__== "__main__":
    print("Testing embeddings...")

    # Text
    text_emb = embed_text("Black leather jacket")
    print(f"✓ Text embedding: {len(text_emb)} dimensions")

    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
    img_path = BASE_DIR / "src" / "data" / "images" / "1.jpg"
    print("IMG PATH:", img_path)
    print("EXISTS:", img_path.exists())

    try:
        img_emb = embed_image(str(img_path))  # <-- str()
        print(f"✓ Image embedding: {len(img_emb)} dimensions")
    except Exception as e:
        print(f"⚠ Image test failed: {e}")
