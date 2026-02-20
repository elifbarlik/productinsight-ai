import faiss
import numpy as np
import json
from typing import List, Tuple, Optional
from pathlib import Path
from src.models import Product
from src.config import text_embedding_dim, image_embedding_dim, table_embedding_dim
import os

class MultimodalVectorStore:
    def __init__(self, text_dim:int=None, image_dim:int=None, table_dim:int=None):
        self.text_dim = text_dim if text_dim is not None else text_embedding_dim
        self.image_dim = image_dim if image_dim is not None else image_embedding_dim
        self.table_dim = table_dim if table_dim is not None else table_embedding_dim

        self.text_index = faiss.IndexFlatL2(self.text_dim)
        self.image_index = faiss.IndexFlatL2(self.image_dim)
        self.table_index = faiss.IndexFlatL2(self.table_dim)

        self.metadata:List[Product] = []
        self.id_map={}

    def add_product(self, product:Product, text_embedding:np.ndarray, image_embedding:np.ndarray, table_embedding:np.ndarray):
        idx = len(self.metadata)

        self.text_index.add(text_embedding.reshape(1, -1).astype('float32'))
        self.image_index.add(image_embedding.reshape(1, -1).astype('float32'))
        self.table_index.add(table_embedding.reshape(1, -1).astype('float32'))

        self.metadata.append(product)
        self.id_map[product.product_id] = idx

    def search_text(self, query_embedding: np.ndarray, k: int = 5, threshold: float = None) -> List[Tuple[int, float]]:
        distances, indices = self.text_index.search(query_embedding.reshape(1, -1).astype('float32'), k)

        results = list(zip(indices[0].tolist(), distances[0].tolist()))


        if threshold is not None:
            results = [(idx, dist) for idx, dist in results if dist < threshold]

        return results


    def search_image(self, query_embedding: np.ndarray, k: int = 5, threshold: float = None) -> List[Tuple[int, float]]:
        distances, indices = self.image_index.search(query_embedding.reshape(1, -1).astype('float32'), k)

        results = list(zip(indices[0].tolist(), distances[0].tolist()))


        if threshold is not None:
            results = [(idx, dist) for idx, dist in results if dist < threshold]

        return results


    def search_combined(
        self,
        text_embedding: np.ndarray,
        image_embedding: Optional[np.ndarray] = None,
        k: int = 5,
        threshold: float = None,
        text_weight: float = 0.7,
        image_weight: float = 0.3
    ) -> List[Tuple[Product, float]]:
        text_results = self.search_text(text_embedding, k=k * 2)

        scores = {}
        for idx, dist in text_results:
            similarity = 1.0 / (1.0 + dist)
            scores[idx] = scores.get(idx, 0) + similarity * text_weight

        if image_embedding is not None:
            image_results = self.search_image(image_embedding, k=k * 2)
            for idx, dist in image_results:
                similarity = 1.0 / (1.0 + dist)
                scores[idx] = scores.get(idx, 0) + similarity * image_weight

        if threshold is not None:
            scores = {idx: score for idx, score in scores.items() if score >= threshold}

        sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        return [(self.metadata[idx], score) for idx, score in sorted_indices if idx < len(self.metadata)]

    def save(self, text_path:str, image_path:str,table_path:str, metadata_path:str ):
        os.makedirs(os.path.dirname(text_path), exist_ok=True)
        faiss.write_index(self.text_index, text_path)
        faiss.write_index(self.image_index, image_path)
        faiss.write_index(self.table_index, table_path)

        metadata_list = [p.to_dict() for p in self.metadata]
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)

    @classmethod
    def load(cls, text_path:str, image_path:str, table_path:str, metadata_path:str):
        store = cls.__new__(cls)
        store.metadata = []
        store.id_map = {}

        store.text_index = faiss.read_index(text_path)
        store.image_index = faiss.read_index(image_path)
        store.table_index = faiss.read_index(table_path)

        with open(metadata_path, 'r') as f:
            metadata_list = json.load(f)
        for i, metadata in enumerate(metadata_list):
            product = Product(
                product_id=metadata['product_id'],
                name=metadata['name'],
                description=metadata['description'],
                price=metadata['price'],
                color=metadata['color'],
                category=metadata.get('category', ''),
                gender=metadata.get('gender', ''),
                image_path=Path(metadata["image_path"]) if metadata.get("image_path") else None,
                size_chart_path=Path(metadata["size_chart_path"]) if metadata.get("size_chart_path") else None,
            )
            store.metadata.append(product)
            store.id_map[product.product_id] = i

        return store

    def __len__(self):
        return len(self.metadata)


if __name__=="__main__":
    from src.models import Product
    store = MultimodalVectorStore()
    print(f"Vector store created: {store.__dict__}")