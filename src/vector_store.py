import faiss
import numpy as np
import json
from typing import List, Tuple
from pathlib import Path
from src.models import Product
import os

class MultimodalVectorStore:
    def __init__(self, text_dim:int=3072, image_dim:int=512, table_dim:int=3072):
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.table_dim = table_dim

        self.text_index = faiss.IndexFlatL2(text_dim)
        self.image_index = faiss.IndexFlatL2(image_dim)
        self.table_index = faiss.IndexFlatL2(table_dim)

        self.metadata:List[Product] = []
        self.id_map={}

    def add_product(self, product:Product, text_embedding:np.ndarray, image_embedding:np.ndarray, table_embedding:np.ndarray):
        idx = len(self.metadata)

        self.text_index.add(text_embedding.reshape(1, -1).astype('float32'))
        self.image_index.add(image_embedding.reshape(1, -1).astype('float32'))
        self.table_index.add(table_embedding.reshape(1, -1).astype('float32'))

        self.metadata.append(product)
        self.id_map[product.product_id] = idx

    def search_text(self, query_embedding:np.ndarray, k:int=5) -> List[Tuple[int, float]]:
        distances, indices = self.text_index.search(query_embedding.reshape(1,-1).astype('float32'), k)

        return list(zip(indices[0], distances[0]))

    def search_image(self, query_embedding:np.ndarray, k:int=5) -> List[Tuple[int, float]]:
        distances, indices = self.image_index.search(query_embedding.reshape(1,-1).astype('float32'), k)
        return list(zip(indices[0], distances[0]))

    def search_combined(self, text_emb:np.ndarray,image_emb:np.ndarray, k:int=5) -> List[Product]:
        text_results=self.search_text(text_emb,k)
        image_results=self.search_image(image_emb,k)

        scores={}
        for idx, dist in text_results:
            scores[idx]=scores.get(idx,0)+(1-min(dist,1)) *0.6
        for idx, dist in image_results:
            scores[idx]=scores.get(idx,0)+(1-min(dist,1)) *0.4

        sorted_indices=sorted(scores.items(), key=lambda x:x[1], reverse=True)[:k]
        return [self.metadata[idx] for idx , score in sorted_indices]

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
        store =cls()

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