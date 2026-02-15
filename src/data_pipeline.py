from src.catalog import ProductCatalog
from src.embedding_utils import embed_text, embed_image, embed_table
from src.vector_store import MultimodalVectorStore
from src.config import vector_dim

class DataPipeline:
    """Katalog → Embedding → Vector Store"""
    def __init__(self, csv_path:str):
        self.catalog = ProductCatalog(csv_path)
        self.vector_store = MultimodalVectorStore(embedding_dim=vector_dim)

    def prepare_product(self, product):
        text = f"{product.name} {product.description} {product.color}"
        text_emb = embed_text(text)
        image_emb = embed_image(str(product.image_path))
        table_emb = embed_table(str(product.size_chart_path))

        return text, text_emb, image_emb, table_emb

    def build_index(self):
        print(f"{len(self.catalog)} ürün indexleniyor...")

        for i, product in enumerate(self.catalog.products):
            print(f"[{i+1}/{len(self.catalog)}] {product.name}...")

            try:
                text_emb, image_emb, table_emb = self.prepare_product(product)
                self.vector_store.add_product(product, text_emb, image_emb, table_emb)
            except Exception as e:
                print(f"Hata {product.name} : {e}")
                continue
        print(f"{len(self.vector_store)} urun indexlendi")
        return self.vector_store

