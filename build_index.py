from src.data_pipeline import DataPipeline
from src.config import faiss_text_path, faiss_image_path, faiss_table_path, metadata_path, product_csv_path

print("\n\nKATALOG INDEX BUILD")
print("-"* 50)

pipeline = DataPipeline(product_csv_path)
vector_store = pipeline.build_index()

vector_store.save(faiss_text_path, faiss_image_path, faiss_table_path, metadata_path)

print(f"\nTAMAMLANDI!")
print(f"  - {len(vector_store)} ürün")
print(f"  - {faiss_text_path}")
print(f"  - {faiss_image_path}")
print(f"  - {faiss_table_path}")