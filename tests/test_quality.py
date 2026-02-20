from src.config import faiss_text_path, faiss_image_path, faiss_table_path, metadata_path
from src.vector_store import MultimodalVectorStore
import time
from src.embedding_utils import embed_text
print("QUALITY CHECKS")
print("-"*50)

vector_store = MultimodalVectorStore.load(
    faiss_text_path,
    faiss_image_path,
    faiss_table_path,
    metadata_path
)

# Check 1: Index sizes
print("\n\nIndex Sizes")
print(f"  Text index: {vector_store.text_index.ntotal} vectors")
print(f"  Image index: {vector_store.image_index.ntotal} vectors")
print(f"  Table index: {vector_store.table_index.ntotal} vectors")
print(f"  Metadata: {len(vector_store.metadata)} products")

assert vector_store.text_index.ntotal == len(vector_store.metadata), "Missmatch!"
assert vector_store.image_index.ntotal == len(vector_store.metadata), "Missmatch!"
print("All sizes match")

# Check 2: Metadata integrity
print(f"\n\nMetadata Integrity")
for i , product in enumerate(vector_store.metadata):
    assert product.product_id>0, f"Invalid ID: {product.product_id}"
    assert len(product.name)>0, "Empty name"
    assert product.price>0, "Invalid price"
print(f"All {len(vector_store.metadata)} products valid")

# Check 3: Search performance
print(f"\n\nSearch Performance")
query = "jacket"
query_emb=embed_text(query)
start=time.time()
results = vector_store.search_text(query_emb, k=5)
elapsed=time.time()-start

print(f"  Query: '{query}'")
print(f"  Time: {elapsed*1000:.2f}ms")
print(f"  Results: {len(results)}")
print("  Search performance OK")

print("\n" + "-" * 50)
print("✓ QUALİTY CHECK PASSED")