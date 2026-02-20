from src.config import faiss_text_path, faiss_image_path, faiss_table_path, metadata_path
from src.embedding_utils import embed_text, embed_image
from src.vector_store import MultimodalVectorStore

print("RETRIEVAL TEST\n")
print("-"*50)

print("Vector store yukleniyor..")
vector_store = MultimodalVectorStore.load(
    faiss_text_path,
    faiss_image_path,
    faiss_table_path,
    metadata_path
)
print(f"{len(vector_store)} urun yuklendi")

# Test 1: Text Query
print("\n\n Text Query Test")
print("Query: 'black jacket'")
query_text = 'black jacket'
query_emb = embed_text(query_text)
results = vector_store.search_text(query_emb, k=3)
print("\nSonuclar: ")
for idx, distance in results:
    product = vector_store.metadata[idx]
    print(f" - {product.name} (${product.price}) | Distance: {distance: .4f}")

# Test 2: Image Query
print("\n\n Image Query Test")
print("Query: src/data/images/1.jpg")
query_img = embed_image("src/data/images/1.jpg")
results = vector_store.search_image(query_img, k=3)
print("\nSonuclar: ")
for idx, distance in results:
    product = vector_store.metadata[idx]
    print(f" - {product.name} (${product.price}) | Distance: {distance: .4f}")

# Test 3: Combined Search
print("\n️\n Combined Query Test")
print("Query: 'leather shoes'")
query_text = 'leather shoes'
query_text_emb = embed_text(query_text)
query_img_emb = embed_image("src/data/images/1.jpg")
results = vector_store.search_combined(query_text_emb,query_img_emb, k=3)
print("\nSonuclar: ")
for product in results:
    print(f" - {product.name} (${product.price})")

print("-"*50)
print("TUM TESTLER BASARILI")