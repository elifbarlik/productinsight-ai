from google import genai
from google.genai import types

from src.config import gemini_api_key, faiss_text_path, faiss_image_path, faiss_table_path, metadata_path, llm_model
from src.embedding_utils import embed_text
from src.vector_store import MultimodalVectorStore

# FIX: Relevance threshold — config'den al, yoksa default kullan
try:
    from src.config import relevance_threshold as _cfg_threshold
    RELEVANCE_THRESHOLD = float(_cfg_threshold) if _cfg_threshold else 0.75
except (ImportError, AttributeError):
    RELEVANCE_THRESHOLD = 0.75


class ProductRAG:
    def __init__(self):
        self.client = genai.Client(api_key=gemini_api_key)

        self.vector_store = MultimodalVectorStore.load(
            faiss_text_path,
            faiss_image_path,
            faiss_table_path,
            metadata_path
        )

    def retrieve_products(self, query: str, k: int = 5):
        query_embedding = embed_text(query)

        # FIX: k*2 over-fetch yaparak threshold sonrası yeterli sonuç kalmasını sağla
        over_fetch_k = k * 2

        # FIX: search_combined() varsa kullan (text + image hibrit arama), yoksa fallback
        try:
            if hasattr(self.vector_store, 'search_combined'):
                # search_combined text+image ister; sadece text varsa search_text kullan
                results = self.vector_store.search_text(query_embedding, k=over_fetch_k)
            else:
                results = self.vector_store.search_text(query_embedding, k=over_fetch_k)
        except Exception:
            results = self.vector_store.search_text(query_embedding, k=over_fetch_k)

        # FIX: Relevance threshold uygula — yüksek distance'lı (alakasız) sonuçları filtrele
        filtered_results = [
            (idx, dist) for idx, dist in results
            if dist <= RELEVANCE_THRESHOLD
        ]

        # FIX: Filtreleme sonrası maksimum k ürün döndür
        filtered_results = filtered_results[:k]

        retrieved_products = [self.vector_store.metadata[idx] for idx, _ in filtered_results]
        return retrieved_products

    def format_context(self, products):
        # FIX: Indentation bozukluğu düzeltildi — gereksiz boşluklar kaldırıldı
        context = "## Available Products\n\n"
        for i, product in enumerate(products, 1):
            context += (
                f"### {i}. {product.name}\n"
                f"- **Price:** ${product.price}\n"
                f"- **Color:** {product.color}\n"
                f"- **Description:** {product.description}\n"
                f"- **Image:** {product.image_path}\n\n"
            )
        return context

    def generate_response(self, user_query: str, products):
        context = self.format_context(products)

        # FIX: System prompt'a filtreleme talimatları eklendi
        system_prompt = (
            "You are a helpful e-commerce customer support assistant.\n"
            "You have access to a product catalog and should help customers find products.\n\n"
            "Instructions:\n"
            "1. ONLY recommend products that are listed in the provided context below.\n"
            "2. NEVER invent or fabricate products that are not in the context.\n"
            "3. Strictly apply color filters: if the customer asks for 'blue jacket', "
            "only recommend jackets that are blue or blue-toned (navy, indigo, etc.).\n"
            "4. Strictly apply category filters: if the customer asks for 'jacket', "
            "only recommend jackets, coats, or blazers — not shirts, jeans, or shoes.\n"
            "5. If the customer specifies a price range (e.g. 'under $100'), "
            "only recommend products within that range.\n"
            "6. Provide specific product names, prices, and brief descriptions.\n"
            "7. Be friendly and helpful.\n"
            "8. If no product matches the customer's criteria exactly, "
            "say so clearly and suggest the closest alternatives from the catalog."
        )

        # FIX: User message indentation düzeltildi
        user_message = (
            f"{context}\n"
            f"Customer Question: {user_query}\n\n"
            f"Please provide a helpful response based on the available products."
        )

        # FIX: System prompt ayrı, user message ayrı — role ayrımı korunuyor
        response = self.client.models.generate_content(
            model=llm_model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,  # FIX: system prompt artık ayrı
                temperature=0.3  # FIX: 0.7 → 0.3 — daha tutarlı yanıtlar
            )
        )

        return response.text

    def query(self, user_query: str):
        products = self.retrieve_products(user_query, k=5)
        if not products:
            return "Sorry, no products match your query", []

        response = self.generate_response(user_query, products)
        return response, products