from google import genai
from google.genai import types

from src.config import gemini_api_key, faiss_text_path, faiss_image_path, faiss_table_path, metadata_path, llm_model, relevance_threshold, retrieval_k, llm_temperature
from src.embedding_utils import embed_text
from src.vector_store import MultimodalVectorStore

RELEVANCE_THRESHOLD = relevance_threshold


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

        over_fetch_k = k * 2

        results = self.vector_store.search_text(query_embedding, k=over_fetch_k)


        filtered_results = [
            (idx, dist) for idx, dist in results
            if dist <= RELEVANCE_THRESHOLD
        ]


        filtered_results = filtered_results[:k]

        retrieved_products = [self.vector_store.metadata[idx] for idx, _ in filtered_results]
        distances = [dist for _, dist in filtered_results]
        return retrieved_products, distances

    def format_context(self, products):
        context = "## Available Products\n\n"
        for i, product in enumerate(products, 1):
            context += (
                f"### {i}. {product.name}\n"
                f"- **Price:** ${product.price}\n"
                f"- **Color:** {product.color}\n"
                f"- **Category:** {product.category}\n"
                f"- **Gender:** {product.gender}\n"
                f"- **Description:** {product.description}\n"
                f"- **Image:** {product.image_path}\n\n"
            )
        return context

    def generate_response(self, user_query: str, products):
        context = self.format_context(products)

        system_prompt = (
            "You are a helpful e-commerce customer support assistant for an Indian Ethnic Wear store.\n"
            "You have access to a product catalog and should help customers find products.\n\n"
            "Instructions:\n"
            "1. ONLY recommend products that are listed in the provided context below.\n"
            "2. NEVER invent or fabricate products that are not in the context.\n"
            "3. Strictly apply color filters: if the customer asks for 'blue kurta', "
            "only recommend kurtas that are blue or blue-toned (navy, teal, etc.).\n"
            "4. Strictly apply category and gender filters: if the customer asks for 'women kurta', "
            "only recommend products where Category is 'Kurta' (or 'Kurta Set') and Gender is 'Women'.\n"
            "5. If the customer specifies a price range (e.g. 'under ₹3000'), "
            "only recommend products within that range.\n"
            "6. Provide specific product names, prices, and brief descriptions.\n"
            "7. Be friendly and helpful.\n"
            "8. If no product matches the customer's criteria exactly, "
            "say so clearly and suggest the closest alternatives from the catalog."
        )

        user_message = (
            f"{context}\n"
            f"Customer Question: {user_query}\n\n"
            f"Please provide a helpful response based on the available products."
        )

        response = self.client.models.generate_content(
            model=llm_model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=llm_temperature
            )
        )

        return response.text

    def query(self, user_query: str):
        products, distances = self.retrieve_products(user_query, k=retrieval_k)
        if not products:
            return "Sorry, no products match your query", [], []

        response = self.generate_response(user_query, products)
        return response, products, distances