from google import genai
from google.genai import types

from src.config import gemini_api_key, faiss_text_path, faiss_image_path, faiss_table_path, metadata_path, llm_model
from src.embedding_utils import embed_text
from src.vector_store import MultimodalVectorStore


class ProductRAG:
    def __init__(self):
        self.client = genai.Client(api_key=gemini_api_key)

        self.vector_store = MultimodalVectorStore.load(
            faiss_text_path,
            faiss_image_path,
            faiss_table_path,
            metadata_path
        )

    def retrieve_products(self, query: str, k:int=5):
        query_embedding = embed_text(query)
        results = self.vector_store.search_text(query_embedding, k=k)
        retrieved_products = [self.vector_store.metadata[idx] for idx, _ in  results]
        return retrieved_products

    def format_context(self, products):
        context = "##Available Products\n\n"
        for i, product in enumerate(products, 1):
            context +=f"""### {i}. {product.name}
                        - **Price:** ${product.price}
                        - **Color:** {product.color}
                        - **Description:** {product.description}
                        - **Image:** {product.image_path}"""
        return context

    def generate_response(self, user_query:str, products):
        context=self.format_context(products)
        system_prompt = """You are a helpful e-commerce customer support assistant.
        You have access to a product catalog and should help customers find products they're looking for.

        Instructions:
        1. Always refer to the available products in the catalog
        2. Provide specific product names and prices
        3. Be friendly and helpful
        4. If a product isn't available, suggest alternatives from the catalog"""

        user_message = f"""{context}
                        Customer Question: {user_query}
                        Please provide a helpful response based on the available products."""


        response = self.client.models.generate_content(
            model=llm_model,
            contents=system_prompt + "\n\n" + user_message,
            config=types.GenerateContentConfig(
                temperature=0.7
            )
        )

        return response.text

    def query(self, user_query:str):
        products = self.retrieve_products(user_query, k=5)
        if not products:
            return "Sorry, no products match your query", []

        response = self.generate_response(user_query, products)
        return response, products