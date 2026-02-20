from dotenv import load_dotenv
import os

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
llm_model = os.getenv("LLM_MODEL")

# FIX: Değişken adları düzeltildi — model değil, dim
text_embedding_dim = int(os.getenv("TEXT_EMBEDDING_DIM", "768"))
image_embedding_dim = int(os.getenv("IMAGE_EMBEDDING_DIM", "512"))

faiss_text_path= os.getenv("FAISS_TEXT_PATH")
faiss_image_path= os.getenv("FAISS_IMAGE_PATH")
faiss_table_path= os.getenv("FAISS_TABLE_PATH")
metadata_path= os.getenv("METADATA_PATH")
clip_model_name= os.getenv("CLIP_MODEL_NAME")
table_embedding_dim=os.getenv("TABLE_EMBEDDING_DIM")
vector_dim=int(os.getenv("VECTOR_DIM", "512"))

# FIX: Eksik config değerleri eklendi
relevance_threshold = float(os.getenv("RETRIEVAL_THRESHOLD", "0.75"))
retrieval_k = int(os.getenv("RETRIEVAL_K", "5"))
llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# DEPRECATED: Eski adlar — ileride kaldırılacak
text_embedding_model = text_embedding_dim
image_embedding_model = image_embedding_dim