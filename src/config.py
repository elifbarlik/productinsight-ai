from dotenv import load_dotenv
import os

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
llm_model = os.getenv("LLM_MODEL")
text_embedding_model = os.getenv("TEXT_EMBEDDING_DIM")
image_embedding_model = os.getenv("IMAGE_EMBEDDING_DIM")
faiss_text_path= os.getenv("FAISS_TEXT_PATH")
faiss_image_path= os.getenv("FAISS_IMAGE_PATH")
faiss_table_path= os.getenv("FAISS_TABLE_PATH")
metadata_path= os.getenv("METADATA_PATH")
clip_model_name= os.getenv("CLIP_MODEL_NAME")
table_embedding_dim=os.getenv("TABLE_EMBEDDING_DIM")
vector_dim=os.getenv("VECTOR_DIM")