from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class Settings:
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "house-embeddings")
    model_name: str = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", 10))
    dataset_path: str = os.getenv("DATASET_PATH", "./data/Houses-dataset")

settings = Settings()
