from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

def get_env_var(key: str, default: str = "") -> str:
    """
    Get environment variable, with support for Streamlit secrets.
    Tries os.getenv first, then st.secrets if available.
    """
    # Try standard environment variable first
    value = os.getenv(key, "")
    if value:
        return value
    
    # Try Streamlit secrets if available (for Streamlit Cloud)
    try:
        import streamlit as st
        # Access secrets - they can be top-level or nested
        if hasattr(st, 'secrets'):
            # Try top-level key first
            if hasattr(st.secrets, 'get'):
                secret_value = st.secrets.get(key)
                if secret_value:
                    return str(secret_value)
            # Try direct access
            try:
                if key in st.secrets:
                    return str(st.secrets[key])
            except (TypeError, KeyError):
                pass
    except (ImportError, RuntimeError, AttributeError):
        # Streamlit not available or not in Streamlit context
        pass
    
    return default

@dataclass
class Settings:
    pinecone_api_key: str = get_env_var("PINECONE_API_KEY", "")
    pinecone_index: str = get_env_var("PINECONE_INDEX", "house-embeddings")
    model_name: str = get_env_var("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    default_top_k: int = int(get_env_var("DEFAULT_TOP_K", "10"))
    dataset_path: str = get_env_var("DATASET_PATH", "./data/Houses-dataset")

settings = Settings()
