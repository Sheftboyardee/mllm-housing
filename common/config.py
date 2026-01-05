from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

def get_env_var(key: str, default: str = "") -> str:
    """
    Get environment variable, with support for Streamlit secrets.
    Tries os.getenv first, then st.secrets if available.
    """
    # Environment variable if local
    value = os.getenv(key, "")
    if value:
        return value
    
    # Try Streamlit secrets
    StreamlitSecretNotFoundError = None
    try:
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:
        class StreamlitSecretNotFoundError(Exception):
            pass
    
    try:
        import streamlit as st
        
        try:
            if hasattr(st, 'secrets'):
                try:
                    if hasattr(st.secrets, 'get'):
                        secret_value = st.secrets.get(key)
                        if secret_value:
                            return str(secret_value)
                except Exception:
                    pass
                
                try:
                    secret_value = st.secrets[key]
                    if secret_value:
                        return str(secret_value)
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
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
