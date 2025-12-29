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
    # Import StreamlitSecretNotFoundError first so we can catch it properly
    StreamlitSecretNotFoundError = None
    try:
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:
        # For older Streamlit versions, create a dummy class that matches the name
        class StreamlitSecretNotFoundError(Exception):
            pass
    
    try:
        import streamlit as st
        
        # Access secrets - catch ALL exceptions since we want to fall back to defaults
        # StreamlitSecretNotFoundError is raised when secrets file doesn't exist
        try:
            if hasattr(st, 'secrets'):
                # Try using get method
                try:
                    if hasattr(st.secrets, 'get'):
                        secret_value = st.secrets.get(key)
                        if secret_value:
                            return str(secret_value)
                except Exception:
                    # Catch all exceptions (including StreamlitSecretNotFoundError)
                    pass
                
                # Try direct access
                try:
                    secret_value = st.secrets[key]
                    if secret_value:
                        return str(secret_value)
                except Exception:
                    # Catch all exceptions (including StreamlitSecretNotFoundError, KeyError, etc.)
                    pass
        except Exception:
            # Catch any exception raised when accessing st.secrets
            # This includes StreamlitSecretNotFoundError when secrets file doesn't exist
            pass
    except Exception:
        # Catch all exceptions at the outermost level
        # This includes ImportError, RuntimeError, AttributeError, and StreamlitSecretNotFoundError
        # We want to fall back to defaults in all these cases
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
