from pinecone import Pinecone
from .config import settings

# Lazy init to avoid errors at import time
_pc = None
_index = None

def get_pinecone_client():
    """Get or create Pinecone client instance."""
    global _pc
    if _pc is None:
        if not settings.pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY is not set. "
                "Please set it in your environment variables or Streamlit secrets."
            )
        _pc = Pinecone(api_key=settings.pinecone_api_key)
    return _pc

def get_index():
    """Get or create Pinecone index instance."""
    global _index
    if _index is None:
        pc = get_pinecone_client()
        if not settings.pinecone_index:
            raise ValueError(
                "PINECONE_INDEX is not set. "
                "Please set it in your environment variables or Streamlit secrets."
            )
        _index = pc.Index(settings.pinecone_index)
    return _index


try:
    if settings.pinecone_api_key:
        pc = get_pinecone_client()
        index = get_index()
    else:
        pc = None
        index = None
except Exception:
    pc = None
    index = None
