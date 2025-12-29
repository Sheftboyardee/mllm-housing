from sentence_transformers import SentenceTransformer
from functools import lru_cache
from .config import settings

@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(settings.model_name)

def embed_texts(texts, show_progress_bar=False):
    """
    Generate embeddings for text(s).
    
    Args:
        texts: Single string or list of strings
        show_progress_bar: Whether to show progress bar (useful for batch processing)
    
    Returns:
        numpy array of embeddings (single embedding if single text, array if list)
    """
    model = get_embedding_model()
    # Convert single string to list
    if isinstance(texts, str):
        texts = [texts]
    return model.encode(texts, batch_size=64, show_progress_bar=show_progress_bar)
