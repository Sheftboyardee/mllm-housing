from typing import List, Dict, Any, Optional
from .embeddings import embed_texts
from .pinecone_client import get_index
from .config import settings


def search_houses(query: str, top_k: int | None = None, filters: dict | None = None) -> List[Dict[str, Any]]:
    """    
    Args:
        query: Search query 
        top_k: Number of results to return 
        filters: Optional hard filters
    
    Returns:
        List of match dictionaries containing:
        - id: House ID
        - score: Similarity score
        - metadata: House metadata (bedrooms, bathrooms, area, zipcode, price, description, etc.)
    """
    if top_k is None:
        top_k = settings.default_top_k
    
    query_vec = embed_texts([query])[0]
    query_params = {
        "vector": query_vec.tolist(),
        "top_k": top_k,
        "include_metadata": True,
    }
    
    if filters is not None:
        query_params["filter"] = filters
    
    index = get_index()
    response = index.query(**query_params)
    
    return response.matches

