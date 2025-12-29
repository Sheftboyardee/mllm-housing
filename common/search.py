"""
Shared search helper for querying house embeddings in Pinecone.
"""

from typing import List, Dict, Any, Optional
from .embeddings import embed_texts
from .pinecone_client import index
from .config import settings


def search_houses(
    query: str,
    top_k: int | None = None,
    filters: dict | None = None
) -> List[Dict[str, Any]]:
    """
    Search for houses using a natural language query.
    
    Args:
        query: Natural language search query (e.g., "3 bedroom house with large kitchen")
        top_k: Number of results to return (defaults to settings.default_top_k)
        filters: Optional Pinecone metadata filters (e.g., {"bedrooms": {"$eq": 3}})
    
    Returns:
        List of match dictionaries containing:
        - id: House ID
        - score: Similarity score
        - metadata: House metadata (bedrooms, bathrooms, area, zipcode, price, description, etc.)
    
    Example:
        >>> results = search_houses("modern house with 4 bedrooms")
        >>> for match in results:
        ...     print(f"House {match['id']}: {match['score']}")
        ...     print(f"  {match['metadata']['description']}")
    """
    if top_k is None:
        top_k = settings.default_top_k
    
    # Embed the query
    query_vec = embed_texts([query])[0]
    
    # Prepare query parameters
    query_params = {
        "vector": query_vec.tolist(),
        "top_k": top_k,
        "include_metadata": True,
    }
    
    # Add filters if provided
    if filters is not None:
        query_params["filter"] = filters
    
    # Query Pinecone
    response = index.query(**query_params)
    
    return response.matches

