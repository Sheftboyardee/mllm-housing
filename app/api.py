"""
FastAPI backend for semantic house search.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import sys
import json
from pathlib import Path

# Add parent directory to path to import common modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.search import search_houses
from common.config import settings

# Initialize FastAPI app
app = FastAPI(
    title="MLLM House Search API",
    description="Semantic search API for finding houses using natural language queries",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup."""
    import logging
    logger = logging.getLogger("uvicorn")
    
    # Check configuration
    if not settings.pinecone_api_key:
        logger.error("⚠️ PINECONE_API_KEY is not set!")
    if not settings.pinecone_index:
        logger.error("⚠️ PINECONE_INDEX is not set!")
    
    # Test Pinecone connection
    try:
        from common.pinecone_client import get_index
        index = get_index()
        # Try a simple describe_index_stats to verify connection
        stats = index.describe_index_stats()
        logger.info(f"✅ Connected to Pinecone index: {settings.pinecone_index}")
        logger.info(f"   Index stats: {stats}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Pinecone: {str(e)}")
        logger.error("    Please check your PINECONE_API_KEY and PINECONE_INDEX")
    
    # Test embedding model
    try:
        from common.embeddings import get_embedding_model
        model = get_embedding_model()
        logger.info(f"✅ Embedding model loaded: {settings.model_name}")
    except Exception as e:
        logger.error(f"❌ Failed to load embedding model: {str(e)}")
        logger.error(f"   Model: {settings.model_name}")


# Request/Response Models
class SearchFilters(BaseModel):
    """Optional filters for house search."""
    min_bedrooms: Optional[int] = Field(None, ge=0, description="Minimum number of bedrooms")
    max_bedrooms: Optional[int] = Field(None, ge=0, description="Maximum number of bedrooms")
    min_bathrooms: Optional[float] = Field(None, ge=0, description="Minimum number of bathrooms")
    max_bathrooms: Optional[float] = Field(None, ge=0, description="Maximum number of bathrooms")
    min_price: Optional[float] = Field(None, ge=0, description="Minimum price")
    max_price: Optional[float] = Field(None, ge=0, description="Maximum price")
    min_area: Optional[int] = Field(None, ge=0, description="Minimum area in sqft")
    max_area: Optional[int] = Field(None, ge=0, description="Maximum area in sqft")
    zipcode: Optional[str] = Field(None, description="Zipcode (exact match)")


class HouseMetadata(BaseModel):
    """House metadata from search results."""
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    area: Optional[int] = None
    price: Optional[float] = None
    zipcode: Optional[str] = None
    description: Optional[str] = None
    images: Optional[Dict[str, str]] = None


class HouseMatch(BaseModel):
    """A single house match from search results."""
    id: str = Field(..., description="House ID")
    score: float = Field(..., description="Similarity score (0-1)")
    metadata: HouseMetadata = Field(..., description="House metadata")


class SearchRequest(BaseModel):
    """Request model for house search."""
    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: Optional[int] = Field(10, ge=1, le=50, description="Number of results to return")
    filters: Optional[SearchFilters] = Field(None, description="Optional filters")


class SearchResponse(BaseModel):
    """Response model for house search."""
    query: str = Field(..., description="The search query")
    results_count: int = Field(..., description="Number of results returned")
    results: List[HouseMatch] = Field(..., description="List of matching houses")


def build_pinecone_filters(filters: Optional[SearchFilters]) -> Optional[Dict[str, Any]]:
    """
    Convert SearchFilters to Pinecone filter format.
    
    Args:
        filters: Optional SearchFilters object
        
    Returns:
        Pinecone filter dictionary or None
    """
    if filters is None:
        return None
    
    pinecone_filters = {}
    
    # Bedroom filters
    if filters.min_bedrooms is not None or filters.max_bedrooms is not None:
        bedroom_filter = {}
        if filters.min_bedrooms is not None:
            bedroom_filter["$gte"] = filters.min_bedrooms
        if filters.max_bedrooms is not None:
            bedroom_filter["$lte"] = filters.max_bedrooms
        if bedroom_filter:
            pinecone_filters["bedrooms"] = bedroom_filter
    
    # Bathroom filters
    if filters.min_bathrooms is not None or filters.max_bathrooms is not None:
        bathroom_filter = {}
        if filters.min_bathrooms is not None:
            bathroom_filter["$gte"] = filters.min_bathrooms
        if filters.max_bathrooms is not None:
            bathroom_filter["$lte"] = filters.max_bathrooms
        if bathroom_filter:
            pinecone_filters["bathrooms"] = bathroom_filter
    
    # Price filters
    if filters.min_price is not None or filters.max_price is not None:
        price_filter = {}
        if filters.min_price is not None:
            price_filter["$gte"] = filters.min_price
        if filters.max_price is not None:
            price_filter["$lte"] = filters.max_price
        if price_filter:
            pinecone_filters["price"] = price_filter
    
    # Area filters
    if filters.min_area is not None or filters.max_area is not None:
        area_filter = {}
        if filters.min_area is not None:
            area_filter["$gte"] = filters.min_area
        if filters.max_area is not None:
            area_filter["$lte"] = filters.max_area
        if area_filter:
            pinecone_filters["area"] = area_filter
    
    # Zipcode filter
    if filters.zipcode is not None:
        pinecone_filters["zipcode"] = {"$eq": str(filters.zipcode)}
    
    return pinecone_filters if pinecone_filters else None


def convert_match_to_model(match: Dict[str, Any]) -> HouseMatch:
    """
    Convert a Pinecone match dictionary to HouseMatch model.
    
    Args:
        match: Dictionary from Pinecone query response
        
    Returns:
        HouseMatch model instance
    """
    # Handle both dict-like and object-like matches
    if hasattr(match, 'metadata'):
        metadata_dict = match.metadata
        house_id = match.id
        score = match.score
    else:
        metadata_dict = match.get("metadata", {})
        house_id = match.get("id", "N/A")
        score = match.get("score", 0.0)
    
    # Convert metadata_dict to a regular dict
    if not isinstance(metadata_dict, dict):
        metadata_dict = dict(metadata_dict) if hasattr(metadata_dict, '__dict__') else {}
    
    # Parse images field if it's a JSON string
    if "images" in metadata_dict and isinstance(metadata_dict["images"], str):
        try:
            metadata_dict["images"] = json.loads(metadata_dict["images"])
        except (json.JSONDecodeError, TypeError):
            # If parsing fails, set to None or empty dict
            metadata_dict["images"] = None
    
    return HouseMatch(
        id=str(house_id),
        score=float(score),
        metadata=HouseMetadata(**metadata_dict)
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MLLM House Search API",
        "version": "1.0.0",
        "description": "Semantic search API for finding houses using natural language queries",
        "endpoints": {
            "search": "/api/search",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pinecone_index": settings.pinecone_index,
        "model": settings.model_name
    }


@app.post("/api/search", response_model=SearchResponse)
async def search_houses_endpoint(request: SearchRequest):
    """
    Search for houses using natural language queries.
    
    Example queries:
    - "Modern 3-bedroom house with a large kitchen and backyard"
    - "Affordable family home with 4 bedrooms"
    - "Luxury house with pool and garden"
    """
    try:
        # Build Pinecone filters
        pinecone_filters = build_pinecone_filters(request.filters)
        
        # Perform search
        matches = search_houses(
            query=request.query,
            top_k=request.top_k,
            filters=pinecone_filters
        )
        
        # Convert matches to response models
        house_matches = [convert_match_to_model(match) for match in matches]
        
        return SearchResponse(
            query=request.query,
            results_count=len(house_matches),
            results=house_matches
        )
        
    except Exception as e:
        import traceback
        error_detail = f"Error during search: {str(e)}"
        # Include traceback in detail for debugging (remove in production if needed)
        error_detail += f"\n\nTraceback:\n{traceback.format_exc()}"
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )


@app.get("/api/search", response_model=SearchResponse)
async def search_houses_get(
    query: str = Query(..., min_length=1, description="Natural language search query"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results to return"),
    min_bedrooms: Optional[int] = Query(None, ge=0, description="Minimum bedrooms"),
    max_bedrooms: Optional[int] = Query(None, ge=0, description="Maximum bedrooms"),
    min_bathrooms: Optional[float] = Query(None, ge=0, description="Minimum bathrooms"),
    max_bathrooms: Optional[float] = Query(None, ge=0, description="Maximum bathrooms"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price"),
    min_area: Optional[int] = Query(None, ge=0, description="Minimum area (sqft)"),
    max_area: Optional[int] = Query(None, ge=0, description="Maximum area (sqft)"),
    zipcode: Optional[str] = Query(None, description="Zipcode (exact match)")
):
    """
    Search for houses using natural language queries (GET endpoint).
    Example: /api/search?query=modern+3+bedroom+house&top_k=5
    """
    try:
        # Build filters from query parameters
        filters = None
        if any([min_bedrooms, max_bedrooms, min_bathrooms, max_bathrooms, 
                min_price, max_price, min_area, max_area, zipcode]):
            filters = SearchFilters(
                min_bedrooms=min_bedrooms,
                max_bedrooms=max_bedrooms,
                min_bathrooms=min_bathrooms,
                max_bathrooms=max_bathrooms,
                min_price=min_price,
                max_price=max_price,
                min_area=min_area,
                max_area=max_area,
                zipcode=zipcode
            )
        
        # Build Pinecone filters
        pinecone_filters = build_pinecone_filters(filters)
        
        # Perform search
        matches = search_houses(
            query=query,
            top_k=top_k,
            filters=pinecone_filters
        )
        
        # Convert matches to response models
        house_matches = [convert_match_to_model(match) for match in matches]
        
        return SearchResponse(
            query=query,
            results_count=len(house_matches),
            results=house_matches
        )
        
    except Exception as e:
        import traceback
        error_detail = f"Error during search: {str(e)}"
        # Include traceback in detail for debugging (remove in production if needed)
        error_detail += f"\n\nTraceback:\n{traceback.format_exc()}"
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

