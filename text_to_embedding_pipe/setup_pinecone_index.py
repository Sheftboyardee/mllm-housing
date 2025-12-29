"""
Helper script to create and configure the Pinecone index.
Run this before uploading embeddings to ensure the index exists with correct settings.
"""

import sys
from pathlib import Path

# Add parent directory to path to import common modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config import settings
from common.pinecone_client import pc


def create_index_if_not_exists(
    index_name: str = None,
    dimension: int = 384,  # all-MiniLM-L6-v2 produces 384-dim embeddings
    metric: str = "cosine"
):
    """
    Create Pinecone index if it doesn't exist.
    
    Args:
        index_name: Name of the index (defaults to settings.pinecone_index)
        dimension: Vector dimension (384 for all-MiniLM-L6-v2)
        metric: Similarity metric (cosine, euclidean, or dotproduct)
    """
    if index_name is None:
        index_name = settings.pinecone_index
    
    print(f"Checking for Pinecone index: {index_name}")
    
    # List existing indexes
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name in existing_indexes:
        print(f"Index '{index_name}' already exists")
        
        # Get index info to verify dimensions
        index_info = pc.describe_index(index_name)
        print(f"Dimension: {index_info.dimension}")
        print(f"Metric: {index_info.metric}")
        
        if index_info.dimension != dimension:
            print(f"Warning: Index dimension ({index_info.dimension}) doesn't match expected ({dimension})")
            print(f"You may need to recreate the index or use a different embedding model")
        
        return True
    else:
        print(f"Creating new index '{index_name}'...")
        print(f"Dimension: {dimension}")
        print(f"Metric: {metric}")
        
        try:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric
            )
            print(f"Successfully created index '{index_name}'")
            return True
        except Exception as e:
            print(f"Error creating index: {str(e)}")
            return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Pinecone index for house embeddings")
    parser.add_argument(
        "--index-name",
        type=str,
        default=None,
        help="Name of the Pinecone index (defaults to PINECONE_INDEX from .env)"
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=384,
        help="Vector dimension (384 for all-MiniLM-L6-v2, default: 384)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean", "dotproduct"],
        help="Similarity metric (default: cosine)"
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not settings.pinecone_api_key:
        print("Error: PINECONE_API_KEY not found in environment variables")
        print("Please set it in your .env file or environment")
        return
    
    success = create_index_if_not_exists(
        index_name=args.index_name,
        dimension=args.dimension,
        metric=args.metric
    )
    
    if success:
        print("\nIndex setup complete!")
        print("You can now run: python text_to_embedding_pipe/main.py")
    else:
        print("\nIndex setup failed. Check your Pinecone credentials.")


if __name__ == "__main__":
    main()

