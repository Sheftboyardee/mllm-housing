"""
CLI application for searching houses using natural language queries.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.search import search_houses
from common.config import settings
from common.pinecone_client import get_pinecone_client


def build_filters(args: argparse.Namespace) -> dict | None:
    """
    Build Pinecone filter dictionary from command line arguments.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        Filter dictionary or None if no filters specified
    """
    filters = {}
    
    if args.min_bedrooms is not None:
        filters["bedrooms"] = {"$gte": args.min_bedrooms}
    if args.max_bedrooms is not None:
        if "bedrooms" in filters:
            filters["bedrooms"]["$lte"] = args.max_bedrooms
        else:
            filters["bedrooms"] = {"$lte": args.max_bedrooms}
    
    if args.min_bathrooms is not None:
        filters["bathrooms"] = {"$gte": args.min_bathrooms}
    if args.max_bathrooms is not None:
        if "bathrooms" in filters:
            filters["bathrooms"]["$lte"] = args.max_bathrooms
        else:
            filters["bathrooms"] = {"$lte": args.max_bathrooms}
    
    if args.min_price is not None:
        filters["price"] = {"$gte": args.min_price}
    if args.max_price is not None:
        if "price" in filters:
            filters["price"]["$lte"] = args.max_price
        else:
            filters["price"] = {"$lte": args.max_price}
    
    if args.min_area is not None:
        filters["area"] = {"$gte": args.min_area}
    if args.max_area is not None:
        if "area" in filters:
            filters["area"]["$lte"] = args.max_area
        else:
            filters["area"] = {"$lte": args.max_area}
    
    if args.zipcode is not None:
        filters["zipcode"] = {"$eq": str(args.zipcode)}
    
    return filters if filters else None


def format_price(price: float) -> str:
    """Format price as currency string."""
    return f"${int(price):,}"


def pretty_print_results(results: list, top_k: int):
    """
    Print in format:
    #1: House ID 42 | Similarity: 0.82
        3 bed, 2 bath, 2500 sqft, $450000
        Modern kitchen with stainless steel appliances...
    """
    if not results:
        print("No results found.")
        return
    
    print(f"\nFound {len(results)} result(s):\n")
    
    for idx, match in enumerate(results, 1):
        house_id = match["id"]
        score = match["score"]
        metadata = match.get("metadata", {})
        
        bedrooms = int(metadata.get("bedrooms", 0))
        bathrooms = int(metadata.get("bathrooms", 0))
        area = int(metadata.get("area", 0))
        price = float(metadata.get("price", 0))
        description = metadata.get("description", "No description available")

        print(f"#{idx}: House ID {house_id} | Similarity: {score:.2f}")
        print(f"    {bedrooms} bed, {bathrooms} bath, {area} sqft, {format_price(price)}")
        print(f"    {description}\n")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Search for houses using natural language queries",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Pinecone index
    parser.add_argument(
        "--pinecone_index",
        type=str,
        default=None,
        help="Pinecone index name (overrides PINECONE_INDEX env var)"
    )
    
    # Top K
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        dest="top_k",
        help=f"Number of results to return (default: {settings.default_top_k})"
    )
    
    parser.add_argument(
        "--min-bedrooms",
        type=int,
        default=None,
        dest="min_bedrooms",
        help="Minimum number of bedrooms"
    )
    parser.add_argument(
        "--max-bedrooms",
        type=int,
        default=None,
        dest="max_bedrooms",
        help="Maximum number of bedrooms"
    )
    
    parser.add_argument(
        "--min-bathrooms",
        type=float,
        default=None,
        dest="min_bathrooms",
        help="Minimum number of bathrooms"
    )
    parser.add_argument(
        "--max-bathrooms",
        type=float,
        default=None,
        dest="max_bathrooms",
        help="Maximum number of bathrooms"
    )
    
    parser.add_argument(
        "--min-price",
        type=float,
        default=None,
        dest="min_price",
        help="Minimum price"
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=None,
        dest="max_price",
        help="Maximum price"
    )
    
    parser.add_argument(
        "--min-area",
        type=int,
        default=None,
        dest="min_area",
        help="Minimum area in sqft"
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=None,
        dest="max_area",
        help="Maximum area in sqft"
    )
    
    parser.add_argument(
        "--zipcode",
        type=str,
        default=None,
        help="Zipcode (exact match)"
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Override pinecone_index if provided
    if args.pinecone_index:
        settings.pinecone_index = args.pinecone_index
        # Reset cached index so it uses new index name
        import common.pinecone_client
        common.pinecone_client._index = None
        print(f"Using Pinecone index: {args.pinecone_index}")
    
    filters = build_filters(args)
    
    query = input("Describe the house you want: ").strip()
    
    if not query:
        print("Error: Query cannot be empty.")
        return
    
    print(f"\nSearching for: '{query}'...")
    if filters:
        print(f"Filters: {filters}")
    
    try:
        results = search_houses(
            query=query,
            top_k=args.top_k,
            filters=filters
        )
        
        pretty_print_results(results, args.top_k or settings.default_top_k)
        
    except Exception as e:
        print(f"Error during search: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

