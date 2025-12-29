"""
Text to Embedding Pipeline
Converts house descriptions into embeddings and uploads to Pinecone.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path to import common modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.embeddings import embed_texts
from common.pinecone_client import index


def read_descriptions(parquet_path: str) -> pd.DataFrame:
    """Read descriptions from parquet file."""
    print(f"Reading descriptions from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} houses")
    return df


def create_full_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine description and metadata into one string per house.
    
    Creates a full_text column with format:
    "House with X bedrooms, Y bathrooms, Z sqft, zipcode W. [description]"
    """
    print("Creating full_text from description and metadata...")
    
    # Fill NaN values with defaults
    df = df.copy()
    df["bedrooms"] = df["bedrooms"].fillna(0)
    df["bathrooms"] = df["bathrooms"].fillna(0)
    df["area"] = df["area"].fillna(0)
    df["zipcode"] = df["zipcode"].fillna("")
    df["description"] = df["description"].fillna("")
    
    df["full_text"] = (
        "House with "
        + df["bedrooms"].astype(str) + " bedrooms, "
        + df["bathrooms"].astype(str) + " bathrooms, "
        + df["area"].astype(str) + " sqft, zipcode "
        + df["zipcode"].astype(str)
        + ". " + df["description"]
    )
    
    return df


def compute_embeddings(texts: List[str]) -> np.ndarray:
    """Compute embeddings for the given texts."""
    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = embed_texts(texts)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings


def prepare_pinecone_vectors(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    include_image_paths: bool = True
) -> List[Dict[str, Any]]:
    """
    Prepare vectors for Pinecone upsert.
    
    Args:
        df: DataFrame with house data
        embeddings: Numpy array of embeddings
        include_image_paths: Whether to include image paths in metadata
    
    Returns:
        List of vector dictionaries ready for Pinecone
    """
    print("Preparing vectors for Pinecone...")
    
    vectors = []
    for i, row in df.iterrows():
        vector = {
            "id": str(row["house_id"]),
            "values": embeddings[i].tolist(),
            "metadata": {
                "description": str(row["description"]),
                "bedrooms": int(row["bedrooms"]),
                "bathrooms": int(row["bathrooms"]),
                "area": int(row["area"]),
                "zipcode": str(row["zipcode"]),
                "price": float(row["price"])
            }
        }
        
        # Pinecone metadata only accepts string, number, boolean, or list, thus convert images dict to JSON
        if include_image_paths and "images" in row and pd.notna(row["images"]):
            if isinstance(row["images"], dict):
                vector["metadata"]["images"] = json.dumps(row["images"])
            elif isinstance(row["images"], str):
                # Already a string, use as-is
                vector["metadata"]["images"] = row["images"]
        
        vectors.append(vector)
    
    print(f"Prepared {len(vectors)} vectors")
    return vectors


def upsert_to_pinecone(vectors: List[Dict[str, Any]], batch_size: int = 100):
    """
    Upsert vectors to Pinecone in batches.
    
    Args:
        vectors: List of vector dictionaries
        batch_size: Number of vectors to upsert per batch
    """
    print(f"Upserting {len(vectors)} vectors to Pinecone in batches of {batch_size}...")
    
    total_batches = (len(vectors) + batch_size - 1) // batch_size
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        try:
            index.upsert(vectors=batch)
            print(f"Upserted batch {batch_num}/{total_batches} ({len(batch)} vectors)")
        except Exception as e:
            print(f"Error upserting batch {batch_num}: {str(e)}")
            raise
    
    print(f"Successfully upserted all {len(vectors)} vectors to Pinecone")


def save_embeddings_local(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    output_path: str
):
    """
    Save embeddings and metadata to local parquet file for debugging.
    
    Args:
        df: DataFrame with house data
        embeddings: Numpy array of embeddings
        output_path: Path to save the parquet file
    """
    print(f"Saving embeddings to {output_path}...")
    
    df_output = df.copy()
    
    # Add embeddings as a list column
    df_output["embedding"] = [emb.tolist() for emb in embeddings]
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to parquet
    df_output.to_parquet(output_path, index=False)
    print(f"Saved embeddings to {output_path}")


def main():
    """Main function to run the text-to-embedding pipeline."""
    # Paths
    base_path = Path(__file__).parent.parent
    descriptions_path = base_path / "data" / "descriptions.parquet"
    embeddings_path = base_path / "data" / "embeddings.parquet"
    
    # Check if descriptions file exists
    if not descriptions_path.exists():
        print(f"Error: Descriptions file not found at {descriptions_path}")
        print("Please ensure descriptions.parquet exists or convert from JSON first.")
        return
    
    df = read_descriptions(str(descriptions_path))
    
    # Validate required columns
    required_cols = ["house_id", "bedrooms", "bathrooms", "area", "zipcode", "price", "description"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return
    
    df = create_full_text(df)
    
    # Putting it all together
    embeddings = compute_embeddings(df["full_text"].tolist())
    vectors = prepare_pinecone_vectors(df, embeddings)
    upsert_to_pinecone(vectors)
    save_embeddings_local(df, embeddings, str(embeddings_path))
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()

