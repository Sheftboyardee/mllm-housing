import json
import pandas as pd
from pathlib import Path
import sys
import os


def convert_json_to_parquet(json_path: str, parquet_path: str):
    """  
    Args:
        json_path: Path to JSON file with house descriptions
        parquet_path: Path to save parquet file
    """
    print(f"Reading JSON from {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} houses")
    
    # Flatten data
    records = []
    for house in data:
        record = {
            "house_id": house["house_id"],
            "description": house.get("description", ""),
        }
        
        if "metadata" in house:
            metadata = house["metadata"]
            record["bedrooms"] = metadata.get("bedrooms", 0)
            record["bathrooms"] = metadata.get("bathrooms", 0)
            record["area"] = metadata.get("area", 0)
            record["zipcode"] = metadata.get("zipcode", "")
            record["price"] = metadata.get("price", 0.0)
        else:
            record["bedrooms"] = 0
            record["bathrooms"] = 0
            record["area"] = 0
            record["zipcode"] = ""
            record["price"] = 0.0
        
        if "images" in house:
            record["images"] = house["images"]
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    output_dir = os.path.dirname(parquet_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving to {parquet_path}...")
    df.to_parquet(parquet_path, index=False)
    print(f"Converted {len(df)} houses to parquet format")
    print(f"Saved to {parquet_path}")


def main():
    base_path = Path(__file__).parent.parent
    
    json_path = base_path / "data" / "Houses-dataset" / "house_descriptions.json"
    parquet_path = base_path / "data" / "descriptions.parquet"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        parquet_path = Path(sys.argv[2])
    
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        return
    
    convert_json_to_parquet(str(json_path), str(parquet_path))


if __name__ == "__main__":
    main()

