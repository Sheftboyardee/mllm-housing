# Text to Embedding Pipeline

Complete guide to convert house descriptions to embeddings and upload to Pinecone.

## Prerequisites

1. **Environment Setup**
   - Create a `.env` file in the project root with:
     ```
     PINECONE_API_KEY=your_api_key_here
     PINECONE_INDEX=house-embeddings
     MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
     DEFAULT_TOP_K=10
     ```

2. **Dependencies**
   - All required packages should be in `requirements.txt`
   - Install with: `pip install -r requirements.txt`

## Complete Workflow

### Step 1: Generate House Descriptions (if not done)

If you haven't generated descriptions yet:
```bash
python image_to_text_pipe/generate_descriptions.py
```

This creates: `data/Houses-dataset/house_descriptions.json`

### Step 2: Convert JSON to Parquet

Convert the JSON descriptions to parquet format:
```bash
python text_to_embedding_pipe/convert_json_to_parquet.py
```

This creates: `data/descriptions.parquet`

**Optional:** Specify custom paths:
```bash
python text_to_embedding_pipe/convert_json_to_parquet.py path/to/input.json path/to/output.parquet
```

### Step 3: Create Pinecone Index (if needed)

Create the Pinecone index with correct dimensions:
```bash
python text_to_embedding_pipe/setup_pinecone_index.py
```

**Optional:** Customize index settings:
```bash
python text_to_embedding_pipe/setup_pinecone_index.py --index-name my-index --dimension 384 --metric cosine
```

**Note:** 
- Default dimension is 384 (for `all-MiniLM-L6-v2`)
- If using a different embedding model, adjust `--dimension` accordingly
- The script will check if the index exists and verify dimensions

### Step 4: Generate Embeddings and Upload to Pinecone

Run the main pipeline:
```bash
python text_to_embedding_pipe/main.py
```

This will:
1. Read `data/descriptions.parquet`
2. Create `full_text` by combining metadata + description
3. Generate embeddings using the configured model
4. Upload vectors to Pinecone in batches of 100
5. Save local copy to `data/embeddings.parquet`

## File Status

### ✅ `convert_json_to_parquet.py` - COMPLETE
- Converts JSON to parquet format
- Handles metadata extraction
- Includes image paths
- Error handling for missing files

### ✅ `main.py` - COMPLETE
- Reads parquet file
- Creates full_text from metadata + description
- Generates embeddings
- Prepares vectors with metadata
- Upserts to Pinecone in batches
- Saves local backup
- Error handling and validation

### ✅ `setup_pinecone_index.py` - NEW
- Creates Pinecone index if it doesn't exist
- Verifies index dimensions match embedding model
- Handles existing indexes gracefully

## Troubleshooting

### "Index not found" error
- Run `setup_pinecone_index.py` first to create the index
- Verify `PINECONE_INDEX` in `.env` matches the index name

### "Dimension mismatch" error
- Check that your embedding model dimension matches the index
- `all-MiniLM-L6-v2` = 384 dimensions
- Recreate index with correct dimension if needed

### "Missing required columns" error
- Ensure `convert_json_to_parquet.py` ran successfully
- Check that `house_descriptions.json` has all required fields

### "API key not found" error
- Verify `.env` file exists in project root
- Check `PINECONE_API_KEY` is set correctly
- Restart your terminal/IDE after creating `.env`

## Verification

After uploading, verify data in Pinecone:

1. **Check Pinecone Dashboard**
   - Log into Pinecone console
   - Verify index has vectors
   - Check metadata fields

2. **Test Search**
   ```bash
   python app/main.py --pinecone_index house-embeddings
   ```

3. **Check Local Backup**
   - `data/embeddings.parquet` contains all embeddings locally
   - Can be used for debugging or re-uploading

## Quick Start (All Steps)

```bash
# 1. Generate descriptions (if needed)
python image_to_text_pipe/generate_descriptions.py

# 2. Convert to parquet
python text_to_embedding_pipe/convert_json_to_parquet.py

# 3. Create Pinecone index
python text_to_embedding_pipe/setup_pinecone_index.py

# 4. Upload to Pinecone
python text_to_embedding_pipe/main.py
```

## Notes

- **Batch Size**: Default is 100 vectors per batch (configurable in code)
- **Embedding Model**: Uses `sentence-transformers/all-MiniLM-L6-v2` by default (384 dimensions)
- **Metadata**: Each vector includes: description, bedrooms, bathrooms, area, zipcode, price, images
- **Resume**: If upload fails, you can re-run `main.py` - it will re-upload all vectors (Pinecone handles duplicates)

