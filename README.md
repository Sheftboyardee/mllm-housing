# MLLM House Search

A semantic search system for finding houses using natural language queries. Describe what you want in plain English with optional filtering, and get semantically similar properties ranked by relevance.

## Features

- 🏠 **Natural Language Search**: Describe properties in plain English
- 🔍 **Semantic Similarity**: Uses vector embeddings to find semantically similar properties
- 🚀 **FastAPI Backend**: RESTful API for programmatic access
- 💻 **Streamlit Frontend**: Interactive web demo
- 🎯 **Flexible Filtering**: Optional filters for bedrooms, bathrooms, price, area, and zipcode

## How It Works

1. **Query Processing**: Natural language query is converted to a vector embedding using a sentence transformer model
2. **Vector Search**: The embedding is searched against a Pinecone vector database containing house descriptions
3. **Similarity Ranking**: Results are ranked by cosine similarity (semantic similarity)
4. **Filtering**: Optional metadata filters (price, bedrooms, etc.) can be applied
5. **Results**: Top-K most similar houses are returned with their metadata


## Setup

### Prerequisites

- Python 3.11+
- Pinecone account and API key
- Environment variables configured

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Sheftboyardee/mllm-housing
cd mllm-housing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory:
```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=your_pinecone_index_name
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
DEFAULT_TOP_K=10
```

## Usage

### Option 1: Streamlit Demo

Run the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Natural language search interface
- Optional filters (bedrooms, bathrooms, price, area, zipcode)
- Results displayed with similarity scores
- Option to use FastAPI backend or direct search

### Option 2: FastAPI Backend

Start the FastAPI server:
```bash
python app/api.py
```

Or using uvicorn directly:
```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

**API Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Option 3: Use Streamlit with FastAPI Backend

- See run/SETUP_GUIDE.md

## API Examples

### POST /api/search

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Modern 3-bedroom house with a large kitchen and backyard",
    "top_k": 5,
    "filters": {
      "min_bedrooms": 3,
      "max_price": 500000
    }
  }'
```

### GET /api/search

```bash
curl "http://localhost:8000/api/search?query=modern+house+with+pool&top_k=10&min_bedrooms=3"
```

### Response Format

```json
{
  "query": "Modern 3-bedroom house with a large kitchen and backyard",
  "results_count": 5,
  "results": [
    {
      "id": "42",
      "score": 0.85,
      "metadata": {
        "bedrooms": 3,
        "bathrooms": 2.5,
        "area": 2500,
        "price": 450000,
        "zipcode": "85255",
        "description": "Modern kitchen with stainless steel appliances...",
        "images": {
          "frontal": "path/to/image.jpg"
        }
      }
    }
  ]
}
```


## Project Structure

```
mllm-housing/
├── app/
│   ├── api.py              # FastAPI backend
│   ├── streamlit_app.py    # Streamlit frontend
│   └── main.py             # CLI application
├── common/
│   ├── config.py           # Configuration settings
│   ├── embeddings.py       # Embedding model utilities
│   ├── pinecone_client.py  # Pinecone client setup
│   └── search.py           # Search functionality
├── data/
│   ├── descriptions.parquet
│   └── Houses-dataset/
├── image_to_text_pipe/     # Image to description pipeline
├── text_to_embedding_pipe/ # Text to embedding pipeline
└── requirements.txt
```
