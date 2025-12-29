# Setup Guide: FastAPI + Streamlit Integration

Prerequisites

1. **Python 3.11+** installed
2. **Dependencies installed**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment variables configured**:
   Create a `.env` file in the project root with:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX=your_pinecone_index_name
   MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
   DEFAULT_TOP_K=10
   ```

## Step-by-Step Setup

### Option 1: Using Two Terminal Windows (Recommended)

#### Terminal 1: Start FastAPI Server

**Windows:**
```bash
run_api.bat
```

**Or manually:**
```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

**Linux/Mac:**
```bash
chmod +x run_api.sh
./run_api.sh
```

**Or manually:**
```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

**Verify FastAPI is running:**
- Open browser: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

#### Terminal 2: Start Streamlit App

```bash
streamlit run app/streamlit_app.py
```

You should see:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

### Option 2: Using Background Processes (Windows PowerShell)

Run FastAPI in the background:
```powershell
Start-Process powershell -ArgumentList "-NoExit", "-Command", "uvicorn app.api:app --reload --host 0.0.0.0 --port 8000"
```

Then start Streamlit normally:
```bash
streamlit run app/streamlit_app.py
```

## Enabling FastAPI Integration in Streamlit

1. **Open Streamlit** in your browser (usually http://localhost:8501)

2. **In the sidebar**, check the checkbox: **"Use FastAPI Backend"**

3. **Verify API URL** is set to: `http://localhost:8000` (default)

4. **The app will automatically:**
   - Check if FastAPI is available via `/health` endpoint
   - Show a green indicator if connected
   - Show a warning if FastAPI is not available (and fall back to direct search)

## Verification

### Check FastAPI Status

1. **Health Check:**
   ```bash
   curl http://localhost:8000/health
   ```
   Should return:
   ```json
   {
     "status": "healthy",
     "pinecone_index": "your_index_name",
     "model": "sentence-transformers/all-MiniLM-L6-v2"
   }
   ```

2. **Test API Endpoint:**
   ```bash
   curl -X POST "http://localhost:8000/api/search" ^
     -H "Content-Type: application/json" ^
     -d "{\"query\": \"modern house\", \"top_k\": 3}"
   ```

### Check Streamlit Integration

1. In Streamlit sidebar, you should see:
   - "Use FastAPI Backend" checkbox (checked)
   - API URL: `http://localhost:8000`
   - No warning messages about API being unavailable

2. Perform a search - it should use the FastAPI backend

3. Check the browser's Network tab (F12) to see requests to `http://localhost:8000/api/search`

## Troubleshooting

### FastAPI won't start

**Port already in use:**
```bash
# Windows: Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

**Or use a different port:**
```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8001
```
Then update Streamlit API URL to `http://localhost:8001`


### Fallback to Direct Search

If FastAPI is unavailable, Streamlit will:
- Show a warning in the sidebar
- Automatically use direct search (calling `search_houses()` directly)
- Still work perfectly, just without the API layer
