#!/bin/bash
# Script to run the FastAPI server

echo "Starting FastAPI server..."
echo "API will be available at http://localhost:8000"
echo "API docs at http://localhost:8000/docs"
echo ""
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

