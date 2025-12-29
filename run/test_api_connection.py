"""
Quick diagnostic script to test FastAPI connection and identify issues.
Run this to check if your FastAPI server is configured correctly.
"""

import requests
import json
import sys

API_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("=" * 60)
    print("Testing FastAPI Health Endpoint")
    print("=" * 60)
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed!")
            print(f"   Pinecone Index: {data.get('pinecone_index', 'N/A')}")
            print(f"   Model: {data.get('model', 'N/A')}")
            return True
        else:
            print(f"❌ Health check failed: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {API_URL}")
        print("   Make sure FastAPI server is running!")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_search():
    """Test the search endpoint."""
    print("\n" + "=" * 60)
    print("Testing Search Endpoint")
    print("=" * 60)
    try:
        payload = {
            "query": "modern house",
            "top_k": 3
        }
        print(f"Sending request: {json.dumps(payload, indent=2)}")
        response = requests.post(
            f"{API_URL}/api/search",
            json=payload,
            timeout=30
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Search successful!")
            print(f"   Query: {data.get('query', 'N/A')}")
            print(f"   Results: {data.get('results_count', 0)}")
            if data.get('results'):
                print(f"\n   First result:")
                first = data['results'][0]
                print(f"     ID: {first.get('id', 'N/A')}")
                print(f"     Score: {first.get('score', 0):.4f}")
                print(f"     Bedrooms: {first.get('metadata', {}).get('bedrooms', 'N/A')}")
            return True
        else:
            print(f"❌ Search failed!")
            print(f"   Response: {response.text}")
            try:
                error_json = response.json()
                if 'detail' in error_json:
                    print(f"\n   Error Details:")
                    print(f"   {error_json['detail']}")
            except:
                pass
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {API_URL}")
        print("   Make sure FastAPI server is running!")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    print("\n🔍 FastAPI Connection Diagnostic Tool\n")
    
    health_ok = test_health()
    if not health_ok:
        print("\n⚠️  Health check failed. Fix issues before testing search.")
        sys.exit(1)
    
    search_ok = test_search()
    
    print("\n" + "=" * 60)
    if health_ok and search_ok:
        print("✅ All tests passed! FastAPI is working correctly.")
    else:
        print("❌ Some tests failed. Check the error messages above.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

