"""
Simple API Testing Script
Test your deployed RAG API
"""

import requests
import json
import time

def test_api(base_url):
    """Test all API endpoints"""
    
    print("\n" + "="*70)
    print(f"🧪 Testing API at: {base_url}")
    print("="*70)
    
    # Test 1: Root endpoint
    print("\n1️⃣ Testing ROOT endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Root endpoint works!")
            print(f"   Response: {response.json()['message']}")
        else:
            print(f"❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Health check
    print("\n2️⃣ Testing HEALTH endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed!")
            print(f"   Status: {data['status']}")
            print(f"   Chunks loaded: {data['chunks_loaded']}")
            print(f"   Model: {data['embedding_model']}")
        else:
            print(f"❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Query endpoint
    print("\n3️⃣ Testing QUERY endpoint...")
    try:
        start = time.time()
        response = requests.post(
            f"{base_url}/query",
            json={
                "question": "What is MetaGPT?",
                "top_k": 3,
                "use_cache": True
            },
            timeout=30
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Query works!")
            print(f"   Time: {elapsed:.2f}s")
            print(f"   Answer preview: {data['answer'][:100]}...")
            print(f"   Sources: {len(data['sources'])}")
            print(f"   Cached: {data.get('cached', False)}")
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"   {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Cache stats
    print("\n4️⃣ Testing CACHE STATS...")
    try:
        response = requests.get(f"{base_url}/cache/stats")
        if response.status_code == 200:
            data = response.json()
            print("✅ Cache stats work!")
            print(f"   Hits: {data['hits']}")
            print(f"   Misses: {data['misses']}")
            print(f"   Hit rate: {data['hit_rate']}")
        else:
            print(f"❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 5: Streaming (optional - just check endpoint exists)
    print("\n5️⃣ Testing STREAMING endpoint availability...")
    try:
        response = requests.post(
            f"{base_url}/query/stream",
            json={"question": "What is MetaGPT?", "top_k": 2},
            stream=True,
            timeout=30
        )
        if response.status_code == 200:
            print("✅ Streaming endpoint works!")
            # Read first few chunks
            chunks = 0
            for chunk in response.iter_content(chunk_size=None):
                chunks += 1
                if chunks >= 3:
                    break
            print(f"   Received {chunks}+ chunks")
        else:
            print(f"❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*70)
    print("🎉 Testing complete!")
    print("="*70)
    print("\n📖 Full API docs available at: {}/docs".format(base_url))


if __name__ == "__main__":
    print("\n🚀 RAG API Tester")
    print("="*70)
    
    # Test local first
    print("\n📍 Testing LOCAL server...")
    test_api("http://localhost:8000")
    
    # Ask for deployed URL
    print("\n" + "="*70)
    deployed = input("\n📝 Enter your DEPLOYED URL (or press Enter to skip): ").strip()
    
    if deployed:
        # Remove trailing slash if present
        deployed = deployed.rstrip('/')
        print(f"\n📍 Testing DEPLOYED server...")
        test_api(deployed)
    else:
        print("\n⏭️  Skipping deployed URL test")
    
    print("\n✅ All done! Check the results above.")
    print("\nNext steps:")
    print("1. If local tests pass ✅ - Push to GitHub")
    print("2. Deploy to Render/Railway")
    print("3. Test with deployed URL")
    print("4. Share your live API URL! 🌍\n")