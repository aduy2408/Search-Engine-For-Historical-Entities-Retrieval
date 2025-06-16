#!/usr/bin/env python3
"""
Test API Performance - Test the Flask API performance
"""

import requests
import time
import json
import sys

def test_api_performance():
    """Test the API performance"""
    base_url = "http://localhost:5001"
    
    print("🌐 Testing API Performance")
    print("=" * 40)
    
    # Test if API is running
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not running. Start it with:")
            print("   python search_engine/api.py")
            return
    except requests.exceptions.RequestException:
        print("❌ API is not running. Start it with:")
        print("   python search_engine/api.py")
        return
    
    print("✅ API is running")
    print()
    
    # Test queries
    test_queries = [
        ("Hồ Chí Minh", "entity"),
        ("chiến tranh", "text"),
        ("Nguyễn", "hybrid"),
        ("Hà Nội", "entity"),
        ("cách mạng", "text"),
    ]
    
    print("Testing API query performance:")
    print("-" * 40)
    
    total_time = 0
    query_count = 0
    
    for query, search_type in test_queries:
        print(f"Query: '{query}' ({search_type})")
        
        start_time = time.time()
        
        try:
            response = requests.get(
                f"{base_url}/api/search",
                params={
                    'q': query,
                    'type': search_type,
                    'limit': 10
                },
                timeout=30
            )
            
            end_time = time.time()
            query_time = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ⏱️  {query_time:.0f}ms | {data.get('total_results', 0)} results")
                
                total_time += query_time
                query_count += 1
                
                # Performance assessment
                if query_time < 200:
                    print("  🎉 EXCELLENT")
                elif query_time < 500:
                    print("  ✅ GOOD")
                elif query_time < 1000:
                    print("  ⚠️  ACCEPTABLE")
                else:
                    print("  ❌ SLOW")
            else:
                print(f"  ❌ ERROR: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print("  ❌ TIMEOUT")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
        
        print()
    
    # Overall assessment
    if query_count > 0:
        avg_time = total_time / query_count
        print("=" * 40)
        print(f"📊 Average API response time: {avg_time:.0f}ms")
        
        if avg_time < 300:
            print("🎯 EXCELLENT: API is very responsive!")
        elif avg_time < 600:
            print("✅ GOOD: API performance is acceptable")
        elif avg_time < 1000:
            print("⚠️  ACCEPTABLE: API could be faster")
        else:
            print("❌ SLOW: API needs optimization")
    
    # Test cache performance
    print("\n🔄 Testing cache performance:")
    print("-" * 40)
    
    # First request (cache miss)
    start_time = time.time()
    response = requests.get(
        f"{base_url}/api/search",
        params={'q': 'Hồ Chí Minh', 'type': 'entity', 'limit': 10}
    )
    first_time = (time.time() - start_time) * 1000
    
    # Second request (cache hit)
    start_time = time.time()
    response = requests.get(
        f"{base_url}/api/search",
        params={'q': 'Hồ Chí Minh', 'type': 'entity', 'limit': 10}
    )
    second_time = (time.time() - start_time) * 1000
    
    print(f"First request (cache miss): {first_time:.0f}ms")
    print(f"Second request (cache hit): {second_time:.0f}ms")
    
    if second_time < first_time * 0.5:
        print("🎉 Cache is working effectively!")
    elif second_time < first_time * 0.8:
        print("✅ Cache provides some benefit")
    else:
        print("⚠️  Cache may not be working optimally")

def test_concurrent_api():
    """Test concurrent API performance"""
    import threading
    import statistics
    
    print("\n🔄 Testing concurrent API performance:")
    print("-" * 40)
    
    base_url = "http://localhost:5001"
    results = []
    results_lock = threading.Lock()
    
    def worker_request(thread_id):
        """Worker function for concurrent API testing"""
        try:
            start_time = time.time()
            response = requests.get(
                f"{base_url}/api/search",
                params={'q': 'Hồ Chí Minh', 'type': 'entity', 'limit': 10},
                timeout=10
            )
            end_time = time.time()
            
            request_time = (end_time - start_time) * 1000
            
            with results_lock:
                results.append(request_time)
                
        except Exception as e:
            print(f"Thread {thread_id} error: {e}")
    
    # Test with 3 concurrent requests
    num_threads = 3
    start_time = time.time()
    
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=worker_request, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join(timeout=15)
    
    total_time = time.time() - start_time
    
    if results:
        avg_time = statistics.mean(results)
        print(f"Concurrent requests: {len(results)}/{num_threads}")
        print(f"Average response time: {avg_time:.0f}ms")
        print(f"Total time: {total_time:.2f}s")
        
        if avg_time < 500:
            print("🎉 Excellent concurrent performance!")
        elif avg_time < 1000:
            print("✅ Good concurrent performance")
        else:
            print("⚠️  Concurrent performance could be better")
    else:
        print("❌ No concurrent requests completed successfully")

if __name__ == "__main__":
    try:
        test_api_performance()
        test_concurrent_api()
        
        print("\n" + "=" * 40)
        print("🏁 API Performance Test Complete!")
        print("\nTo start the API server:")
        print("  python search_engine/api.py")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
