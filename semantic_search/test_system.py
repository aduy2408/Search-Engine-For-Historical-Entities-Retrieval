#!/usr/bin/env python3
"""
Test script for the standalone semantic search system
Tests CSV loading and basic functionality without building full indexes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from csv_loader import CSVDataLoader
from semantic_indexer import SemanticIndexer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_csv_loader():
    """Test CSV data loading"""
    print("🔍 Testing CSV Data Loader")
    print("=" * 50)
    
    try:
        loader = CSVDataLoader()
        
        # Validate CSV structure
        is_valid, errors = loader.validate_csv_structure()
        if not is_valid:
            print("❌ CSV validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("✅ CSV structure validation passed")
        
        # Load a small sample of documents
        documents = loader.load_csv_data(max_docs=5)
        if not documents:
            print("❌ Failed to load documents")
            return False
        
        print(f"✅ Successfully loaded {len(documents)} documents")
        
        # Get statistics
        stats = loader.get_document_stats(documents)
        print(f"📊 Average content length: {stats['avg_content_length']:.1f} characters")
        print(f"📊 Documents with title: {stats['documents_with_title']}")
        print(f"📊 Documents with content: {stats['documents_with_content']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing CSV loader: {e}")
        return False

def test_semantic_indexer_init():
    """Test semantic indexer initialization"""
    print("\n🧠 Testing Semantic Indexer Initialization")
    print("=" * 50)
    
    try:
        indexer = SemanticIndexer()
        print("✅ Semantic indexer initialized successfully")
        print(f"📱 Device: {indexer.device}")
        print(f"🤖 Model: {indexer.model_name}")
        
        # Test CSV loading through indexer
        documents = indexer.load_documents_from_csv(max_docs=3)
        if documents:
            print(f"✅ Loaded {len(documents)} documents through indexer")
            return True
        else:
            print("❌ Failed to load documents through indexer")
            return False
            
    except Exception as e:
        print(f"❌ Error testing semantic indexer: {e}")
        return False

def test_config():
    """Test configuration"""
    print("\n⚙️  Testing Configuration")
    print("=" * 50)
    
    try:
        from config import (
            CSV_DATA_PATH, SEMANTIC_MODEL_NAME, SEMANTIC_INDEX_DIR,
            API_HOST, API_PORT
        )
        
        print(f"✅ CSV Data Path: {CSV_DATA_PATH}")
        print(f"✅ Model Name: {SEMANTIC_MODEL_NAME}")
        print(f"✅ Index Directory: {SEMANTIC_INDEX_DIR}")
        print(f"✅ API Host: {API_HOST}:{API_PORT}")
        
        # Check if CSV file exists
        if os.path.exists(CSV_DATA_PATH):
            print("✅ CSV file exists and is accessible")
        else:
            print("❌ CSV file not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("\n📦 Testing Dependencies")
    print("=" * 50)
    
    dependencies = {
        'torch': 'PyTorch for neural networks',
        'transformers': 'Hugging Face Transformers for PhoBERT',
        'faiss': 'Facebook AI Similarity Search',
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'flask': 'Web API framework'
    }
    
    missing_deps = []
    
    for dep, description in dependencies.items():
        try:
            if dep == 'faiss':
                # Try both faiss-cpu and faiss-gpu
                try:
                    import faiss
                except ImportError:
                    import faiss_cpu as faiss
            else:
                __import__(dep)
            print(f"✅ {dep}: {description}")
        except ImportError:
            print(f"❌ {dep}: {description} - NOT INSTALLED")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    
    print("✅ All dependencies are installed")
    return True

def main():
    """Run all tests"""
    print("🚀 Standalone Semantic Search System Test")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Configuration", test_config),
        ("CSV Loader", test_csv_loader),
        ("Semantic Indexer", test_semantic_indexer_init)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready.")
        print("\nNext steps:")
        print("1. Run 'python build_semantic_indexes.py' to build indexes")
        print("2. Run 'python semantic_api.py' to start the API server")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
