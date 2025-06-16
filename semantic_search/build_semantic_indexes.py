#!/usr/bin/env python3
"""
Build standalone semantic indexes using PhoBERT for Vietnamese historical texts
This script loads data from CSV files and generates embeddings with FAISS index
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from semantic_indexer import SemanticIndexer
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['torch', 'transformers', 'faiss-cpu', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.info("Please install them using:")
        logger.info(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def load_documents_from_csv(max_docs=None):
    """Load documents directly from CSV files"""
    try:
        # Initialize semantic indexer and load documents
        indexer = SemanticIndexer()
        documents = indexer.load_documents_from_csv(max_docs=max_docs)
        logger.info(f"Successfully loaded {len(documents)} documents from CSV")
        return documents

    except Exception as e:
        logger.error(f"Error loading documents from CSV: {e}")
        return None

def build_semantic_indexes():
    """Build semantic indexes using PhoBERT"""
    print("🧠 Building Semantic Indexes with PhoBERT")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Check requirements
        if not check_requirements():
            return
        
        # Load documents from CSV
        logger.info("Loading documents from CSV...")
        documents = load_documents_from_csv()
        if documents is None:
            return

        logger.info(f"Loaded {len(documents)} documents")

        # Initialize semantic indexer
        logger.info("Initializing PhoBERT semantic indexer...")
        semantic_indexer = SemanticIndexer()

        # Process documents and generate embeddings
        logger.info("Generating semantic embeddings...")
        logger.info("This may take a while depending on the number of documents and your hardware...")

        semantic_indexer.process_documents(documents)
        
        # Save semantic indexes
        logger.info("Saving semantic indexes...")
        semantic_indexer.save_indexes()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Display results
        print("\n🎯 SEMANTIC INDEX BUILD RESULTS:")
        print("=" * 50)
        print(f"✅ Processing completed in {processing_time:.2f} seconds")
        print(f"📊 Total documents processed: {len(documents)}")
        print(f"🧠 Embeddings generated: {len(semantic_indexer.embeddings)}")
        print(f"🔍 FAISS index built: {semantic_indexer.faiss_index.ntotal if semantic_indexer.faiss_index else 0} vectors")
        print(f"🤖 Model used: {semantic_indexer.model_name}")
        print(f"⚡ Device used: {semantic_indexer.device}")
        
        print("\n✅ Semantic indexes built successfully!")
        print("🚀 You can now use semantic search in the search engine!")
        print("\n💡 Usage examples:")
        print("   - API: GET /api/search?q=your_query&type=semantic")
        print("   - Hybrid search now includes semantic similarity")
        
    except Exception as e:
        logger.error(f"❌ Error during semantic index build: {e}")
        raise

def test_semantic_search():
    """Test the semantic search functionality"""
    print("\n🧪 Testing Semantic Search")
    print("=" * 40)
    
    try:
        # Load semantic indexer
        semantic_indexer = SemanticIndexer()
        semantic_indexer.load_indexes()
        
        # Test queries
        test_queries = [
            "lễ hội truyền thống",
            "văn hóa dân tộc",
            "di tích lịch sử",
            "nghệ thuật múa",
            "ẩm thực địa phương"
        ]
        
        print("Testing with sample queries:")
        for query in test_queries:
            results = semantic_indexer.search_similar(query, limit=3)
            print(f"\n🔍 Query: '{query}'")
            print(f"   Found {len(results)} similar documents")
            for i, (doc_id, score) in enumerate(results[:2], 1):
                print(f"   {i}. Doc {doc_id}: similarity = {score:.3f}")
        
        print("\n✅ Semantic search test completed!")
        
    except Exception as e:
        logger.error(f"❌ Error during semantic search test: {e}")

def show_model_info():
    """Show information about the PhoBERT model"""
    print("\n📋 PhoBERT Model Information")
    print("=" * 40)
    print("Model: vinai/phobert-base")
    print("Language: Vietnamese")
    print("Architecture: RoBERTa-based")
    print("Embedding dimension: 768")
    print("Max sequence length: 256 tokens")
    print("Paper: https://arxiv.org/abs/2003.00744")
    print("\nPhoBERT is specifically trained on Vietnamese text and should")
    print("provide better semantic understanding for Vietnamese documents")
    print("compared to multilingual models.")

if __name__ == "__main__":
    show_model_info()
    build_semantic_indexes()
    
    # Ask user if they want to test
    try:
        test_choice = input("\nWould you like to test semantic search? (y/n): ").strip().lower()
        if test_choice in ['y', 'yes']:
            test_semantic_search()
    except KeyboardInterrupt:
        print("\n\nBuild completed. Exiting...")
