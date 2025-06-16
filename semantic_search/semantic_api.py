"""
Standalone Semantic Search API for Vietnamese Historical Texts
Provides semantic search functionality using PhoBERT embeddings
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sys
import os
from typing import Dict, List

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from semantic_indexer import SemanticIndexer
from config import API_HOST, API_PORT, DEBUG_MODE, DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global semantic indexer instance
semantic_indexer = None

def initialize_semantic_indexer():
    """Initialize the semantic indexer"""
    global semantic_indexer
    try:
        semantic_indexer = SemanticIndexer()
        semantic_indexer.load_indexes()
        logger.info("Semantic indexer initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize semantic indexer: {e}")
        return False

@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Standalone Semantic Search API for Vietnamese Historical Texts',
        'status': 'active',
        'model': 'PhoBERT (vinai/phobert-base)',
        'endpoints': {
            '/api/semantic/search': 'Semantic search endpoint',
            '/api/semantic/status': 'API status endpoint',
            '/api/semantic/stats': 'Index statistics endpoint'
        }
    })

@app.route('/api/semantic/status')
def status():
    """API status endpoint"""
    global semantic_indexer

    if semantic_indexer is None:
        return jsonify({
            'status': 'not_initialized',
            'message': 'Semantic indexer not initialized',
            'version': '1.0.0'
        })

    return jsonify({
        'status': 'ready',
        'message': 'Semantic search API is ready',
        'version': '1.0.0',
        'model': semantic_indexer.model_name,
        'total_documents': len(semantic_indexer.documents),
        'total_embeddings': len(semantic_indexer.embeddings),
        'faiss_index_loaded': semantic_indexer.faiss_index is not None
    })

@app.route('/api/semantic/stats')
def stats():
    """Get index statistics"""
    global semantic_indexer

    if semantic_indexer is None:
        return jsonify({'error': 'Semantic indexer not initialized'}), 500

    return jsonify({
        'total_documents': len(semantic_indexer.documents),
        'total_embeddings': len(semantic_indexer.embeddings),
        'faiss_index_size': semantic_indexer.faiss_index.ntotal if semantic_indexer.faiss_index else 0,
        'model_name': semantic_indexer.model_name,
        'embedding_dimension': len(list(semantic_indexer.embeddings.values())[0]) if semantic_indexer.embeddings else 0
    })

@app.route('/api/semantic/search')
def semantic_search():
    """Semantic search endpoint"""
    global semantic_indexer

    if semantic_indexer is None:
        return jsonify({
            'error': 'Semantic indexer not initialized',
            'message': 'Please ensure semantic indexes are built and loaded'
        }), 500

    # Get query parameters
    query = request.args.get('q', '').strip()
    limit = min(int(request.args.get('limit', DEFAULT_SEARCH_LIMIT)), MAX_SEARCH_LIMIT)

    if not query:
        return jsonify({
            'error': 'Query parameter "q" is required',
            'example': '/api/semantic/search?q=văn hóa truyền thống&limit=10'
        }), 400

    try:
        # Perform semantic search
        results = semantic_indexer.search_similar(query, limit=limit)

        # Format results
        formatted_results = []
        for doc_id, similarity_score in results:
            if doc_id in semantic_indexer.documents:
                doc = semantic_indexer.documents[doc_id]
                formatted_results.append({
                    'id': doc_id,
                    'title': doc.get('title', ''),
                    'content': doc.get('content', '')[:500] + '...' if len(doc.get('content', '')) > 500 else doc.get('content', ''),
                    'url': doc.get('url', ''),
                    'similarity_score': float(similarity_score),
                    'source': 'semantic_search'
                })

        return jsonify({
            'query': query,
            'search_type': 'semantic',
            'status': 'success',
            'documents': formatted_results,
            'total_results': len(formatted_results),
            'limit': limit,
            'model': semantic_indexer.model_name
        })

    except Exception as e:
        logger.error(f"Error during semantic search: {e}")
        return jsonify({
            'error': 'Search failed',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    logger.info("Starting Standalone Semantic Search API")
    logger.info("Initializing semantic indexer...")

    if initialize_semantic_indexer():
        logger.info("Semantic indexer initialized successfully")
        logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
        app.run(debug=DEBUG_MODE, host=API_HOST, port=API_PORT)
    else:
        logger.error("Failed to initialize semantic indexer")
        logger.error("Please build semantic indexes first using: python build_semantic_indexes.py")
        logger.info("Starting API in error mode...")
        app.run(debug=DEBUG_MODE, host=API_HOST, port=API_PORT)
